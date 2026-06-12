import json
import re
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset

# Default model order — temporarily single model for dev
MODEL_ORDER_DEFAULT = [
    # "scalar_LxzGordon/URM-LLaMa-3.1-8B",
    # "scalar_Ray2333/GRM-Gemma-2B-rewardmodel-ft",
    # "scalar_Ray2333/GRM-Llama3-8B-rewardmodel-ft",
    # "scalar_Skywork/Skywork-Reward-Llama-3.1-8B",
    # "scalar_sfairXC/FsfairX-LLaMA3-RM-v0.1",
    "scalar_weqweasdas/RM-Mistral-7B",
]


def _load_records_any(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file; return list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _as_float_list(x: Any, expected_dim: Optional[int] = None) -> Optional[List[float]]:
    """
    Args:
        x: The input to be processed. Supports list, tuple, or a JSON-formatted list string.
        expected_dim: If set, pad or trim the result to this length.

    Returns:
        List[float] if the input is a list, tuple, or valid JSON list string;
        otherwise, None.
    """
    if x is None:
        return None
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return None
    if isinstance(x, (list, tuple)):
        out = [float(v) for v in x]
        if expected_dim is not None:
            if len(out) >= expected_dim:
                out = out[:expected_dim]
            else:
                out = out + [0.0] * (expected_dim - len(out))
        return out
    return None


def extract_score(text: Any) -> Optional[Any]:
    """
    Extract judge prediction from generative.judge_thinking text.
    Returns 'Response 1', 'Response 2', or None (unparsable).
    Used for swapped/p_D/y computation in network_cont_loss mode.
    """
    if text is None:
        return None
    text = str(text)
    match = re.search(
        r'(?:\\?"score\\?"|score)\s*:\s*'
        r'(?:"(Response 1|Response 2|true|false|True|False|\d+(?:\.\d+)?)"|'
        r'(Response 1|Response 2|true|false|True|False|\d+(?:\.\d+)?))',
        text,
    )
    if match:
        for group in match.groups():
            if group is not None:
                if re.match(r'^\d+\.\d+$', group):
                    return int(float(group))
                if re.match(r'^\d+$', group):
                    return int(group)
                return group
    return None


def get_tokens_judge(rec: Dict[str, Any]) -> float:
    gen = rec.get("generative", {}) or {}
    return float(gen.get("tokens_judge", 0))


def get_meta_train_dev(rec: Dict[str, Any]) -> Dict[str, str]:
    return {
        "source": str(rec.get("source", "")),
    }

def get_meta_test_rmbench(rec: Dict[str, Any]) -> Dict[str, str]:
    return {
        "domain": str(rec.get("domain", "")),
        "difficulty": str(rec.get("difficulty", "")),
    }


class RouterDataset(Dataset):
    """
    return:
    - chosen_scores: (num_models,)  # temporarily 1
    - rejected_scores: (num_models,)
    - tokens_chosen:   (num_models,)
    - tokens_rejected: (num_models,)
    - emb_cat: (2D,)  (chosen_emb || rejected_emb) - D = 16
    - heavy_correct: generative.is_correct
    - tokens_judge:  (1,)
    """
    def __init__(
        self,
        path: str,
        split: str,
        emb_chosen_key: str = "encoding_chosen_16d",
        emb_rejected_key: str = "encoding_rejected_16d",
        stage1_index: int = 0,
        emb_dim: int = 16,
        model_order=None,
        use_swapped_pd: bool = False,
    ):
        self.records = _load_records_any(path)
        self.split = split
        # idx -> record mapping (make sure it's replicable)
        self.records.sort(key=lambda r: int(r.get("idx", 0)))
        self.emb_chosen_key = emb_chosen_key
        self.emb_rejected_key = emb_rejected_key
        self.emb_dim = emb_dim
        self.stage1_index = stage1_index

        # explicitely define model order so that we don't get lost
        self.model_order = model_order
        if model_order is None:
            self.model_order = MODEL_ORDER_DEFAULT
        self.use_swapped_pd = use_swapped_pd

    def _collect_scalar_blocks(self, rec: Dict) -> List[Dict]:
        # out = []
        # for k, v in rec.items():
        #     if k.startswith("scalar_") and isinstance(v, dict) and "model_id" in v:
        #         out.append(v)
        # return out
        return {key: rec[key] for key in self.model_order}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        idx = rec.get("idx")

        # ---- scalar blocks -> (num_models,) scores
        m2b = self._collect_scalar_blocks(rec)

        chosen_scores = torch.tensor(
            [float(m2b.get(mid, {}).get("chosen_score", 0.0)) for mid in self.model_order],
            dtype=torch.float32
        )
        rejected_scores = torch.tensor(
            [float(m2b.get(mid, {}).get("rejected_score", 0.0)) for mid in self.model_order],
            dtype=torch.float32
        )
        tokens_chosen = torch.tensor(
            [float(m2b.get(mid, {}).get("tokens_chosen", 0.0)) for mid in self.model_order],
            dtype=torch.float32
        )
        tokens_rejected = torch.tensor(
            [float(m2b.get(mid, {}).get("tokens_rejected", 0.0)) for mid in self.model_order],
            dtype=torch.float32
        )

        # ---- embeddings
        chosen_emb = _as_float_list(rec.get(self.emb_chosen_key), expected_dim=self.emb_dim)
        rejected_emb = _as_float_list(rec.get(self.emb_rejected_key), expected_dim=self.emb_dim)
        emb_cat = torch.tensor(chosen_emb + rejected_emb, dtype=torch.float32) if (chosen_emb and rejected_emb) else torch.zeros(2 * self.emb_dim, dtype=torch.float32)

        # ---- generative
        gen = rec.get("generative", {}) or {}
        is_correct = float(bool(gen.get("is_correct")))
        loss_D = is_correct

        # Stage I correctness
        stage1_chosen = chosen_scores[self.stage1_index]
        stage1_rejected = rejected_scores[self.stage1_index]
        is_correct_M = torch.tensor(1.0 if stage1_chosen > stage1_rejected else 0.0, dtype=torch.float32)
        is_wrong_M = 1.0 - is_correct_M

        if self.use_swapped_pd:
            prediction = extract_score(gen.get("judge_thinking"))
            swapped = -float(2 * ((prediction == "Response 1" and is_correct == False) or (prediction == "Response 2" and is_correct == True)) - 1)
            if prediction == "Response 1":
                p_D = 0.8
            elif prediction == "Response 2":
                p_D = 0.2
            else:
                p_D = 0.5
            y = float(0.0) if swapped == -1.0 else float(1.0)

        out = {
            "idx": idx,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
            "tokens_chosen": tokens_chosen,
            "tokens_rejected": tokens_rejected,
            "emb_cat": emb_cat,
            "loss_D": torch.tensor(loss_D, dtype=torch.float32),
            "is_correct_M": is_correct_M,
            "is_wrong_M": is_wrong_M,
            "tokens_judge": torch.tensor(float(gen.get("tokens_judge", 0)), dtype=torch.float32),
        }
        if self.use_swapped_pd:
            out["swapped"] = torch.tensor(swapped, dtype=torch.float32)
            out["p_D"] = torch.tensor(p_D, dtype=torch.float32)
            out["y"] = torch.tensor(y, dtype=torch.float32)

        # === META FIELDS FOR GROUPING + DETAILED JSON (CHANGED PART) ===
        if self.split in ("train", "dev"):
            out["source"] = str(rec.get("source", ""))
        elif self.split == "test_rmbench":
            out["domain"] = str(rec.get("domain", ""))
            out["difficulty"] = str(rec.get("difficulty", ""))
            out["chosen_style"] = str(rec.get("chosen_style", ""))
            out["rejected_style"] = str(rec.get("rejected_style", ""))

        return out


def collate_router(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "idx": [b["idx"] for b in batch],
        "chosen_scores": torch.stack([b["chosen_scores"] for b in batch]),
        "rejected_scores": torch.stack([b["rejected_scores"] for b in batch]),
        "tokens_chosen": torch.stack([b["tokens_chosen"] for b in batch]),
        "tokens_rejected": torch.stack([b["tokens_rejected"] for b in batch]),
        "emb_cat": torch.stack([b["emb_cat"] for b in batch]),
        "loss_D": torch.stack([b["loss_D"] for b in batch]),
        "is_correct_M": torch.stack([b["is_correct_M"] for b in batch]),
        "is_wrong_M": torch.stack([b["is_wrong_M"] for b in batch]),
        "tokens_judge": torch.stack([b["tokens_judge"] for b in batch]),
    }

    if "swapped" in batch[0]:
        out["swapped"] = torch.stack([b["swapped"] for b in batch])
        out["p_D"] = torch.stack([b["p_D"] for b in batch])
        out["y"] = torch.stack([b["y"] for b in batch])

    if "source" in batch[0]:
        out["source"] = [b.get("source", "") for b in batch]
    if "domain" in batch[0]:
        out["domain"] = [b.get("domain", "") for b in batch]
    if "difficulty" in batch[0]:
        out["difficulty"] = [b.get("difficulty", "") for b in batch]
    if "chosen_style" in batch[0]:
        out["chosen_style"] = [b.get("chosen_style", "") for b in batch]
    if "rejected_style" in batch[0]:
        out["rejected_style"] = [b.get("rejected_style", "") for b in batch]

    return out