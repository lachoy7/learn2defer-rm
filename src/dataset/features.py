#!/usr/bin/env python3
"""
Extract scalar features from scalar_dataset (rmbench, rewardbench).
Write one JSON per dataset with min/max/mean/variance of chosen_score and
rejected_score per sample_id, plus token counts per model.

Optional: add 32-dim PCA embeddings from ModernBERT (encoder backbone from
ModernBertForQuestionAnswering) for prompt, response1 (chosen), response2 (rejected).
Run locally (--embeddings --device cuda) or on Modal GPU (--embeddings --modal).
See https://modal.com/docs/guide/apps and https://huggingface.co/docs/transformers/en/model_doc/modernbert
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np

from utils.paths import SCALAR_DATASET_DIR, SCALAR_FEATURES_DIR

SCALAR_ROOT = SCALAR_DATASET_DIR
FEATURES_ROOT = SCALAR_FEATURES_DIR
DATASETS = ("rmbench", "rewardbench", "skywork")  # skip skywork
MODERNBERT_MODEL_ID = "answerdotai/ModernBERT-base"
PCA_DIMS = 32
EMBED_BATCH_SIZE = 32
MAX_LENGTH = 512

try:
    import modal
    _modal_available = True
except ImportError:
    modal = None  # type: ignore
    _modal_available = False


# ----------------------------- Core feature logic -----------------------------


def variance(values: list[float], ddof: int = 0) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.var(values, ddof=ddof))


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _load_modernbert_encoder(device: str = "cuda"):
    """Load ModernBERT encoder (same backbone as ModernBertForQuestionAnswering) for embeddings."""
    from transformers import AutoModel, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL_ID)
    # Use base model for embeddings; ForQuestionAnswering adds a span head we don't need
    model = AutoModel.from_pretrained(MODERNBERT_MODEL_ID)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _embed_texts_batched(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = EMBED_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Mean-pool last hidden state over non-padding tokens; return (n, hidden_size)."""
    import torch
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # last_hidden_state: (batch, seq_len, hidden_size)
        mask = enc["attention_mask"]
        hidden = out.last_hidden_state  # (B, L, D)
        mask_expand = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expand).sum(dim=1) / mask_expand.sum(dim=1).clamp(min=1e-9)
        all_embeds.append(pooled.cpu().numpy())
    return np.vstack(all_embeds)


def compute_embeddings_and_pca(
    sample_texts: list[tuple[str, str, str]],
    device: str = "cuda",
    pca_dims: int = PCA_DIMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    For each (prompt, chosen, rejected) get ModernBERT embedding then PCA to pca_dims.
    Returns (prompt_emb_32, chosen_emb_32, rejected_emb_32, pca_fit).
    """
    from sklearn.decomposition import PCA

    model, tokenizer, device = _load_modernbert_encoder(device)
    prompts = [t[0] for t in sample_texts]
    chosen = [t[1] for t in sample_texts]
    rejected = [t[2] for t in sample_texts]

    prompt_emb = _embed_texts_batched(prompts, model, tokenizer, device)
    chosen_emb = _embed_texts_batched(chosen, model, tokenizer, device)
    rejected_emb = _embed_texts_batched(rejected, model, tokenizer, device)

    # Fit PCA on all embeddings together for a common basis
    combined = np.vstack([prompt_emb, chosen_emb, rejected_emb])
    pca = PCA(n_components=min(pca_dims, combined.shape[0], combined.shape[1]))
    pca.fit(combined)

    prompt_32 = pca.transform(prompt_emb)
    chosen_32 = pca.transform(chosen_emb)
    rejected_32 = pca.transform(rejected_emb)
    return prompt_32, chosen_32, rejected_32, pca


def add_embeddings_to_features(
    all_features: dict[str, dict],
    by_sample: dict[str, list[dict]],
    dataset_name: str,
    device: str = "cuda",
) -> None:
    """Compute ModernBERT + PCA embeddings and add to all_features in-place."""
    sample_ids = list(by_sample.keys())
    sample_texts = []
    for sid in sample_ids:
        r = by_sample[sid][0]
        sample_texts.append(
            (_norm_text(r.get("prompt")), _norm_text(r.get("chosen")), _norm_text(r.get("rejected")))
        )

    prompt_32, chosen_32, rejected_32, _ = compute_embeddings_and_pca(sample_texts, device=device)

    for i, sid in enumerate(sample_ids):
        all_features[sid]["prompt_embedding_32d"] = prompt_32[i].tolist()
        all_features[sid]["response1_embedding_32d"] = chosen_32[i].tolist()   # chosen
        all_features[sid]["response2_embedding_32d"] = rejected_32[i].tolist()  # rejected


def _merge_remote_embeddings(
    all_features: dict[str, dict],
    sample_ids: list[str],
    prompt_32: list[list[float]],
    chosen_32: list[list[float]],
    rejected_32: list[list[float]],
) -> None:
    """Merge embedding lists (from Modal) into all_features in-place."""
    for i, sid in enumerate(sample_ids):
        all_features[sid]["prompt_embedding_32d"] = prompt_32[i]
        all_features[sid]["response1_embedding_32d"] = chosen_32[i]
        all_features[sid]["response2_embedding_32d"] = rejected_32[i]


# ----------------------------- Modal GPU app -----------------------------

def _make_modal_image():
    """Image for Modal: ModernBERT + sklearn for PCA."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .pip_install(
            "transformers>=4.45.0",
            "torch",
            "scikit-learn>=1.0.0",
            "numpy>=1.26.0",
        )
    )


if _modal_available:

    scalar_features_app = modal.App("scalar-features-embeddings")

    @scalar_features_app.cls(
        image=_make_modal_image(),
        gpu="A100-80GB",
        timeout=3600 * 2,
        scaledown_window=300,
    )
    class ModalEmbeddingComputer:
        """Run ModernBERT + PCA on Modal GPU. Same logic as compute_embeddings_and_pca."""

        @modal.enter()
        def load_model(self):
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._device = "cuda"
            self._tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL_ID)
            self._model = AutoModel.from_pretrained(MODERNBERT_MODEL_ID)
            self._model.to(self._device)
            self._model.eval()

        @modal.method()
        def compute_embeddings_and_pca_remote(
            self,
            sample_texts: list[tuple[str, str, str]],
            pca_dims: int = PCA_DIMS,
            batch_size: int = EMBED_BATCH_SIZE,
            max_length: int = MAX_LENGTH,
        ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
            """
            Compute prompt/chosen/rejected embeddings and PCA on GPU.
            Returns (prompt_32, chosen_32, rejected_32) as list of lists for serialization.
            """
            import numpy as np
            from sklearn.decomposition import PCA
            import torch

            prompts = [t[0] for t in sample_texts]
            chosen = [t[1] for t in sample_texts]
            rejected = [t[2] for t in sample_texts]

            def embed_batch(texts):
                all_embeds = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    enc = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = self._model(**enc)
                    mask = enc["attention_mask"]
                    hidden = out.last_hidden_state
                    mask_expand = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask_expand).sum(dim=1) / mask_expand.sum(dim=1).clamp(min=1e-9)
                    all_embeds.append(pooled.cpu().numpy())
                return np.vstack(all_embeds)

            prompt_emb = embed_batch(prompts)
            chosen_emb = embed_batch(chosen)
            rejected_emb = embed_batch(rejected)

            combined = np.vstack([prompt_emb, chosen_emb, rejected_emb])
            n_components = min(pca_dims, combined.shape[0], combined.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(combined)

            prompt_32 = pca.transform(prompt_emb)
            chosen_32 = pca.transform(chosen_emb)
            rejected_32 = pca.transform(rejected_emb)

            return (
                prompt_32.tolist(),
                chosen_32.tolist(),
                rejected_32.tolist(),
            )


def process_dataset(
    dataset_name: str,
    with_embeddings: bool = False,
    device: str = "cuda",
    modal_embedder: Any = None,
) -> None:
    in_dir = SCALAR_ROOT / dataset_name
    out_dir = FEATURES_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"Skip {dataset_name}: not a directory")
        return

    # Collect all records from every JSON in this dataset
    records: list[dict] = []
    for path in sorted(in_dir.glob("*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not records:
        print(f"No records for {dataset_name}")
        return

    # Group by sample_id (string key so "30" and "chat/100" both work)
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("sample_id")
        if sid is None:
            continue
        by_sample[str(sid)].append(r)

    all_features: dict[str, dict] = {}
    for sample_id, sample_records in by_sample.items():
        chosen_scores = [r["chosen_score"] for r in sample_records if "chosen_score" in r]
        rejected_scores = [r["rejected_score"] for r in sample_records if "rejected_score" in r]

        # Tokens per model: aggregate per (sample_id, model_id)
        tokens_per_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tokens_chosen": 0, "tokens_rejected": 0, "tokens_total": 0}
        )
        for r in sample_records:
            mid = r.get("model_id", "unknown")
            tokens_per_model[mid]["tokens_chosen"] += r.get("tokens_chosen", 0)
            tokens_per_model[mid]["tokens_rejected"] += r.get("tokens_rejected", 0)
            tokens_per_model[mid]["tokens_total"] += r.get("tokens_total", 0)
        tokens_per_model = {k: dict(v) for k, v in tokens_per_model.items()}

        all_features[sample_id] = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "chosen_score": {
                "min": float(np.min(chosen_scores)) if chosen_scores else None,
                "max": float(np.max(chosen_scores)) if chosen_scores else None,
                "mean": safe_mean(chosen_scores),
                "variance": variance(chosen_scores),
            },
            "rejected_score": {
                "min": float(np.min(rejected_scores)) if rejected_scores else None,
                "max": float(np.max(rejected_scores)) if rejected_scores else None,
                "mean": safe_mean(rejected_scores),
                "variance": variance(rejected_scores),
            },
            "tokens_per_model": tokens_per_model,
        }

    if with_embeddings:
        if modal_embedder is not None:
            sample_ids = list(by_sample.keys())
            sample_texts = [
                (
                    _norm_text(by_sample[sid][0].get("prompt")),
                    _norm_text(by_sample[sid][0].get("chosen")),
                    _norm_text(by_sample[sid][0].get("rejected")),
                )
                for sid in sample_ids
            ]
            prompt_32, chosen_32, rejected_32 = modal_embedder.compute_embeddings_and_pca_remote.remote(
                sample_texts, pca_dims=PCA_DIMS
            )
            _merge_remote_embeddings(all_features, sample_ids, prompt_32, chosen_32, rejected_32)
        else:
            add_embeddings_to_features(all_features, by_sample, dataset_name, device=device)

    out_path = out_dir / "features.json"
    with open(out_path, "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"{dataset_name}: wrote 1 feature file with {len(by_sample)} ids to {out_path}")


# ----------------------------- Skywork: add consistent sample_id -----------------------------

def _norm_text(x: Any) -> str:
    """Normalize prompt/chosen/rejected for hashing (skywork uses dict with 'content')."""
    if isinstance(x, dict):
        return x.get("content", "")
    return str(x) if x is not None else ""


def _skywork_sample_id_for_record(record: dict) -> str:
    """Stable sample_id from (prompt, chosen, rejected) so same triple gets same id across models."""
    key = json.dumps(
        {
            "prompt": _norm_text(record.get("prompt")),
            "chosen": _norm_text(record.get("chosen")),
            "rejected": _norm_text(record.get("rejected")),
        },
        sort_keys=True,
    )
    return "skywork_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def add_skywork_sample_ids() -> None:
    """
    For each JSON in scalar_dataset/skywork/, add sample_id to each record that doesn't have one.
    sample_id is derived from (prompt, chosen, rejected) so the same example gets the same id
    across all model files. Only writes when at least one record in a file is missing sample_id.
    """
    skywork_dir = SCALAR_ROOT / "skywork"
    if not skywork_dir.is_dir():
        print("Skywork dir not found:", skywork_dir)
        return

    for path in sorted(skywork_dir.glob("*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"Skip {path}: not a list")
            continue

        missing = [i for i, r in enumerate(data) if "sample_id" not in r]
        if not missing:
            print(f"Skip {path}: all {len(data)} records already have sample_id")
            continue

        for i in missing:
            data[i]["sample_id"] = _skywork_sample_id_for_record(data[i])

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated {path.name}: added sample_id to {len(missing)} records (total {len(data)}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scalar features (optionally with ModernBERT embeddings).")
    parser.add_argument("--embeddings", action="store_true", help="Add 32-dim PCA embeddings (prompt, response1, response2) via ModernBERT.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embedding model when running locally (cuda or cpu).")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASETS), help="Process only this dataset (default: all).")
    parser.add_argument("--modal", action="store_true", help="Run embedding computation on Modal GPU (A100-80GB). Requires --embeddings.")
    parser.add_argument("--add-skywork-sample-ids", action="store_true", help="Add consistent sample_id to skywork scalar JSONs (only where missing).")
    args = parser.parse_args()

    if args.add_skywork_sample_ids:
        add_skywork_sample_ids()
        if not args.embeddings and not args.dataset:
            return

    if args.modal and not args.embeddings:
        parser.error("--modal requires --embeddings")

    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    names = [args.dataset] if args.dataset else list(DATASETS)

    if args.modal and _modal_available:
        embedder = ModalEmbeddingComputer()
        with modal.enable_output():
            with scalar_features_app.run():
                for name in names:
                    process_dataset(
                        name,
                        with_embeddings=True,
                        device=args.device,
                        modal_embedder=embedder,
                    )
    else:
        if args.modal and not _modal_available:
            raise RuntimeError("--modal requested but modal is not installed. pip install modal")
        for name in names:
            process_dataset(
                name,
                with_embeddings=args.embeddings,
                device=args.device,
                modal_embedder=None,
            )


if __name__ == "__main__":
    main()
