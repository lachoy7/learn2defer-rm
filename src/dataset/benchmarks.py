from datasets import load_dataset
from typing import Iterator, Dict, Optional

def iter_rewardbench(max_samples: Optional[int] = None) -> Iterator[Dict]:
    ds = load_dataset("allenai/reward-bench", split="filtered")
    if max_samples is not None:
        # take max samples randomly
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    for ex in ds:
        yield {
            "id": ex["id"],
            "bench": "rewardbench",
            "bench_split": "filtered",
            "sample_id": int(ex["id"]),
            "subset": ex.get("subset"),
            "prompt": ex["prompt"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

def iter_rmbench(max_samples: Optional[int] = None) -> Iterator[Dict]:
    ds = load_dataset("THU-KEG/RM-Bench", split="train")
    if max_samples is not None:
        # take max samples randomly
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    styles = ["concise", "detailed_plain", "detailed_markdown"]
    for ex in ds:
        for i in range(3):
            for j in range(3):
                diff = "hard" if i < j else "normal" if i == j else "easy"
                yield {
                    "id": f"{ex['id']}-{i}-{j}",
                    "bench": "rmbench",
                    "bench_split": "train",
                    "sample_id": ex["id"],
                    "domain": ex.get("domain"),
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"][i],
                    "rejected": ex["rejected"][j],
                    "chosen_style": styles[i],
                    "rejected_style": styles[j],
                    "difficulty": diff,
                }

def iter_skywork(max_samples: Optional[int] = None) -> Iterator[Dict]:
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train")
    if max_samples is not None:
        # take max samples randomly
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    for ex in ds:
        prompt = ex["chosen"][0]["content"]
        chosen_resp = ex["chosen"][1]["content"]
        rejected_resp = ex["rejected"][1]["content"]
        assert prompt == ex["rejected"][0]["content"]
        yield {
            "bench": "skywork",
            "bench_split": "train",
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        }