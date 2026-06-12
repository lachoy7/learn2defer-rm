#!/usr/bin/env python3
"""Inspect indexed datasets: response alignment and per-model accuracy."""

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from utils.paths import GENERATIVE_DATASET_DIR
from dataset.router_dataset import MODEL_ORDER_DEFAULT


def load_indexed_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        first = f.read(1)
    with open(path) as f:
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def responses_match(row: dict) -> bool:
    return (
        row.get("chosen_response_scalar") == row.get("chosen_response_generative")
        and row.get("rejected_response_scalar") == row.get("rejected_response_generative")
    )


def inspect_response_alignment(path: Path) -> None:
    rows = load_indexed_dataset(path)
    match_count = mismatch_count = 0
    mismatch_indices = []
    for i, row in enumerate(rows):
        if responses_match(row):
            match_count += 1
        else:
            mismatch_count += 1
            mismatch_indices.append(row.get("idx", i))

    print(f"File: {path}")
    print(f"  Matches: {match_count:,}")
    print(f"  Mismatches: {mismatch_count:,}")
    if mismatch_indices:
        print(f"  First mismatched idx values: {mismatch_indices[:10]}")


def stage1_accuracy(path: Path, split_col: str = "source") -> pd.DataFrame:
    rows = load_indexed_dataset(path)
    records = []
    for row in rows:
        for model_id in MODEL_ORDER_DEFAULT:
            block = row.get(model_id, {})
            records.append({
                "idx": row.get("idx"),
                split_col: row.get(split_col, ""),
                "model_id": model_id,
                "is_correct": block.get("chosen_score", 0) > block.get("rejected_score", 0),
            })
        gen = row.get("generative", {})
        records.append({
            "idx": row.get("idx"),
            split_col: row.get(split_col, ""),
            "model_id": "generative",
            "is_correct": bool(gen.get("is_correct")),
        })
    df = pd.DataFrame(records)
    return (
        df.groupby([split_col, "model_id"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect generative/scalar dataset alignment and accuracy.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(GENERATIVE_DATASET_DIR / "df_train_cleaned_indexed_train.json"),
    )
    parser.add_argument("--accuracy", action="store_true", help="Print per-source/model accuracy table")
    args = parser.parse_args()

    path = Path(args.path)
    inspect_response_alignment(path)
    if args.accuracy:
        print("\nPer-model accuracy:")
        print(stage1_accuracy(path).to_string(index=False))


if __name__ == "__main__":
    main()
