#!/usr/bin/env python3
"""Stratified train/dev split for indexed generative datasets."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import _bootstrap  # noqa: F401
from utils.paths import GENERATIVE_DATASET_DIR


def train_dev_split(
    input_file: str | None = None,
    output_dir: str | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    output_dir = output_dir or str(GENERATIVE_DATASET_DIR)
    input_file = input_file or os.path.join(output_dir, "df_train_cleaned_indexed.json")
    train_output = os.path.join(output_dir, "df_train_cleaned_indexed_train.json")
    dev_output = os.path.join(output_dir, "df_train_cleaned_indexed_dev.json")

    df_split = pd.read_json(input_file, orient="columns")[["idx", "source"]]
    idx_source = df_split.groupby("idx")["source"].first().reset_index()
    train_idx, dev_idx = train_test_split(
        idx_source["idx"],
        test_size=test_size,
        stratify=idx_source["source"],
        random_state=random_seed,
    )

    print(f"Train idx: {len(train_idx):,} | Dev idx: {len(dev_idx):,}")
    os.makedirs(output_dir, exist_ok=True)
    full_df = pd.read_json(input_file, orient="columns")

    def write_filtered(output_path, idx_list):
        chunk_size = 50_000
        first_write = True
        for i in range(0, len(full_df), chunk_size):
            chunk = full_df.iloc[i : i + chunk_size]
            filtered = chunk[chunk["idx"].isin(idx_list)]
            if len(filtered) == 0:
                continue
            mode = "w" if first_write else "a"
            filtered.to_json(output_path, orient="records", lines=True, force_ascii=False, mode=mode)
            first_write = False

    write_filtered(train_output, train_idx)
    write_filtered(dev_output, dev_idx)
    print(f"Wrote {train_output}")
    print(f"Wrote {dev_output}")


if __name__ == "__main__":
    train_dev_split()
