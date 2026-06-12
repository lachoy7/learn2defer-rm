#!/usr/bin/env python3
"""Convert JSONL indexed datasets to pretty-printed JSON arrays."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from utils.paths import GENERATIVE_DATASET_DIR

DEFAULT_FILES = [
    "df_train_cleaned_indexed_dev.json",
    "df_train_cleaned_indexed_train.json",
    "df_test_rmbench_indexed.json",
    "df_test_rewardbench_indexed.json",
]


def adapt_file(input_path: Path) -> None:
    with open(input_path, "r") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            with open(input_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Pretty-printed (already array): {input_path}")
            return
    except json.JSONDecodeError:
        pass
    data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Converted JSONL to array: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL datasets to JSON arrays.")
    parser.add_argument("--dir", type=str, default=str(GENERATIVE_DATASET_DIR))
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()
    root = Path(args.dir)
    for name in args.files:
        adapt_file(root / name)


if __name__ == "__main__":
    main()
