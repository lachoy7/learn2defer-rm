#!/usr/bin/env python3
"""Reorganize indexed dataset JSON files into the router training schema."""

import argparse
import json
import os
from pathlib import Path

from utils.paths import GENERATIVE_DATASET_DIR

import _bootstrap  # noqa: F401


def transform_line(obj: dict) -> dict:
    scalar = {}
    scalar_map = [
        ("model_id_x", "model_id"),
        ("model_id", "model_id"),
        ("tokens_chosen", "tokens_chosen"),
        ("tokens_rejected", "tokens_rejected"),
        ("tokens_total", "tokens_total"),
        ("chosen_score", "chosen_score"),
        ("rejected_score", "rejected_score"),
        ("params", "params"),
        ("flops_est", "flops_est"),
        ("margin", "margin"),
        ("p_choose", "p_choose"),
        ("is_correct_scalar", "is_correct"),
    ]
    for src_key, dst_key in scalar_map:
        if src_key in obj:
            scalar[dst_key] = obj[src_key]

    generative = {}
    gen_map = [
        ("response", "judge_thinking"),
        ("judge_tokens", "tokens_judge"),
        ("chosen_response_tokens", "tokens_chosen"),
        ("rejected_response_tokens", "tokens_rejected"),
        ("is_correct_generative", "is_correct"),
    ]
    for src_key, dst_key in gen_map:
        if src_key in obj:
            generative[dst_key] = obj[src_key]

    out = {
        "idx": obj.get("idx"),
        "bench": obj.get("bench_x") or obj.get("bench"),
        "bench_split": obj.get("bench_split"),
        "source": obj.get("source"),
        "prompt": obj.get("prompt_x") or obj.get("prompt_y") or obj.get("prompt"),
        "chosen_response": obj.get("chosen_response_scalar"),
        "rejected_response": obj.get("rejected_response_scalar"),
        "scalar": scalar,
        "generative": generative,
    }
    if out.get("bench") == "rmbench":
        out["domain"] = obj.get("domain_x")
        out["chosen_style"] = obj.get("chosen_style")
        out["rejected_style"] = obj.get("rejected_style")
        out["difficulty"] = obj.get("difficulty")
    return out


def _read_records(path: Path):
    with open(path, "r") as f:
        first = f.readline()
    with open(path, "r") as f:
        if first.strip().startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise RuntimeError(f"{path}: expected JSON array")
            yield from data
        else:
            f.seek(0)
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    yield json.loads(line_stripped)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"{path}: line {i + 1}: invalid JSON: {e}") from e


def process_file(path: Path, out_path: Path, dry_run: bool) -> int:
    count = 0
    if dry_run:
        for _ in _read_records(path):
            count += 1
        return count

    with open(out_path, "w") as f:
        for obj in _read_records(path):
            rec = transform_line(obj)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Reorganize indexed dataset JSON(L) files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "dir",
        nargs="?",
        default=str(GENERATIVE_DATASET_DIR),
        help="Directory with df_train*.json and df_test*.json files",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    json_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix == ".json"
        and (p.name.startswith("df_train") or p.name.startswith("df_test"))
    )
    if not json_files:
        print("No df_train*.json or df_test*.json files found in", root)
        return

    for path in json_files:
        out_name = "reorg_" + path.name
        out_path = root / out_name
        if args.dry_run:
            print("[DRY-RUN] Would process", path.name, "->", out_name)
            n = process_file(path, out_path, dry_run=True)
            print("  Records transformed:", n)
        else:
            print("Processing", path.name, "->", out_name)
            n = process_file(path, out_path, dry_run=False)
            print("  Wrote", n, "records to", out_path)


if __name__ == "__main__":
    main()
