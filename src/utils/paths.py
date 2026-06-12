"""Repository path constants."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATA_ROOT = Path(os.environ.get("L2D_DATA_ROOT", REPO_ROOT))
GENERATIVE_DATASET_DIR = DATA_ROOT / "generative_dataset"
SCALAR_DATASET_DIR = DATA_ROOT / "scalar_dataset"
SCALAR_FEATURES_DIR = SCALAR_DATASET_DIR / "features"
PROCESSED_FEATURES_DIR = SCALAR_DATASET_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

TRAIN_FEATURES = PROCESSED_FEATURES_DIR / "reorg_df_train_cleaned_indexed_train_features.jsonl"
DEV_FEATURES = PROCESSED_FEATURES_DIR / "reorg_df_train_cleaned_indexed_dev_features.jsonl"
TEST_RMBENCH_FEATURES = PROCESSED_FEATURES_DIR / "reorg_df_test_rmbench_indexed_features.jsonl"
