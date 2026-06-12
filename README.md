# Learn-to-Defer Reward Model Router

Class project for **Natural Language Processing with Deep Learning**, Winter 2026

This repository implements a **learn-to-defer** router that decides when to trust a cheap scalar reward model (Stage I) versus escalating to an expensive generative judge (Stage II). The router is trained under an escalation budget using Lagrangian optimization.

**Runs locally or on [Modal](https://modal.com)** — training, generative judging, and embedding extraction all have Modal apps built in. Use your laptop for development and Modal for GPU-heavy jobs without changing code paths.

## Repository layout

```
learn2defer-rm/
├── src/
│   ├── train/                  # RouterNet model + training loops (Modal apps included)
│   ├── dataset/                # data loaders, benchmarks, feature extraction
│   ├── eval/                   # strategy evaluation + results analysis
│   └── utils/                  # paths, config, prompts, wandb, inference
├── scripts/                    # runnable entry points (local + `modal run` targets)
├── results/                    # checkpoints, training logs, and evaluation outputs
└── environment.yml             # micromamba env spec (name: l2d)
```

## Installation

### Micromamba (recommended)

This project uses a micromamba environment named `**l2d**`. If you already have it:

```bash
micromamba activate l2d
cd learn2defer-rm
pip install -e .
```

To create the env from scratch, or refresh an existing one:

```bash
micromamba env create -f environment.yml          # first time
# or
micromamba env update -n l2d -f environment.yml --prune

micromamba activate l2d
pip install -e .
```

The env includes Python 3.12, PyTorch, and project dependencies (including `wandb` and `modal`).

### pip / venv (alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[modal]"    # required for Modal cloud jobs
```

## Modal (cloud GPU)

Modal is first-class in this repo. Several `scripts/` entry points embed Modal apps — run them with `modal run` from the repo root (no local GPU required).

### Setup

```bash
pip install modal          # or: pip install -e ".[modal]"
modal setup                # one-time auth
```

### What runs on Modal


| Task                               | Script                                   | GPU       | Notes                               |
| ---------------------------------- | ---------------------------------------- | --------- | ----------------------------------- |
| Router training (Lagrangian BCE)   | `scripts/train_lag_bce.py`               | T4        | Checkpoints saved to Modal volumes  |
| Router training (Lagrangian hinge) | `scripts/train_lag.py`                   | T4        | Same pattern                        |
| Router training (fixed-λ BCE)      | `scripts/train_bce.py`                   | T4        | Sweeps over λ values                |
| R3 generative judging              | `scripts/generate_responses.py`          | A100-80GB | Writes to Modal volume `r3-outputs` |
| Scalar feature embeddings          | `scripts/get_scalar_features.py --modal` | A100-80GB | Requires `--embeddings`             |


Training scripts accept the same CLI flags locally and on Modal. Checkpoints and logs are written to Modal **Volumes** (per `input_mode` for ablations).

### Examples

```bash
# Train on Modal (primary method)
modal run scripts/train_lag_bce.py --input-mode full
modal run scripts/train_lag_bce.py --input-mode scalar_only
modal run scripts/train_lag_bce.py --input-mode embedding_only

# Other trainers
modal run scripts/train_lag.py --input-mode full
modal run scripts/train_bce.py --input-mode embedding_only

# Generate R3 judge responses (background job; pulls from Modal volume when done)
python scripts/generate_responses.py
# then: modal volume get r3-outputs /outputs/<bench>__R3_Qwen3_14B_14k__responses.json .

# ModernBERT embeddings on Modal
modal run scripts/get_scalar_features.py -- --embeddings --modal --dataset rmbench
```

Modal app definitions live in `src/train/` (training) and `scripts/generate_responses.py` / `src/dataset/features.py` (data generation). Volumes persist checkpoints across runs — use `modal volume ls` to inspect them.

## Data pipeline

1. **Scalar scores** — collect per-model chosen/rejected scores into `scalar_dataset/`.
2. **Generative labels** — run the R3 judge locally or on Modal (`scripts/generate_responses.py`) and merge into `generative_dataset/`.
3. **Feature files** — build processed JSONL features used by the router:

```bash
python scripts/get_scalar_features.py
python scripts/reorg_dataset.py generative_dataset
```

Processed router features live in `scalar_dataset/processed/`. Set `L2D_DATA_ROOT` to point at a custom data directory if needed.

## Training

The router has two input heads that can be ablated:


| `input_mode`     | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `full`           | Scalar RM features **and** response embeddings (default) |
| `scalar_only`    | Stage-I RM scores, margin, and token counts only         |
| `embedding_only` | Concatenated chosen/rejected response embeddings only    |


### Local

```bash
micromamba activate l2d

python scripts/train_lag_bce.py --input-mode full --output-dir results/lag_bce_full
python scripts/train_lag_bce.py --input-mode scalar_only --output-dir results/lag_bce_scalar_only
python scripts/train_lag_bce.py --input-mode embedding_only --output-dir results/lag_bce_embedding_only

python scripts/train_lag.py      # Lagrangian hinge-style loss
python scripts/train_bce.py      # fixed-lambda BCE sweep
```

### Modal

Same commands, prefixed with `modal run` — see [Modal (cloud GPU)](#modal-cloud-gpu) above.

### Weights & Biases logging

Training scripts support optional W&B logging (local or Modal). Log in once, then pass `--wandb`:

```bash
wandb login

python scripts/train_lag_bce.py \
  --input-mode full \
  --output-dir results/lag_bce_full \
  --wandb \
  --wandb-project learn2defer-rm \
  --wandb-group lag_bce_ablations \
  --wandb-tags "full,lag_bce"
```

Logged metrics per epoch: `loss`, `train/accuracy`, `train/escalation`, `dev/accuracy`, `dev/escalation`, and `lambda` (for Lagrangian trainers). Disable logging with `WANDB_DISABLED=true` or by omitting `--wandb`.

## Evaluation and analysis

```bash
python scripts/analyze_results.py --folder lag_bce_full
python scripts/inspect_data.py --path generative_dataset/df_train_cleaned_indexed_train.json --accuracy
```

Analysis CSVs and plots are written to `results/<run>/analysis/`.

