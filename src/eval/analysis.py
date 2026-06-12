"""
Analyze router results from JSON files in a results folder.
Computes accuracy, escalation rate per category and per target, style matrix, and easy/normal/hard breakdowns.
Saves CSVs and style matrix plots to <folder>/analysis/.

Usage:
    python scripts/analyze_results.py --folder results/lag_bce_full
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from utils.paths import DEV_FEATURES, RESULTS_DIR, TEST_RMBENCH_FEATURES, TRAIN_FEATURES

STYLE_NAMES = ["concise", "detailed_plain", "detailed_markdown"]

REORG_TEST_JSONL = TEST_RMBENCH_FEATURES
REORG_TRAIN_JSONL = TRAIN_FEATURES
REORG_DEV_JSONL = DEV_FEATURES


def load_baseline_0_1_records(path: Path | None = None) -> pd.DataFrame:
    """
    Load test-set records and build baseline rows for target_escalation_rate=0 (always cheap)
    and target_escalation_rate=1 (always heavy). Uses first scalar_ block for cheap correctness
    and generative.is_correct for heavy. Returns a DataFrame with same columns as router records
    so it can be concatenated to the main results df.
    """
    path = path or REORG_TEST_JSONL
    if not path.exists():
        return pd.DataFrame()
    rows_0 = []
    rows_1 = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            gen = rec.get("generative") or {}
            is_correct_D = float(bool(gen.get("is_correct")))
            # First scalar_ block (sorted for reproducibility) for cheap correctness
            scalar_keys = sorted(k for k in rec if k.startswith("scalar_") and isinstance(rec.get(k), dict))
            if not scalar_keys:
                continue
            block = rec.get(scalar_keys[0]) or {}
            ch = float(block.get("chosen_score", 0.0))
            rej = float(block.get("rejected_score", 0.0))
            is_correct_M = 1.0 if ch > rej else 0.0
            domain = rec.get("domain", "")
            chosen_style = rec.get("chosen_style", "")
            rejected_style = rec.get("rejected_style", "")
            difficulty = rec.get("difficulty", "")
            rows_0.append({
                "idx": idx,
                "domain_or_source": domain,
                "domain": domain,
                "chosen_style": chosen_style,
                "rejected_style": rejected_style,
                "difficulty": difficulty,
                "is_correct": is_correct_M,
                "escalate": 0.0,
                "target_escalation_rate": 0.0,
                "model_id": "baseline_target0_index0",
            })
            rows_1.append({
                "idx": idx,
                "domain_or_source": domain,
                "domain": domain,
                "chosen_style": chosen_style,
                "rejected_style": rejected_style,
                "difficulty": difficulty,
                "is_correct": is_correct_D,
                "escalate": 1.0,
                "target_escalation_rate": 1.0,
                "model_id": "baseline_target1_index0",
            })
    if not rows_0:
        return pd.DataFrame()
    return pd.concat([pd.DataFrame(rows_0), pd.DataFrame(rows_1)], ignore_index=True)


def load_tokens_judge_jsonl(path: Path | None = None) -> pd.DataFrame:
    """Load idx and tokens_judge from reorg jsonl (one line per record). Returns DataFrame with columns idx, tokens_judge."""
    path = path or REORG_TEST_JSONL
    if not path.exists():
        return pd.DataFrame(columns=["idx", "tokens_judge"])
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            gen = rec.get("generative") or {}
            tokens_judge = float(gen.get("tokens_judge", 0))
            rows.append({"idx": idx, "tokens_judge": tokens_judge})
    return pd.DataFrame(rows)


def load_results_to_df(results_dir: Path | str) -> pd.DataFrame:
    """Load all JSON files from results_dir into a single DataFrame."""
    results_dir = Path(results_dir)
    all_records = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        for r in data:
            rec = dict(r)
            rec["model_id"] = p.stem
            all_records.append(rec)
    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    # Extract target_escalation_rate from model_id (e.g. target0.05_index0 -> 0.05)
    m = df["model_id"].str.extract(r"target([\d.]+)_index", expand=False)
    df["target_escalation_rate"] = pd.to_numeric(m, errors="coerce")
    # Align with rmbench naming
    if "domain_or_source" in df.columns and "domain" not in df.columns:
        df["domain"] = df["domain_or_source"]
    return df


def load_best_epoch_train_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Load train/dev accuracy and escalation at the best epoch for each model.
    Uses same best-epoch logic as train_lag_bce.py and train_lag.py:
    - When dev_escalation <= target: best = argmax dev_accuracy
    - Else: best = argmin |dev_escalation - target|
    For train_bce.py (fixed lambda): best = argmax dev_accuracy.
    """
    results_dir = Path(results_dir)
    rows = []
    for p in sorted(results_dir.glob("train_log_*.csv")):
        log_df = pd.read_csv(p)
        if log_df.empty or "dev_accuracy" not in log_df.columns:
            continue
        stem = p.stem
        # Match train_log_target0.05_index0 or train_log_target0.05_index0_embeddingonly
        target_m = re.search(r"train_log_target([\d.]+)_index", stem)
        lambda_m = re.search(r"train_log_lambda([\d.]+)_index", stem)
        if target_m:
            target = float(target_m.group(1))
            # Lagrangian logic: escalation met -> best by dev_acc; else best by closest escal to target
            log_df = log_df.copy()
            log_df["escalation_met"] = log_df["dev_escalation"] <= target
            escal_met = log_df[log_df["escalation_met"]]
            if not escal_met.empty:
                best_row = escal_met.loc[escal_met["dev_accuracy"].idxmax()]
            else:
                log_df["escal_diff"] = (log_df["dev_escalation"] - target).abs()
                best_row = log_df.loc[log_df["escal_diff"].idxmin()]
        elif lambda_m:
            target = float(lambda_m.group(1))  # treat as "target" for merge key; train_bce uses lambda
            best_row = log_df.loc[log_df["dev_accuracy"].idxmax()]
        else:
            continue
        rows.append({
            "target_escalation_rate": target,
            "best_epoch": int(best_row["epoch"]),
            "train_accuracy": float(best_row["train_accuracy"]),
            "dev_accuracy": float(best_row["dev_accuracy"]),
            "train_escalation": float(best_row["train_escalation"]),
            "dev_escalation": float(best_row["dev_escalation"]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_baseline_train_dev_metrics(
    train_path: Path | None = None,
    dev_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compute train_accuracy and dev_accuracy for target_escalation_rate=0 (always cheap)
    and target_escalation_rate=1 (always heavy) from train and dev feature JSONL files.
    Returns a DataFrame with same columns as load_best_epoch_train_metrics for concat.
    """
    train_path = train_path or REORG_TRAIN_JSONL
    dev_path = dev_path or REORG_DEV_JSONL
    if not train_path.exists() or not dev_path.exists():
        return pd.DataFrame()

    def _accuracies_from_jsonl(path: Path) -> tuple[float, float]:
        """Return (mean is_correct_M, mean is_correct_D) over all records in path."""
        sum_m = sum_d = n = 0.0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                gen = rec.get("generative") or {}
                is_correct_D = float(bool(gen.get("is_correct")))
                scalar_keys = sorted(k for k in rec if k.startswith("scalar_") and isinstance(rec.get(k), dict))
                if not scalar_keys:
                    continue
                block = rec.get(scalar_keys[0]) or {}
                ch = float(block.get("chosen_score", 0.0))
                rej = float(block.get("rejected_score", 0.0))
                is_correct_M = 1.0 if ch > rej else 0.0
                sum_m += is_correct_M
                sum_d += is_correct_D
                n += 1
        if n == 0:
            return (np.nan, np.nan)
        return (sum_m / n, sum_d / n)

    train_m, train_d = _accuracies_from_jsonl(train_path)
    dev_m, dev_d = _accuracies_from_jsonl(dev_path)

    return pd.DataFrame([
        {
            "target_escalation_rate": 0.0,
            "best_epoch": 0,
            "train_accuracy": train_m,
            "dev_accuracy": dev_m,
            "train_escalation": 0.0,
            "dev_escalation": 0.0,
        },
        {
            "target_escalation_rate": 1.0,
            "best_epoch": 0,
            "train_accuracy": train_d,
            "dev_accuracy": dev_d,
            "train_escalation": 1.0,
            "dev_escalation": 1.0,
        },
    ])


def accuracy_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (mean of is_correct) per target_escalation_rate and domain."""
    gcols = ["target_escalation_rate"]
    if "domain" in df.columns:
        gcols.append("domain")
    return (
        df.groupby(gcols, dropna=False)
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "count"))
        .reset_index()
    )


def escalation_rate_per_category(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Escalation rate (proportion where escalate > threshold) per target_escalation_rate and domain."""
    df = df.copy()
    df["did_escalate"] = (df["escalate"] > threshold).astype(float)
    gcols = ["target_escalation_rate"]
    if "domain" in df.columns:
        gcols.append("domain")
    out = (
        df.groupby(gcols, dropna=False)
        .agg(escalation_rate=("did_escalate", "mean"), n=("did_escalate", "count"))
        .reset_index()
    )
    out["actual_escalation_rate"] = out["escalation_rate"]
    cols = ["target_escalation_rate", "actual_escalation_rate"]
    if "domain" in out.columns:
        cols.append("domain")
    cols.extend(["escalation_rate", "n"])
    return out[cols]


def accuracy_per_target(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (mean of is_correct) per target_escalation_rate only (overall, not per domain)."""
    return (
        df.groupby("target_escalation_rate", dropna=False)
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "count"))
        .reset_index()
    )


def escalation_rate_per_target(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Escalation rate (proportion where escalate > threshold) per target_escalation_rate only (overall)."""
    df = df.copy()
    df["did_escalate"] = (df["escalate"] > threshold).astype(float)
    return (
        df.groupby("target_escalation_rate", dropna=False)
        .agg(escalation_rate=("did_escalate", "mean"), n=("did_escalate", "count"))
        .reset_index()
    )


def rmbench_style_matrix(df_rm: pd.DataFrame):
    """Return 3x3 matrix per target_escalation_rate and domain (row=chosen style, col=rejected style)."""
    gcols = ["target_escalation_rate"]
    if "domain" in df_rm.columns:
        gcols.append("domain")

    mats = []
    for keys, g in df_rm.groupby(gcols, dropna=False):
        mat = pd.pivot_table(
            g,
            values="is_correct",
            index="chosen_style",
            columns="rejected_style",
            aggfunc="mean",
        ).reindex(index=STYLE_NAMES, columns=STYLE_NAMES)
        mat = mat.fillna(np.nan)
        mat.attrs["keys"] = keys
        mats.append(mat)
    return mats


def plot_style_matrix(mat: pd.DataFrame, keys: tuple, out_path: Path) -> None:
    """Plot a 3x3 style matrix as a heatmap and save to PNG."""
    arr = mat.values.astype(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(STYLE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(STYLE_NAMES)
    ax.set_xlabel("Rejected style")
    ax.set_ylabel("Chosen style")
    target = keys[0]
    domain = keys[1] if len(keys) == 2 else "overall"
    ax.set_title(f"Style matrix (target={target}, domain={domain})")
    for i in range(3):
        for j in range(3):
            v = arr[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_train_log(csv_path: Path, out_path: Path) -> None:
    """Plot train/dev loss, acc, escalation, and lambda in 4 subplots on one figure."""
    df = pd.read_csv(csv_path)
    # Extract target from filename, e.g. train_log_target0.05_index0.csv -> 0.05
    stem = csv_path.stem
    target = stem.replace("train_log_target", "").split("_index")[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    # Plot 1: Loss (train - CSV has single 'loss' column)
    ax1 = axes[0, 0]
    ax1.plot(df["epoch"], df["loss"], label="Train loss", color="C0")
    if "dev_loss" in df.columns:
        ax1.plot(df["epoch"], df["dev_loss"], label="Dev loss", color="C1")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss (target={target})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Train and dev accuracy
    ax2 = axes[0, 1]
    ax2.plot(df["epoch"], df["train_accuracy"], label="Train acc", color="C0")
    ax2.plot(df["epoch"], df["dev_accuracy"], label="Dev acc", color="C1")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy (target={target})")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Train and dev escalation
    ax3 = axes[1, 0]
    ax3.plot(df["epoch"], df["train_escalation"], label="Train escalation", color="C0")
    ax3.plot(df["epoch"], df["dev_escalation"], label="Dev escalation", color="C1")
    ax3.set_ylabel("Escalation rate")
    ax3.set_title(f"Escalation (target={target})")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Lambda
    ax4 = axes[1, 1]
    ax4.plot(df["epoch"], df["lambda"], color="C2")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("λ")
    ax4.set_title(f"Lambda (target={target})")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _target_from_train_log_stem(stem: str) -> float | None:
    """Extract target escalation rate from train_log filename stem. Returns None if not matched."""
    m = re.search(r"train_log_target([\d.]+)_index", stem)
    return float(m.group(1)) if m else None


def plot_train_summary_all_targets(log_data: list[tuple[float, pd.DataFrame]], out_path: Path) -> None:
    """Plot train metrics (accuracy, escalation, lambda) with one curve per target escalation rate."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for target, df in sorted(log_data, key=lambda x: x[0]):
        label = f"target={target}"
        axes[0].plot(df["epoch"], df["train_accuracy"], label=label)
        axes[1].plot(df["epoch"], df["train_escalation"], label=label)
        axes[2].plot(df["epoch"], df["lambda"], label=label)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Escalation rate")
    axes[1].set_title("Train escalation")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("λ")
    axes[2].set_title("Lambda")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dev_summary_all_targets(log_data: list[tuple[float, pd.DataFrame]], out_path: Path) -> None:
    """Plot dev metrics (accuracy, escalation, lambda) with one curve per target escalation rate."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for target, df in sorted(log_data, key=lambda x: x[0]):
        label = f"target={target}"
        axes[0].plot(df["epoch"], df["dev_accuracy"], label=label)
        axes[1].plot(df["epoch"], df["dev_escalation"], label=label)
        axes[2].plot(df["epoch"], df["lambda"], label=label)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Dev accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Escalation rate")
    axes[1].set_title("Dev escalation")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("λ")
    axes[2].set_title("Lambda")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_train_dev_summary(log_data: list[tuple[float, pd.DataFrame]], out_path: Path) -> None:
    """Combined figure: row1 = train & dev accuracy (shared y), row2 = train & dev escalation (shared y). One legend. Lambda is in a separate figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_train_acc, ax_dev_acc = axes[0, 0], axes[0, 1]
    ax_train_escal, ax_dev_escal = axes[1, 0], axes[1, 1]

    # Collect global y limits for accuracy and escalation
    acc_min, acc_max = np.inf, -np.inf
    escal_min, escal_max = np.inf, -np.inf
    for target, df in log_data:
        acc_min = min(acc_min, df["train_accuracy"].min(), df["dev_accuracy"].min())
        acc_max = max(acc_max, df["train_accuracy"].max(), df["dev_accuracy"].max())
        escal_min = min(escal_min, df["train_escalation"].min(), df["dev_escalation"].min())
        escal_max = max(escal_max, df["train_escalation"].max(), df["dev_escalation"].max())

    for target, df in sorted(log_data, key=lambda x: x[0]):
        label = f"{target}"
        ax_train_acc.plot(df["epoch"], df["train_accuracy"], label=label)
        ax_dev_acc.plot(df["epoch"], df["dev_accuracy"], label=label)
        ax_train_escal.plot(df["epoch"], df["train_escalation"], label=label)
        ax_dev_escal.plot(df["epoch"], df["dev_escalation"], label=label)

    ax_train_acc.set_ylabel("Accuracy")
    ax_train_acc.set_title("Train accuracy")
    ax_train_acc.set_ylim(acc_min, acc_max)
    ax_train_acc.grid(True, alpha=0.3)

    ax_dev_acc.set_title("Dev accuracy")
    ax_dev_acc.set_ylim(acc_min, acc_max)
    ax_dev_acc.set_ylabel("")
    ax_dev_acc.set_yticklabels([])
    ax_dev_acc.grid(True, alpha=0.3)

    ax_train_escal.set_ylabel("Escalation rate")
    ax_train_escal.set_title("Train escalation")
    ax_train_escal.set_ylim(escal_min, escal_max)
    ax_train_escal.grid(True, alpha=0.3)

    ax_dev_escal.set_title("Dev escalation")
    ax_dev_escal.set_ylim(escal_min, escal_max)
    ax_dev_escal.set_ylabel("")
    ax_dev_escal.set_yticklabels([])
    ax_dev_escal.grid(True, alpha=0.3)

    # Single legend for the whole figure
    handles, labels = ax_train_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=len(handles), frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_train_dev_lambda(log_data: list[tuple[float, pd.DataFrame]], out_path: Path) -> None:
    """Separate figure: lambda vs epoch, one curve per target escalation rate."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for target, df in sorted(log_data, key=lambda x: x[0]):
        ax.plot(df["epoch"], df["lambda"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("λ")
    ax.set_title("Lambda")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def rmbench_easy_normal_hard(mat: pd.DataFrame) -> dict:
    """easy=lower triangle, normal=diagonal, hard=upper triangle (RM-Bench defn)."""
    arr = mat.values.astype(float)
    easy = np.nanmean(arr[np.tril_indices(3, k=-1)])
    normal = np.nanmean(np.diag(arr))
    hard = np.nanmean(arr[np.triu_indices(3, k=1)])
    return {
        "easy": easy,
        "normal": normal,
        "hard": hard,
        "style_gap(easy-hard)": easy - hard,
    }


def main(results_dir: Path) -> tuple:
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    df = load_results_to_df(results_dir)
    if df.empty:
        print("No JSON files found in", results_dir, "- run eval to generate router records first.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Add baseline rows for target_escalation_rate=0 (always cheap) and 1 (always heavy) from test features
    baseline_df = load_baseline_0_1_records()
    if not baseline_df.empty:
        df = pd.concat([df, baseline_df], ignore_index=True)
        print("Added baseline rows for target_escalation_rate=0 and 1 from test features.")
    print("Loaded", len(df), "records from", df["target_escalation_rate"].nunique(), "target rates\n")

    # Load best-epoch train/dev metrics (same logic as train_lag_bce.py)
    train_metrics = load_best_epoch_train_metrics(results_dir)
    # Add train/dev accuracy for baseline target 0 and 1 (always cheap / always heavy)
    baseline_metrics = load_baseline_train_dev_metrics()
    if not baseline_metrics.empty:
        train_metrics = pd.concat([train_metrics, baseline_metrics], ignore_index=True)
        print("Added train/dev metrics for target_escalation_rate=0 and 1 from train/dev features.")

    # Accuracy per category
    acc_df = accuracy_per_category(df)
    acc_df.to_csv(analysis_dir / "accuracy_per_category.csv", index=False)
    print("=== Accuracy per category ===")
    print(acc_df.to_string(index=False))
    print()

    # Accuracy per target_escalation_rate (overall, not per domain)
    acc_per_target = accuracy_per_target(df)
    if not train_metrics.empty:
        acc_per_target = acc_per_target.merge(
            train_metrics[["target_escalation_rate", "train_accuracy", "dev_accuracy"]],
            on="target_escalation_rate", how="left"
        )
    acc_per_target.to_csv(analysis_dir / "accuracy_per_target.csv", index=False)
    print("=== Accuracy per target_escalation_rate ===")
    print(acc_per_target.to_string(index=False))
    print()

    # Escalation rate per category (target_escalation_rate + domain)
    esc_df = escalation_rate_per_category(df, threshold=0.5)
    esc_df.to_csv(analysis_dir / "escalation_rate_per_category.csv", index=False)
    print("=== Escalation rate (escalate > 0.5) per category (target + domain) ===")
    print(esc_df.to_string(index=False))
    print()

    # Escalation rate per target_escalation_rate (overall, not per domain)
    esc_per_target = escalation_rate_per_target(df, threshold=0.5)
    if not train_metrics.empty:
        esc_per_target = esc_per_target.merge(
            train_metrics[["target_escalation_rate", "train_escalation", "dev_escalation"]],
            on="target_escalation_rate", how="left"
        )

    # Merge accuracy and escalation into one summary
    merge_cols = ["target_escalation_rate"] + (["domain"] if "domain" in df.columns else [])
    summary = acc_df.merge(esc_df.drop(columns=["n"]), on=merge_cols, how="outer")
    summary.to_csv(analysis_dir / "accuracy_escalation_summary.csv", index=False)
    print("=== Combined accuracy + escalation rate ===")
    print(summary.to_string(index=False))
    print()

    # Load tokens_judge once for both esc_per_target and style_easy_normal_hard
    tokens_df = load_tokens_judge_jsonl()
    df_with_tokens = None
    if not tokens_df.empty:
        df_with_tokens = df.merge(tokens_df[["idx", "tokens_judge"]], on="idx", how="left")
        df_with_tokens["tokens_judge"] = df_with_tokens["tokens_judge"].fillna(0)

    # Add to escalation_rate_per_target: total_tokens_judge_section (sum across domains), total_tokens_judge_escalated, percent_tokens_saved
    if df_with_tokens is not None:
        by_target = ["target_escalation_rate"]
        total_section_target = (
            df_with_tokens.groupby(by_target, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
            .rename(columns={"tokens_judge": "total_tokens_judge_section"})
        )
        escalated_target = (
            df_with_tokens.loc[df_with_tokens["escalate"] > 0.5]
            .groupby(by_target, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
            .rename(columns={"tokens_judge": "total_tokens_judge_escalated"})
        )
        not_escalated_target = (
            df_with_tokens.loc[df_with_tokens["escalate"] <= 0.5]
            .groupby(by_target, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
            .rename(columns={"tokens_judge": "tokens_judge_not_escalated"})
        )
        esc_per_target = esc_per_target.merge(total_section_target, on=by_target, how="left")
        esc_per_target = esc_per_target.merge(escalated_target, on=by_target, how="left")
        esc_per_target = esc_per_target.merge(not_escalated_target, on=by_target, how="left")
        esc_per_target["total_tokens_judge_section"] = esc_per_target["total_tokens_judge_section"].fillna(0).astype(int)
        esc_per_target["total_tokens_judge_escalated"] = esc_per_target["total_tokens_judge_escalated"].fillna(0).astype(int)
        esc_per_target["tokens_judge_not_escalated"] = esc_per_target["tokens_judge_not_escalated"].fillna(0)
        total_sec = esc_per_target["total_tokens_judge_section"]
        esc_per_target["percent_tokens_saved"] = np.where(
            total_sec > 0,
            (esc_per_target["tokens_judge_not_escalated"] / total_sec) * 100.0,
            0.0,
        )
        esc_per_target = esc_per_target.drop(columns=["tokens_judge_not_escalated"])
    esc_per_target.to_csv(analysis_dir / "escalation_rate_per_target.csv", index=False)
    print("=== Escalation rate (escalate > 0.5) per target_escalation_rate ===")
    print(esc_per_target.to_string(index=False))
    print()

    # Per-target summary: target_escalation, escalation_rate, accuracy, total_tokens_judge_section, total_tokens_judge_escalated, percent_tokens_saved
    per_target_summary = esc_per_target.merge(
        acc_per_target[["target_escalation_rate", "accuracy"]], on="target_escalation_rate", how="left"
    )
    per_target_summary = per_target_summary.rename(columns={"target_escalation_rate": "target_escalation"})
    out_cols = ["target_escalation", "escalation_rate", "accuracy"]
    for c in ["total_tokens_judge_section", "total_tokens_judge_escalated", "percent_tokens_saved"]:
        if c in per_target_summary.columns:
            out_cols.append(c)
    per_target_summary = per_target_summary[out_cols]
    per_target_summary.to_csv(analysis_dir / "per_target_summary.csv", index=False)
    print("=== Per-target summary ===")
    print(per_target_summary.to_string(index=False))
    print()

    # Style matrix and easy/normal/hard
    df_rm = df.copy()
    summary_rows = []
    for mat in rmbench_style_matrix(df_rm):
        keys = mat.attrs["keys"]
        stats = rmbench_easy_normal_hard(mat)
        row = {"target_escalation_rate": keys[0]}
        if len(keys) == 2:
            row["domain"] = keys[1]
        row.update(stats)
        summary_rows.append(row)

    rm_summary = pd.DataFrame(summary_rows).sort_values(
        ["target_escalation_rate"] + (["domain"] if "domain" in pd.DataFrame(summary_rows).columns else [])
    )
    # Add actual_escalation_rate (from esc_df) right after target_escalation_rate
    merge_cols_rm = ["target_escalation_rate"] + (["domain"] if "domain" in rm_summary.columns else [])
    rm_summary = rm_summary.merge(
        esc_df[merge_cols_rm + ["actual_escalation_rate"]],
        on=merge_cols_rm,
        how="left",
    )
    # Add total tokens_judge for instances where escalation policy chose to escalate (escalate > 0.5)
    # Also: total tokens_judge in section, and percent of tokens saved (tokens when didn't escalate / total)
    if df_with_tokens is not None:
        # Use same df_with_tokens as esc_per_target (already built above)
        # Total tokens_judge in RM-bench for this (target, domain) section
        total_tokens_by_section = (
            df_with_tokens.groupby(merge_cols_rm, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
            .rename(columns={"tokens_judge": "total_tokens_judge_section"})
        )
        # Sum when escalated (escalate > 0.5)
        escalated_tokens = (
            df_with_tokens.loc[df_with_tokens["escalate"] > 0.5]
            .groupby(merge_cols_rm, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
        )
        escalated_tokens = escalated_tokens.rename(columns={"tokens_judge": "total_tokens_judge_escalated"})
        # Sum when did not escalate (escalate <= 0.5) = "tokens saved"
        not_escalated_tokens = (
            df_with_tokens.loc[df_with_tokens["escalate"] <= 0.5]
            .groupby(merge_cols_rm, dropna=False)["tokens_judge"]
            .sum()
            .reset_index()
            .rename(columns={"tokens_judge": "tokens_judge_not_escalated"})
        )
        rm_summary = rm_summary.merge(total_tokens_by_section, on=merge_cols_rm, how="left")
        rm_summary = rm_summary.merge(escalated_tokens, on=merge_cols_rm, how="left")
        rm_summary = rm_summary.merge(not_escalated_tokens, on=merge_cols_rm, how="left")
        rm_summary["total_tokens_judge_escalated"] = rm_summary["total_tokens_judge_escalated"].fillna(0).astype(int)
        rm_summary["total_tokens_judge_section"] = rm_summary["total_tokens_judge_section"].fillna(0).astype(int)
        rm_summary["tokens_judge_not_escalated"] = rm_summary["tokens_judge_not_escalated"].fillna(0)
        # Percent of tokens saved = (tokens when didn't escalate) / total tokens_judge in section
        total = rm_summary["total_tokens_judge_section"]
        rm_summary["percent_tokens_saved"] = np.where(
            total > 0,
            (rm_summary["tokens_judge_not_escalated"] / total) * 100.0,
            0.0,
        )
        rm_summary = rm_summary.drop(columns=["tokens_judge_not_escalated"])
    cols_order = ["target_escalation_rate", "actual_escalation_rate"]
    if "domain" in rm_summary.columns:
        cols_order.append("domain")
    if "total_tokens_judge_section" in rm_summary.columns:
        cols_order.append("total_tokens_judge_section")
    if "total_tokens_judge_escalated" in rm_summary.columns:
        cols_order.append("total_tokens_judge_escalated")
    if "percent_tokens_saved" in rm_summary.columns:
        cols_order.append("percent_tokens_saved")
    cols_order.extend([c for c in rm_summary.columns if c not in cols_order])
    rm_summary = rm_summary[cols_order]
    rm_summary.to_csv(analysis_dir / "style_easy_normal_hard_summary.csv", index=False)
    print("=== Style matrix easy/normal/hard summary ===")
    print(rm_summary.sort_values(by=["domain", "target_escalation_rate"] if "domain" in rm_summary.columns else ["target_escalation_rate"]).to_string(index=False))

    # Plot and save style matrices as PNGs
    mats = rmbench_style_matrix(df_rm)
    if _HAS_MATPLOTLIB:
        for mat in mats:
            keys = mat.attrs["keys"]
            target = keys[0]
            domain = keys[1] if len(keys) == 2 else "overall"
            safe_domain = str(domain).replace("/", "-")
            fname = f"style_matrix_target{target}_{safe_domain}.png"
            plot_style_matrix(mat, keys, analysis_dir / fname)
        print(f"\nSaved {len(mats)} style matrix plots to {analysis_dir}")

        # Plot train logs (loss, acc, lambda) per target
        for p in sorted(results_dir.glob("train_log_*.csv")):
            target = p.stem.replace("train_log_target", "").split("_index")[0]
            out_path = analysis_dir / f"train_log_target{target}.png"
            plot_train_log(p, out_path)
        n_logs = len(list(results_dir.glob("train_log_*.csv")))
        print(f"Saved {n_logs} train log plots to {analysis_dir}")

        # Two summary plots: train and dev, each with one curve per target escalation rate
        log_data = []
        for p in sorted(results_dir.glob("train_log_*.csv")):
            t = _target_from_train_log_stem(p.stem)
            if t is not None:
                log_df = pd.read_csv(p)
                if not log_df.empty:
                    log_data.append((t, log_df))
        if log_data:
            plot_train_summary_all_targets(log_data, analysis_dir / "train_summary_all_targets.png")
            plot_dev_summary_all_targets(log_data, analysis_dir / "dev_summary_all_targets.png")
            plot_train_dev_summary(log_data, analysis_dir / "train_dev_summary.png")
            plot_train_dev_lambda(log_data, analysis_dir / "train_dev_lambda.png")
            print(f"Saved train_summary_all_targets.png, dev_summary_all_targets.png, train_dev_summary.png, and train_dev_lambda.png to {analysis_dir}")
    else:
        print("\nSkipping plots (matplotlib not installed)")

    return df, summary, rm_summary


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze router results from JSON files in a results folder."
    )
    parser.add_argument(
        "--folder",
        type=str,
        nargs="?",
        default="lag_bce_full",
        help="Results folder name (relative to results/) or absolute path (default: lag_bce_full)",
    )
    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = RESULTS_DIR / folder
    return folder


if __name__ == "__main__":
    results_path = _parse_args()
    df, summary, rm_summary = main(results_path)
