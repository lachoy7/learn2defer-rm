# =============================================================================
# IMPORTS — all dependencies at top (stdlib, third-party, local)
# =============================================================================
import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.data_loaders import get_router_dataloaders
from eval.eval import evaluate_strategy
from train.model import MODE_TAGS, RouterNet
from utils.paths import REPO_ROOT, RESULTS_DIR
from utils.wandb_logging import WandbLogger, add_wandb_args, wandb_kwargs_from_args

LOCAL_RESULTS_DIR = str(RESULTS_DIR)


def _get_loaders():
    return get_router_dataloaders(use_swapped_pd=True)

# =============================================================================
# SAVE DETAILED PER-SAMPLE JSON (for pandas analysis later)
# =============================================================================
def save_detailed_records(strategy: str, model, loader, device, output_path: str):
    records = []
    torch.manual_seed(42)

    with torch.no_grad():
        for b in tqdm(loader, desc=f"Saving {strategy} records"):
            cs = b["chosen_scores"].to(device)
            rs = b["rejected_scores"].to(device)
            tc = b["tokens_chosen"].to(device)
            tr = b["tokens_rejected"].to(device)
            emb = b["emb_cat"].to(device)
            bs = len(b["idx"])

            if strategy == "learned":
                _, s = model(cs, rs, tc, tr, emb)
            elif strategy == "cheap":
                s = torch.zeros(bs, device=device)
            elif strategy == "heavy":
                s = torch.ones(bs, device=device)
            elif strategy == "random":
                s = torch.rand(bs, device=device)

            pred_escalate = (s >= 0.5).float()
            chosen_correct = (1 - pred_escalate) * b["is_correct_M"].to(device) + pred_escalate * b["loss_D"].to(device)

            for i in range(bs):
                rec = {
                    "idx": int(b["idx"][i]),
                    "domain_or_source": b.get("domain", b.get("source", [""]*bs))[i],
                    "chosen_style": b.get("chosen_style", [""]*bs)[i],
                    "rejected_style": b.get("rejected_style", [""]*bs)[i],
                    "difficulty": b.get("difficulty", [""]*bs)[i],
                    "is_correct": float(chosen_correct[i].item()),
                    "strategy": strategy,
                    "escalate": float(s[i].item())
                }
                records.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records):,} records → {output_path}")


def get_dev_acc_and_escal(model, loader, device):
    """One pass over loader; returns (accuracy, escalation_rate) for learned strategy."""
    model.eval()
    total_correct = 0.0
    total_escal = 0.0
    n = 0
    with torch.no_grad():
        for b in loader:
            cs = b["chosen_scores"].to(device)
            rs = b["rejected_scores"].to(device)
            tc = b["tokens_chosen"].to(device)
            tr = b["tokens_rejected"].to(device)
            emb = b["emb_cat"].to(device)
            is_correct_M = b["is_correct_M"].to(device)
            is_correct_D = b["loss_D"].to(device)
            bs = is_correct_M.size(0)
            _, s = model(cs, rs, tc, tr, emb)
            pred_escalate = (s >= 0.5).float()
            chosen_correct = (1 - pred_escalate) * is_correct_M + pred_escalate * is_correct_D
            total_correct += chosen_correct.sum().item()
            total_escal += s.mean().item() * bs
            n += bs
    acc = total_correct / n if n else 0.0
    escal = total_escal / n if n else 0.0
    return acc, escal


# =============================================================================
# TRAINING FUNCTION (takes pre-created loaders)
# =============================================================================
def train_router_full_defer(
    train_loader, dev_loader, test_loader,
    stage1_index: int = 0,  # only model when using 1 model
    lambda_cost: float = 0.1,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    temperature: float = 1.0,  # new hyperparam (default to 1 for now and can try changing later)
    output_dir: str | None = None,  # if set, save checkpoints and records here (e.g. Modal volume)
    input_mode: str = "full",       # "full" | "embedding_only" | "scalar_only"
    use_wandb: bool = False,
    wandb_project: str = "learn2defer-rm",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_tag = MODE_TAGS[input_mode]
    suffix = f" ({input_mode})" if input_mode != "full" else ""
    print(f"Training Learn-to-Defer (BCE fixed λ){suffix} | λ={lambda_cost} | stage1_index={stage1_index} | temperature={temperature}")

    wb = WandbLogger(
        enabled=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name or f"bce_lambda{lambda_cost}_idx{stage1_index}_{input_mode}",
        group=wandb_group or f"bce_{input_mode}",
        tags=wandb_tags,
        trainer="bce",
        config={
            "trainer": "bce",
            "input_mode": input_mode,
            "stage1_index": stage1_index,
            "lambda_cost": lambda_cost,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "temperature": temperature,
            "device": device,
        },
    )

    model = RouterNet(emb_in_dim=32, embed_out_dim=5, explicit_in_dim=4,
                      stage1_index=stage1_index, input_mode=input_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_dev_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    history = []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = total_escal = total_train_correct = 0.0
        n_seen = 0

        for b in train_loader:
            cs = b["chosen_scores"].to(device)
            rs = b["rejected_scores"].to(device)
            tc = b["tokens_chosen"].to(device)
            tr = b["tokens_rejected"].to(device)
            emb = b["emb_cat"].to(device)
            swapped = b["swapped"].to(device)
            p_D = b["p_D"].to(device)
            y = b["y"].to(device)

            # Network cont loss: p_M = sigmoid(swapped * margin / temp), p_fin = (1-s)*p_M + s*p_D
            stage1_chosen = cs[:, model.stage1_index]
            stage1_rejected = rs[:, model.stage1_index]
            margin = stage1_chosen - stage1_rejected
            p_M = torch.sigmoid(swapped * margin / temperature)

            _, s = model(cs, rs, tc, tr, emb)
            p_fin = (1 - s) * p_M + s * p_D
            p_fin = p_fin.clamp(1e-7, 1.0 - 1e-7)
            bce = F.binary_cross_entropy(p_fin, y, reduction="mean")
            loss = bce + lambda_cost * s.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = len(b["loss_D"])
            batch_escal = s.detach().mean().item()
            total_loss += loss.item() * bs
            total_escal += batch_escal * bs
            is_correct_M = b["is_correct_M"].to(device)
            is_correct_D = b["loss_D"].to(device)
            pred_escalate = (s.detach() >= 0.5).float()
            chosen_correct = (1 - pred_escalate) * is_correct_M + pred_escalate * is_correct_D
            total_train_correct += chosen_correct.sum().item()
            n_seen += bs

        if n_seen > 0:
            avg_loss = total_loss / n_seen
            avg_train_escal = total_escal / n_seen
            train_acc = total_train_correct / n_seen
        else:
            avg_loss = 0.0
            avg_train_escal = 0.0
            train_acc = 0.0
        dev_acc, dev_escal = get_dev_acc_and_escal(model, dev_loader, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_escalation": avg_train_escal,
            "train_accuracy": train_acc,
            "lambda": lambda_cost,
            "dev_accuracy": dev_acc,
            "dev_escalation": dev_escal,
        }
        history.append(epoch_metrics)
        wb.log(
            {
                "loss": avg_loss,
                "train/escalation": avg_train_escal,
                "train/accuracy": train_acc,
                "dev/accuracy": dev_acc,
                "dev/escalation": dev_escal,
                "lambda": lambda_cost,
            },
            step=epoch + 1,
        )

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            evaluate_strategy("learned", model, dev_loader, device, "dev")
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Esc: {avg_train_escal:.1%} | Dev Acc: {dev_acc:.4f}")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch + 1
                patience_counter = 0
                save_name = f"router_best_fixed_λ{lambda_cost}_index{stage1_index}{mode_tag}.pth"
                save_path = os.path.join(output_dir, save_name) if output_dir else save_name
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict(), 'epoch': best_epoch,
                            'dev_acc': best_dev_acc, 'lambda_cost': lambda_cost,
                            'stage1_index': stage1_index}, save_path)
                print(f"   → New best saved: {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    # Save training history CSV
    csv_name = f"train_log_lambda{lambda_cost}_index{stage1_index}{mode_tag}.csv"
    csv_path = os.path.join(output_dir, csv_name) if output_dir else csv_name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_escalation", "train_accuracy", "lambda", "dev_accuracy", "dev_escalation"])
        w.writeheader()
        w.writerows(history)
    print(f"Training log saved → {csv_path}")

    # ====================== FINAL REPORT ======================
    print("\n" + "="*90)
    print("FINAL REPORT — ALL STRATEGIES")
    print("="*90)

    for strat in ["cheap", "heavy", "random", "learned"]:
        print(f"\n{'='*40} {strat.upper()} {'='*40}")
        evaluate_strategy(strat, model, dev_loader, device, "dev")
        evaluate_strategy(strat, model, test_loader, device, "test_rmbench")

    # Save detailed records
    records_name = f"router_learned_records_fixed_λ{lambda_cost}_index{stage1_index}{mode_tag}.json"
    records_path = os.path.join(output_dir, records_name) if output_dir else records_name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_detailed_records("learned", model, test_loader, device, records_path)

    wb.log({"best_dev_accuracy": best_dev_acc, "best_epoch": best_epoch})
    wb.finish()
    return model

# Lambda values to sweep over (run with: modal run scripts/train_bce.py)
LAMBDA_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run_lambda_sweep(
    train_loader, dev_loader, test_loader,
    lambdas: list[float] | None = None,
    stage1_index: int = 0,
    epochs: int = 100,
    patience: int = 100,
    temperature: float = 1.0,
    output_dir: str | None = None,
    input_mode: str = "full",       # "full" | "embedding_only" | "scalar_only"
    use_wandb: bool = False,
    wandb_project: str = "learn2defer-rm",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
):
    """Loop over multiple lambda_cost values, training once per value."""
    if lambdas is None:
        lambdas = LAMBDA_SWEEP
    out_dir = output_dir or LOCAL_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    models = []
    for i, lam in enumerate(lambdas):
        print("\n" + "=" * 90)
        print(f"SWEEP {i+1}/{len(lambdas)} — lambda_cost = {lam}")
        print("=" * 90)
        model = train_router_full_defer(
            train_loader,
            dev_loader,
            test_loader,
            stage1_index=stage1_index,
            lambda_cost=lam,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            output_dir=out_dir,
            input_mode=input_mode,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
        )
        models.append(model)
    return models


def _parse_args():
    """Parse CLI args for local runs."""
    p = argparse.ArgumentParser(description="Train Learn-to-Defer router (BCE fixed lambda)")
    p.add_argument("--stage1-index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default=LOCAL_RESULTS_DIR)
    p.add_argument("--input-mode", type=str, default="full", choices=["full", "embedding_only", "scalar_only"],
                   help="full=both heads, embedding_only=embedding features only, scalar_only=scalar RM features only")
    add_wandb_args(p)
    return p.parse_args()


if __name__ == "__main__":
    # =============================================================================
    # FINAL TRAINING CALL (local)
    # =============================================================================
    args = _parse_args()
    train_loader, dev_loader, test_loader = _get_loaders()
    run_lambda_sweep(
        train_loader,
        dev_loader,
        test_loader,
        stage1_index=args.stage1_index,
        epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        **wandb_kwargs_from_args(args),
    )


# =============================================================================
# MODAL GPU TRAINING — run with: modal run scripts/train_bce.py (from repo root)
# =============================================================================
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    L2D_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "tqdm", "wandb")
        .env({"PYTHONPATH": "/app/src", "L2D_DATA_ROOT": "/app"})
        .add_local_dir(str(REPO_ROOT), remote_path="/app")
    )

    L2D_MODAL_APP = modal.App("l2d-rm-train-bce-fixedlambda")
    L2D_VOLUME_FULL = modal.Volume.from_name("l2d-rm-checkpoints-bcefixed-full", create_if_missing=True)
    L2D_VOLUME_EMBEDDING_ONLY = modal.Volume.from_name("l2d-rm-checkpoints-bcefixed-embeddingonly", create_if_missing=True)
    L2D_VOLUME_SCALAR_ONLY = modal.Volume.from_name("l2d-rm-checkpoints-bcefixed-scalaronly", create_if_missing=True)

    def _get_volume_for_mode(mode: str):
        if mode == "embedding_only":
            return L2D_VOLUME_EMBEDDING_ONLY
        if mode == "scalar_only":
            return L2D_VOLUME_SCALAR_ONLY
        return L2D_VOLUME_FULL

    @L2D_MODAL_APP.function(
        image=L2D_IMAGE,
        gpu="T4",  # or "A10G", "A100"
        timeout=3600 * 10,
        volumes={
            "/checkpoints/full": L2D_VOLUME_FULL,
            "/checkpoints/embedding_only": L2D_VOLUME_EMBEDDING_ONLY,
            "/checkpoints/scalar_only": L2D_VOLUME_SCALAR_ONLY,
        },
    )
    def train_on_modal(
        stage1_index: int = 0,
        epochs: int = 100,
        patience: int = 100,
        temperature: float = 1.0,
        input_mode: str = "full",  # "full" | "embedding_only" | "scalar_only"
    ):
        import sys

        os.environ["L2D_DATA_ROOT"] = "/app"
        sys.path.insert(0, "/app/src")

        import train.train_bce as train_mod

        # Include input_mode in experiment path (writes to mode-specific volume)
        out_dir = f"/checkpoints/{input_mode}"
        train_loader, dev_loader, test_loader = train_mod._get_loaders()
        train_mod.run_lambda_sweep(
            train_loader,
            dev_loader,
            test_loader,
            stage1_index=stage1_index,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            output_dir=out_dir,
            input_mode=input_mode,
        )
        vol = _get_volume_for_mode(input_mode)
        vol.commit()
        vol_name = f"l2d-rm-checkpoints-bcefixed-{input_mode}"
        print(f"Checkpoints and records saved to volume '{vol_name}' (path {out_dir})")

    @L2D_MODAL_APP.local_entrypoint()
    def main(
        stage1_index: int = 0,
        epochs: int = 100,
        patience: int = 100,
        temperature: float = 1.0,
        input_mode: str = "full",  # "full" | "embedding_only" | "scalar_only"
    ):
        train_on_modal.remote(
            stage1_index=stage1_index,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            input_mode=input_mode,
        )