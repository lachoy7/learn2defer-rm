# =============================================================================
# IMPORTS — all dependencies at top (stdlib, third-party, local)
# =============================================================================
import argparse
import csv
import json
import os

import torch
from tqdm import tqdm

from dataset.data_loaders import get_router_dataloaders
from eval.eval import evaluate_strategy
from train.model import MODE_TAGS, RouterNet
from utils.paths import REPO_ROOT, RESULTS_DIR
from utils.wandb_logging import WandbLogger, add_wandb_args, wandb_kwargs_from_args

LOCAL_RESULTS_DIR = str(RESULTS_DIR)


def _get_loaders():
    return get_router_dataloaders(use_swapped_pd=False)

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
    # Primal–dual (Lagrangian) constraint settings
    target_escalation: float = 0.25,   # desired E[s] (escalation rate)
    lambda_init: float = 0.0,          # initial dual variable
    lambda_lr: float = 0.05,           # dual learning rate
    lambda_max: float = 100.0,         # clamp for stability
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 100,
    temperature: float = 1.0,  # new hyperparam (default to 1 for now and can try changing later)
    output_dir: str | None = None,  # if set, save checkpoints and records here (e.g. Modal volume)
    input_mode: str = "full",
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
    print(
        f"Training Learn-to-Defer{suffix} | target_escal={target_escalation:.1%} | "
        f"λ0={lambda_init} | λ_lr={lambda_lr} | stage1_index={stage1_index} | temperature={temperature}"
    )

    wb = WandbLogger(
        enabled=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name or f"lag_target{target_escalation:.2f}_idx{stage1_index}_{input_mode}",
        group=wandb_group or f"lag_{input_mode}",
        tags=wandb_tags,
        trainer="lag",
        config={
            "trainer": "lag",
            "input_mode": input_mode,
            "stage1_index": stage1_index,
            "target_escalation": target_escalation,
            "lambda_init": lambda_init,
            "lambda_lr": lambda_lr,
            "lambda_max": lambda_max,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "temperature": temperature,
            "device": device,
        },
    )

    model = RouterNet(
        emb_in_dim=32, embed_out_dim=5, explicit_in_dim=4,
        stage1_index=stage1_index, input_mode=input_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Dual variable (Lagrange multiplier) for the escalation-budget constraint
    lambda_cost = float(lambda_init)

    best_dev_acc = 0.0
    best_escal_diff = float("inf")  # min |dev_escal - target| when target not met yet
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
            is_wrong_M = b["is_wrong_M"].to(device)  # Still load 0-1 for reference and potential fallback later if needed
            is_wrong_D = (1.0 - b["loss_D"]).to(device)

            # temperature is new hyperparam
            stage1_chosen = b["chosen_scores"][:, model.stage1_index].to(device)
            stage1_rejected = b["rejected_scores"][:, model.stage1_index].to(device)
            margin = stage1_chosen - stage1_rejected
            continuous_wrong_M = torch.sigmoid(-margin / temperature)  # temp controls sharpeness of sigmoid (high means more gradual, low more step-like as in 0/1)

            # Updated loss (uses continuous_wrong_M)
            _, s = model(cs, rs, tc, tr, emb)
            loss = ((1 - s) * continuous_wrong_M + s * is_wrong_D + float(lambda_cost) * s).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = len(b["loss_D"])
            total_loss += loss.item() * bs
            total_escal += s.detach().mean().item() * bs
            is_correct_M = (1.0 - is_wrong_M).float()
            is_correct_D = b["loss_D"].to(device)
            pred_escalate = (s.detach() >= 0.5).float()
            chosen_correct = (1 - pred_escalate) * is_correct_M + pred_escalate * is_correct_D
            total_train_correct += chosen_correct.sum().item()
            n_seen += bs

        # Dual ascent step to satisfy E[s] <= target_escalation
        # If escalation is above budget, increase lambda; if below, decrease.
        if n_seen > 0:
            avg_escal = total_escal / n_seen
            lambda_cost = max(0.0, min(lambda_max, lambda_cost + lambda_lr * (avg_escal - target_escalation)))
            train_acc = total_train_correct / n_seen
            avg_loss = total_loss / n_seen
        else:
            avg_escal = 0.0
            train_acc = 0.0
            avg_loss = 0.0

        dev_acc, dev_escal = get_dev_acc_and_escal(model, dev_loader, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_escalation": avg_escal,
            "train_accuracy": train_acc,
            "lambda": lambda_cost,
            "dev_accuracy": dev_acc,
            "dev_escalation": dev_escal,
        }
        history.append(epoch_metrics)
        wb.log(
            {
                "loss": avg_loss,
                "train/escalation": avg_escal,
                "train/accuracy": train_acc,
                "dev/accuracy": dev_acc,
                "dev/escalation": dev_escal,
                "lambda": lambda_cost,
            },
            step=epoch + 1,
        )

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            escalation_met = dev_escal <= target_escalation

            print(
                f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader.dataset):.4f} | "
                f"Esc: {avg_escal:.1%} | λ: {lambda_cost:.4f} | Dev Acc: {dev_acc:.4f} | Dev Esc: {dev_escal:.1%}"
            )

            if escalation_met:
                # Target escalation met: track best by dev accuracy; early stopping is active
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    save_name = f"router_best_dynamic_target{target_escalation:.2f}_index{stage1_index}{mode_tag}.pth"
                    save_path = os.path.join(output_dir, save_name) if output_dir else save_name
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': best_epoch,
                                'dev_acc': best_dev_acc, 'dev_escal': dev_escal, 'lambda_cost': lambda_cost,
                                'stage1_index': stage1_index}, save_path)
                    print(f"   → New best (acc) saved: {save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping (escalation met, no dev acc improvement)")
                        break
            else:
                # Target escalation not met yet: save when dev escalation is closest to target
                escal_diff = abs(dev_escal - target_escalation)
                if escal_diff < best_escal_diff:
                    best_escal_diff = escal_diff
                    best_epoch = epoch + 1
                    save_name = f"router_best_dynamic_target{target_escalation:.2f}_index{stage1_index}{mode_tag}.pth"
                    save_path = os.path.join(output_dir, save_name) if output_dir else save_name
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': best_epoch,
                                'dev_acc': dev_acc, 'dev_escal': dev_escal, 'lambda_cost': lambda_cost,
                                'stage1_index': stage1_index}, save_path)
                    print(f"   → New best (closest escal to target) saved: {save_path}")
                # Early stopping not active until escalation is met; do not increment patience_counter

    # Save training history CSV
    csv_name = f"train_log_target{target_escalation:.2f}_index{stage1_index}{mode_tag}.csv"
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
    records_name = f"router_learned_records_dynamic_target{target_escalation:.2f}_index{stage1_index}{mode_tag}.json"
    records_path = os.path.join(output_dir, records_name) if output_dir else records_name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_detailed_records("learned", model, test_loader, device, records_path)

    wb.log({"best_dev_accuracy": best_dev_acc, "best_epoch": best_epoch})
    wb.finish()
    return model

# Target escalation values to sweep over (run with: modal run scripts/train_lag.py)
TARGET_ESCALATION_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run_target_escalation_sweep(
    train_loader, dev_loader, test_loader,
    target_escalations: list[float] | None = None,
    stage1_index: int = 0,
    lambda_init: float = 0.0,
    lambda_lr: float = 0.05,
    lambda_max: float = 100.0,
    epochs: int = 100,
    patience: int = 100,
    temperature: float = 1.0,
    output_dir: str | None = None,
    input_mode: str = "full",
    use_wandb: bool = False,
    wandb_project: str = "learn2defer-rm",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
):
    """Loop over multiple target_escalation values, training once per value."""
    if target_escalations is None:
        target_escalations = TARGET_ESCALATION_SWEEP
    out_dir = output_dir or LOCAL_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    models = []
    for i, target_escal in enumerate(target_escalations):
        print("\n" + "=" * 90)
        print(f"SWEEP {i+1}/{len(target_escalations)} — target_escalation = {target_escal:.2f}")
        print("=" * 90)
        model = train_router_full_defer(
            train_loader,
            dev_loader,
            test_loader,
            stage1_index=stage1_index,
            target_escalation=target_escal,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
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
    p = argparse.ArgumentParser(description="Train Learn-to-Defer router (Lagrangian)")
    p.add_argument("--stage1-index", type=int, default=0)
    p.add_argument("--lambda-init", type=float, default=0.0)
    p.add_argument("--lambda-lr", type=float, default=0.05)
    p.add_argument("--lambda-max", type=float, default=100.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default=LOCAL_RESULTS_DIR)
    p.add_argument(
        "--input-mode", type=str, default="full",
        choices=["full", "embedding_only", "scalar_only"],
        help="full=both heads, embedding_only=response embeddings, scalar_only=scalar RM features",
    )
    add_wandb_args(p)
    return p.parse_args()


if __name__ == "__main__":
    # =============================================================================
    # FINAL TRAINING CALL (local)
    # =============================================================================
    args = _parse_args()
    train_loader, dev_loader, test_loader = _get_loaders()
    run_target_escalation_sweep(
        train_loader,
        dev_loader,
        test_loader,
        stage1_index=args.stage1_index,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        lambda_max=args.lambda_max,
        epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        **wandb_kwargs_from_args(args),
    )


# =============================================================================
# MODAL GPU TRAINING — run with: modal run scripts/train_lag.py (from repo root)
# =============================================================================
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    L2D_MODAL_APP = modal.App("l2d-rm-train-lag")

    L2D_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "tqdm", "wandb")
        .env({"PYTHONPATH": "/app/src", "L2D_DATA_ROOT": "/app"})
        .add_local_dir(str(REPO_ROOT), remote_path="/app")
    )

    L2D_VOLUME = modal.Volume.from_name("l2d-rm-checkpoints", create_if_missing=True)

    @L2D_MODAL_APP.function(
        image=L2D_IMAGE,
        gpu="T4",  # or "A10G", "A100"
        timeout=3600 * 3,
        volumes={"/checkpoints": L2D_VOLUME},
    )
    def train_on_modal(
        stage1_index: int = 0,
        lambda_init: float = 0.0,
        lambda_lr: float = 0.05,
        lambda_max: float = 100.0,
        epochs: int = 100,
        patience: int = 100,
        temperature: float = 1.0,
        input_mode: str = "full",
    ):
        import sys

        os.environ["L2D_DATA_ROOT"] = "/app"
        sys.path.insert(0, "/app/src")

        import train.train_lag as train_mod

        out_dir = "/checkpoints"
        train_loader, dev_loader, test_loader = train_mod._get_loaders()
        train_mod.run_target_escalation_sweep(
            train_loader,
            dev_loader,
            test_loader,
            stage1_index=stage1_index,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            output_dir=out_dir,
            input_mode=input_mode,
        )
        L2D_VOLUME.commit()
        print(f"Checkpoints and records saved to volume 'l2d-rm-checkpoints' (path {out_dir})")

    @L2D_MODAL_APP.local_entrypoint()
    def main(
        stage1_index: int = 0,
        lambda_init: float = 0.0,
        lambda_lr: float = 0.05,
        lambda_max: float = 100.0,
        epochs: int = 100,
        patience: int = 100,
        temperature: float = 1.0,
        input_mode: str = "full",
    ):
        train_on_modal.remote(
            stage1_index=stage1_index,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            input_mode=input_mode,
        )