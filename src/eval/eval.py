# =============================================================================
# UNIFIED EVALUATION — ALL STRATEGIES with domain + easy/normal/hard accuracies
# =============================================================================
from collections import defaultdict

import torch

def evaluate_strategy(
    strategy: str,
    model,
    loader,
    device,
    split_name="dev",
    seed: int = 42
):
    model.eval() if strategy == "learned" else None
    torch.manual_seed(seed)

    total_correct = total_samples = 0.0
    group_correct = defaultdict(float)
    group_total = defaultdict(int)
    diff_correct = defaultdict(float)
    diff_total = defaultdict(int)

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

            if strategy == "learned":
                _, s = model(cs, rs, tc, tr, emb)
            elif strategy == "cheap":
                s = torch.zeros(bs, device=device)
            elif strategy == "heavy":
                s = torch.ones(bs, device=device)
            elif strategy == "random":
                s = torch.rand(bs, device=device)

            pred_escalate = (s >= 0.5).float()
            chosen_correct = (1 - pred_escalate) * is_correct_M + pred_escalate * is_correct_D

            total_correct += chosen_correct.sum().item()
            total_samples += bs

            # Grouping
            keys = b.get("source" if split_name == "dev" else "domain", ["overall"] * bs)
            for i in range(bs):
                key = keys[i]
                group_correct[key] += chosen_correct[i].item()
                group_total[key] += 1

            # Difficulty (test_rmbench only)
            if split_name.startswith("test") and "difficulty" in b:
                for i in range(bs):
                    d = b["difficulty"][i]
                    diff_correct[d] += chosen_correct[i].item()
                    diff_total[d] += 1

    overall = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"\n=== {split_name.upper()} — {strategy.upper()} ===")
    print(f"Overall Acc: {overall:.4f} ({int(total_samples)} samples)")

    if group_total:
        print("  Per Group:")
        for k in sorted(group_total.keys()):
            acc = group_correct[k] / group_total[k]
            print(f"    {k:25s}: {acc:.4f} ({group_total[k]} samples)")

    if diff_total:
        print("  Per Difficulty:")
        for d in ["easy", "normal", "hard"]:
            if d in diff_total:
                acc = diff_correct[d] / diff_total[d]
                print(f"    {d.capitalize():10s}: {acc:.4f} ({diff_total[d]} samples)")

    return overall