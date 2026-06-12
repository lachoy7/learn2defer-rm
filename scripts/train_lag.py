#!/usr/bin/env python3
"""Train the learn-to-defer router with Lagrangian hinge-style loss.

Local:  python scripts/train_lag.py
Modal:  modal run scripts/train_lag.py
"""
import _bootstrap  # noqa: F401

from train.train_lag import *  # noqa: F401,F403

if __name__ == "__main__":
    import train.train_lag as mod

    args = mod._parse_args()
    train_loader, dev_loader, test_loader = mod._get_loaders()
    mod.run_target_escalation_sweep(
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
        **mod.wandb_kwargs_from_args(args),
    )
