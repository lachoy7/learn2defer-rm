#!/usr/bin/env python3
"""Train the learn-to-defer router with fixed-lambda BCE loss.

Local:  python scripts/train_bce.py
Modal:  modal run scripts/train_bce.py
"""
import _bootstrap  # noqa: F401

from train.train_bce import *  # noqa: F401,F403

if __name__ == "__main__":
    import train.train_bce as mod

    args = mod._parse_args()
    train_loader, dev_loader, test_loader = mod._get_loaders()
    mod.run_lambda_sweep(
        train_loader,
        dev_loader,
        test_loader,
        stage1_index=args.stage1_index,
        epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        **mod.wandb_kwargs_from_args(args),
    )
