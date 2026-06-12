"""Optional Weights & Biases logging for router training."""

from __future__ import annotations

import argparse
import os
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


def wandb_available() -> bool:
    return wandb is not None


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    group.add_argument("--wandb-project", type=str, default="learn2defer-rm")
    group.add_argument("--wandb-entity", type=str, default=None)
    group.add_argument("--wandb-run-name", type=str, default=None)
    group.add_argument("--wandb-group", type=str, default=None, help="Group related sweep runs together")
    group.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")


def wandb_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    tags = None
    if getattr(args, "wandb_tags", None):
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    return {
        "use_wandb": bool(getattr(args, "wandb", False)),
        "wandb_project": getattr(args, "wandb_project", "learn2defer-rm"),
        "wandb_entity": getattr(args, "wandb_entity", None),
        "wandb_run_name": getattr(args, "wandb_run_name", None),
        "wandb_group": getattr(args, "wandb_group", None),
        "wandb_tags": tags,
    }


class WandbLogger:
    """Thin wrapper so training works when wandb is not installed."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        project: str = "learn2defer-rm",
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        trainer: str = "train",
    ):
        self.enabled = enabled and wandb_available() and not os.environ.get("WANDB_DISABLED")
        self._run = None
        if not self.enabled:
            if enabled and not wandb_available():
                print("wandb logging requested but wandb is not installed; skipping.")
            return

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
            job_type=trainer,
            reinit=True,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            wandb.finish()
            self._run = None
