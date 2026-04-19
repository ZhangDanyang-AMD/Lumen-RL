"""Training loop callbacks."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lumenrl.utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from lumenrl.trainer.rl_trainer import RLTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Hook points for the RL training loop."""

    def on_train_begin(self, trainer: "RLTrainer") -> None:
        """Invoked once before the first training step."""

    def on_train_end(self, trainer: "RLTrainer") -> None:
        """Invoked once after training completes."""

    def on_step_begin(self, trainer: "RLTrainer", step: int) -> None:
        """Invoked at the beginning of each optimizer/rollout step."""

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        """Invoked after ``weight_sync`` with metrics from the finished step."""


class LoggingCallback(Callback):
    """Emit structured logs for rolling metrics every ``interval`` steps."""

    def __init__(self, interval: int = 1) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.interval != 0:
            return
        if trainer._rank != 0:
            return
        parts = [f"{k}={v:.6g}" for k, v in sorted(metrics.items())]
        logger.info("step=%d %s", step, " ".join(parts))


class CheckpointCallback(Callback):
    """Save full training state (model, optimizer, step) for crash recovery.

    Uses FSDP2-aware ``get_model_state_dict`` / ``get_optimizer_state_dict``
    when distributed, falling back to plain ``state_dict()`` otherwise.
    Only rank 0 writes to disk.  Old checkpoints beyond ``save_total_limit``
    are pruned automatically.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int,
        save_total_limit: int = 3,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = max(1, int(save_interval))
        self.save_total_limit = max(1, int(save_total_limit))
        self._manager = CheckpointManager()

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.save_interval != 0:
            return

        rank = trainer._rank
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"checkpoint_{step}.pt"

        state: dict[str, Any] = {
            "step": step,
            "metrics": metrics,
            "algo": trainer.config.algorithm.name,
        }

        if trainer._is_distributed:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            try:
                from torch.distributed.checkpoint.state_dict import (
                    get_model_state_dict,
                    get_optimizer_state_dict,
                    StateDictOptions,
                )
                opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                state["model_state_dict"] = get_model_state_dict(
                    trainer._actor_model, options=opts,
                )
                state["optimizer_state_dict"] = get_optimizer_state_dict(
                    trainer._actor_model, trainer._optimizer, options=opts,
                )
            except Exception as exc:
                logger.warning("FSDP2 state_dict extraction failed (%s); saving metrics only.", exc)
        else:
            if trainer._actor_model is not None:
                state["model_state_dict"] = trainer._actor_model.state_dict()
            if trainer._optimizer is not None:
                state["optimizer_state_dict"] = trainer._optimizer.state_dict()

        if rank == 0:
            self._manager.save(state, path, step)
            self._prune_old_checkpoints(ckpt_dir)

        if trainer._is_distributed:
            torch.distributed.barrier()

    def _prune_old_checkpoints(self, ckpt_dir: Path) -> None:
        pattern = re.compile(r"checkpoint_(\d+)\.pt$")
        ckpts: list[tuple[int, Path]] = []
        for p in ckpt_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort(key=lambda x: x[0])
        while len(ckpts) > self.save_total_limit:
            _, old = ckpts.pop(0)
            try:
                old.unlink()
                logger.info("Pruned old checkpoint: %s", old)
            except OSError:
                pass


class EvalCallback(Callback):
    """Run periodic validation using the trainer hook."""

    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.interval != 0:
            return
        if trainer._rank != 0:
            return
        val_metrics = trainer.run_validation()
        logger.info("validation step=%d %s", step, val_metrics)


class WandbCallback(Callback):
    """Optional Weights & Biases logging."""

    def __init__(self, project: str, name: str = "", entity: str | None = None) -> None:
        self.project = project
        self.name = name
        self.entity = entity
        self._wandb: Any = None
        self._enabled = False

    def on_train_begin(self, trainer: "RLTrainer") -> None:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; WandbCallback disabled.")
            return
        self._wandb = wandb
        run_name = self.name or f"lumenrl-{trainer.config.algorithm.name}"
        wandb.init(project=self.project, name=run_name, entity=self.entity)
        self._enabled = True
        wandb.config.update(
            {
                "algorithm": trainer.config.algorithm.name,
                "num_training_steps": int(trainer.config.num_training_steps),
            },
            allow_val_change=True,
        )

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if not self._enabled or self._wandb is None:
            return
        if trainer._rank != 0:
            return
        payload = {f"train/{k}": v for k, v in metrics.items()}
        payload["train/global_step"] = step
        self._wandb.log(payload, step=step)

    def on_train_end(self, trainer: "RLTrainer") -> None:
        if self._enabled and self._wandb is not None:
            self._wandb.finish()
