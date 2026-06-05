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

        model = getattr(trainer, "_actor_model", None) or getattr(trainer, "_draft_model", None)

        if model is not None:
            state["model_state_dict"] = {
                k: v.cpu() for k, v in model.state_dict().items()
            }

        opt = trainer._optimizer
        if opt is not None:
            state["optimizer_state_dict"] = opt.state_dict()
            if hasattr(opt, "fp32_params"):
                state["fp32_params"] = [p.data.cpu().clone() for p in opt.fp32_params]
            if hasattr(opt, "scheduler"):
                state["scheduler_last_epoch"] = opt.scheduler.last_epoch

        if rank == 0:
            n_model = len(state.get("model_state_dict", {}))
            n_fp32 = len(state.get("fp32_params", []))
            logger.info(
                "Saving checkpoint step=%d: %d model keys, %d fp32 params, opt=%s, sched_epoch=%s",
                step, n_model, n_fp32,
                "yes" if "optimizer_state_dict" in state else "no",
                state.get("scheduler_last_epoch", "N/A"),
            )
            self._manager.save(state, path, step)
            self._prune_old_checkpoints(ckpt_dir)

        if trainer._is_distributed:
            torch.distributed.barrier()

    @staticmethod
    def _verify_checkpoint(path: Path, model, opt, step: int) -> None:
        """Load saved checkpoint back and compare against live model weights."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            saved = payload.get("state_dict", payload)
            saved_sd = saved.get("model_state_dict", {})

            live_sd = {k: v.cpu() for k, v in model.state_dict().items()}

            if len(saved_sd) != len(live_sd):
                logger.error("CKPT VERIFY FAIL step=%d: saved %d keys vs live %d keys",
                             step, len(saved_sd), len(live_sd))
                return

            max_diff = 0.0
            for k in live_sd:
                if k not in saved_sd:
                    logger.error("CKPT VERIFY FAIL step=%d: key %s missing from checkpoint", step, k)
                    return
                diff = (saved_sd[k].float() - live_sd[k].float()).abs().max().item()
                max_diff = max(max_diff, diff)

            if max_diff > 1e-6:
                logger.error("CKPT VERIFY FAIL step=%d: max diff=%.6g (should be 0)", step, max_diff)
            else:
                logger.info("CKPT VERIFY OK step=%d: %d keys, max_diff=%.2e", step, len(saved_sd), max_diff)

            saved_fp32 = saved.get("fp32_params", [])
            if saved_fp32 and opt is not None and hasattr(opt, "fp32_params"):
                fp32_max_diff = 0.0
                for sp, lp in zip(saved_fp32, opt.fp32_params):
                    diff = (sp.cpu().float() - lp.data.cpu().float()).abs().max().item()
                    fp32_max_diff = max(fp32_max_diff, diff)
                logger.info("CKPT VERIFY FP32 step=%d: %d params, max_diff=%.2e",
                            step, len(saved_fp32), fp32_max_diff)

        except Exception as exc:
            logger.warning("CKPT VERIFY ERROR step=%d: %s", step, exc)

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
    """Run periodic validation using the trainer hook.

    Must run on **all ranks** because teacher forward + FSDP require
    collective operations.  Only rank 0 logs the results.
    """

    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.interval != 0:
            return
        val_metrics = trainer.run_validation()
        if trainer._rank == 0:
            parts = [f"{k}={v:.6g}" for k, v in sorted(val_metrics.items())]
            logger.info("eval step=%d %s", step, " ".join(parts))
        metrics.update(val_metrics)


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
