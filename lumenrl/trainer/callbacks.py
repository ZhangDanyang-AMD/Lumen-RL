"""Training loop callbacks."""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

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
        parts = [f"{k}={v:.6g}" for k, v in sorted(metrics.items())]
        logger.info("step=%d %s", step, " ".join(parts))


class CheckpointCallback(Callback):
    """Persist checkpoints using :class:`~lumenrl.utils.checkpoint.CheckpointManager`."""

    def __init__(self, checkpoint_dir: str, save_interval: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = max(1, int(save_interval))
        self._manager = CheckpointManager()

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.save_interval != 0:
            return
        path = f"{self.checkpoint_dir.rstrip('/')}/checkpoint_{step}.pt"
        state = {"metrics": metrics, "step": step, "algo": trainer.config.algorithm.name}
        self._manager.save(state, path, step)


class EvalCallback(Callback):
    """Run periodic validation using the trainer hook."""

    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(self, trainer: "RLTrainer", step: int, metrics: dict[str, float]) -> None:
        if step % self.interval != 0:
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
        payload = {f"train/{k}": v for k, v in metrics.items()}
        payload["train/global_step"] = step
        self._wandb.log(payload, step=step)

    def on_train_end(self, trainer: "RLTrainer") -> None:
        if self._enabled and self._wandb is not None:
            self._wandb.finish()
