"""The superseded async_training trainer says what to use instead."""

from __future__ import annotations

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import async_trainer
from lumenrl.trainer.async_trainer import AsyncRLTrainer


def test_async_training_points_to_the_ray_async_modes(tmp_path, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        async_trainer.logger, "warning", lambda msg, *args: warnings.append(msg % args),
    )
    config = LumenRLConfig()
    config.async_training.enabled = True
    config.async_training.weight_sync_dir = str(tmp_path)

    AsyncRLTrainer(config)

    [warning] = warnings
    assert "never loads updated weights" in warning
    assert "controller.ray.fully_async" in warning
    assert "controller.ray.one_step_off_policy" in warning
