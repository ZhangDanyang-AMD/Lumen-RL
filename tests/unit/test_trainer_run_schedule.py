from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lumenrl.trainer.callbacks import (
    CheckpointCallback,
    EvalCallback,
    LoggingCallback,
)
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer


def test_eval_interval_runs_after_500_completed_steps() -> None:
    calls: list[int] = []

    class _Trainer:
        _rank = 0

        @staticmethod
        def run_validation():
            calls.append(1)
            return {"eval/loss": 1.0}

    callback = EvalCallback(interval=500, interval_on_completed_steps=True)
    metrics: dict[str, float] = {}
    callback.on_step_end(_Trainer(), 0, metrics)
    callback.on_step_end(_Trainer(), 498, metrics)
    assert calls == []

    callback.on_step_end(_Trainer(), 499, metrics)
    assert calls == [1]
    assert metrics["eval/loss"] == 1.0


def test_eval_callback_preserves_legacy_zero_based_default() -> None:
    calls: list[int] = []

    class _Trainer:
        _rank = 0

        @staticmethod
        def run_validation():
            calls.append(1)
            return {}

    EvalCallback(interval=500).on_step_end(_Trainer(), 0, {})
    assert calls == [1]


def test_logging_callback_adds_training_acceptance_length() -> None:
    metrics = {
        "step_0_acc": 0.8,
        "step_1_acc": 0.5,
        "step_2_acc": 0.25,
    }

    LoggingCallback(interval=1).on_step_end(
        SimpleNamespace(_rank=0), 0, metrics
    )
    assert metrics["simulated_acc_len"] == pytest.approx(1.3)


def test_checkpoint_interval_runs_after_500_completed_steps(tmp_path: Path) -> None:
    callback = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        save_interval=500,
        save_total_limit=10,
    )
    saved: list[int] = []
    callback._save_standard_checkpoint = (
        lambda _trainer, step, _metrics: saved.append(step)
    )
    trainer = SimpleNamespace(_use_ray_controller=False, _actor_wg=None)

    callback.on_step_end(trainer, 198, {})
    callback.on_step_end(trainer, 498, {})
    assert saved == []

    callback.on_step_end(trainer, 499, {})
    callback.on_step_end(trainer, 999, {})
    assert saved == [500, 1000]


def test_formal_run_forces_final_eval_and_checkpoint(tmp_path: Path) -> None:
    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._rank = 0
    trainer.last_metrics = {"loss": 1.0}
    eval_callback = EvalCallback(
        interval=1500, interval_on_completed_steps=True
    )
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        save_interval=1500,
    )
    saved: list[int] = []
    checkpoint_callback.save_now = (
        lambda _trainer, step, _metrics: saved.append(step)
    )
    trainer.callbacks = [eval_callback, checkpoint_callback]
    trainer.run_validation = lambda: {"eval/loss": 0.5}

    trainer._finalize_periodic_artifacts(57849)

    assert trainer.last_metrics["eval/loss"] == 0.5
    assert saved == [57849]


def test_finalize_does_not_duplicate_completed_step_eval() -> None:
    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._rank = 0
    trainer.last_metrics = {}
    trainer.callbacks = [
        EvalCallback(interval=500, interval_on_completed_steps=True)
    ]
    calls: list[int] = []
    trainer.run_validation = lambda: calls.append(1) or {}

    trainer._finalize_periodic_artifacts(500)
    assert calls == []


def test_finalize_does_not_duplicate_completed_step_checkpoint(tmp_path: Path) -> None:
    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._rank = 0
    trainer.last_metrics = {}
    callback = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        save_interval=500,
    )
    saved: list[int] = []
    callback.save_now = (
        lambda _trainer, step, _metrics: saved.append(step)
    )
    trainer.callbacks = [callback]

    trainer._finalize_periodic_artifacts(500)
    assert saved == []
