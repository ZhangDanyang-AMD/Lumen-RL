from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lumenrl.trainer.callbacks import CheckpointCallback
from lumenrl.utils.checkpoint import (
    CheckpointManager,
    capture_distributed_rng_state,
    capture_rng_state,
    restore_distributed_rng_state,
    restore_rng_state,
)


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.pt"
    state = {"w": torch.tensor([1.0, 2.0]), "n": 3}
    CheckpointManager.save(state, path, step=7)
    payload = CheckpointManager.load(path)
    assert payload["step"] == 7
    assert torch.equal(payload["state_dict"]["w"], state["w"])
    assert payload["state_dict"]["n"] == 3


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch) -> None:
    path = tmp_path / "checkpoint_5.pt"
    CheckpointManager.save({"a": 1}, path, step=5)

    def _fail_after_partial_write(_payload, output_path) -> None:
        output_path.write_bytes(b"partial")
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", _fail_after_partial_write)
    with pytest.raises(OSError, match="disk write failed"):
        CheckpointManager.save({"a": 2}, path, step=5)

    assert CheckpointManager.load(path)["state_dict"]["a"] == 1
    assert not list(tmp_path.glob("*.incomplete"))


def test_get_latest(tmp_path) -> None:
    CheckpointManager.save({"a": 1}, tmp_path / "checkpoint_5.pt", step=5)
    CheckpointManager.save({"a": 2}, tmp_path / "checkpoint_20.pt", step=20)
    latest = CheckpointManager.get_latest(tmp_path)
    assert latest is not None
    assert "checkpoint_20.pt" in latest
    loaded = CheckpointManager.load(latest)
    assert loaded["state_dict"]["a"] == 2


def test_rng_state_roundtrip() -> None:
    original = capture_rng_state()
    try:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        captured = capture_rng_state()

        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(4)

        random.seed(21)
        np.random.seed(22)
        torch.manual_seed(23)
        restore_rng_state(captured)

        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        assert torch.equal(torch.rand(4), expected_torch)
    finally:
        restore_rng_state(original)


def test_distributed_rng_state_requires_matching_world_size() -> None:
    bundle = capture_distributed_rng_state(distributed=False)
    assert bundle["world_size"] == 1
    assert len(bundle["states"]) == 1

    with pytest.raises(RuntimeError, match="topology mismatch"):
        restore_distributed_rng_state(bundle, rank=0, world_size=2)


def test_checkpoint_callback_saves_rng_state(tmp_path) -> None:
    trainer = SimpleNamespace(
        _rank=0,
        _is_distributed=False,
        _actor_model=None,
        _draft_model=None,
        _optimizer=None,
        config=SimpleNamespace(algorithm=SimpleNamespace(name="spec_distill")),
    )
    callback = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        save_interval=1,
        save_total_limit=1,
    )
    callback.save_now(trainer, step=1, metrics={})

    payload = CheckpointManager.load(tmp_path / "checkpoint_1.pt")
    rng_state = payload["state_dict"]["rng_state"]
    assert rng_state["world_size"] == 1
    assert len(rng_state["states"]) == 1


def test_failed_callback_save_does_not_prune_valid_checkpoints(
    tmp_path, monkeypatch
) -> None:
    for step in (1, 2):
        CheckpointManager.save(
            {"step": step}, tmp_path / f"checkpoint_{step}.pt", step
        )
    trainer = SimpleNamespace(
        _rank=0,
        _is_distributed=False,
        _actor_model=None,
        _draft_model=None,
        _optimizer=None,
        config=SimpleNamespace(algorithm=SimpleNamespace(name="spec_distill")),
    )
    callback = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        save_interval=1,
        save_total_limit=2,
    )

    def _fail_save(*_args, **_kwargs) -> None:
        raise OSError("disk write failed")

    monkeypatch.setattr(callback._manager, "save", _fail_save)
    with pytest.raises(OSError, match="disk write failed"):
        callback.save_now(trainer, step=3, metrics={})

    assert sorted(path.name for path in tmp_path.glob("checkpoint_*.pt")) == [
        "checkpoint_1.pt",
        "checkpoint_2.pt",
    ]
