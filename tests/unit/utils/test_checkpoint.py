from __future__ import annotations

import torch

from lumenrl.utils.checkpoint import CheckpointManager


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.pt"
    state = {"w": torch.tensor([1.0, 2.0]), "n": 3}
    CheckpointManager.save(state, path, step=7)
    payload = CheckpointManager.load(path)
    assert payload["step"] == 7
    assert torch.equal(payload["state_dict"]["w"], state["w"])
    assert payload["state_dict"]["n"] == 3


def test_get_latest(tmp_path) -> None:
    CheckpointManager.save({"a": 1}, tmp_path / "checkpoint_5.pt", step=5)
    CheckpointManager.save({"a": 2}, tmp_path / "checkpoint_20.pt", step=20)
    latest = CheckpointManager.get_latest(tmp_path)
    assert latest is not None
    assert "checkpoint_20.pt" in latest
    loaded = CheckpointManager.load(latest)
    assert loaded["state_dict"]["a"] == 2
