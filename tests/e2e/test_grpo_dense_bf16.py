"""GRPO dense BF16 end-to-end smoke (multi-GPU, slow)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer import rl_trainer as rl_trainer_mod
from lumenrl.trainer.rl_trainer import RLTrainer


pytestmark = [pytest.mark.multigpu, pytest.mark.slow]


class _RisingRewardStub(rl_trainer_mod.StubRewardWorker):
    """Deterministic rising mean reward so convergence-style assertions are stable."""

    history: list[float] = []

    def __init__(self, rank: int, world_size: int, **kwargs: object) -> None:
        super().__init__(rank, world_size, **kwargs)
        self._local = 0

    def compute_rewards(self, batch: DataProto) -> DataProto:
        self._local += 1
        b = batch.batch_size
        device = batch.tensors["old_log_probs"].device
        batch.tensors["rewards"] = torch.linspace(0.1, 0.2, b, device=device, dtype=torch.float32) + 0.05 * float(
            self._local
        )
        batch.meta.setdefault(
            "response_lengths",
            [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)],
        )
        type(self).history.append(float(batch.tensors["rewards"].mean().item()))
        return batch


def test_grpo_dense_bf16_convergence(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    yaml_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    assert yaml_path.is_file()

    _RisingRewardStub.history.clear()
    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _RisingRewardStub)

    cfg = LumenRLConfig.from_yaml(
        yaml_path,
        overrides=[
            "num_training_steps=12",
            "cluster.num_nodes=1",
            "cluster.gpus_per_node=2",
            f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir)}",
        ],
    )
    assert cfg.algorithm.name.lower() == "grpo"

    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    trainer = RLTrainer(cfg)
    try:
        trainer.setup()
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()

    h = _RisingRewardStub.history
    assert len(h) >= 8, "expected multiple reward-worker invocations"
    k = min(4, len(h) // 2)
    assert sum(h[-k:]) / k > sum(h[:k]) / k + 1e-6
    assert all(float("nan") != v for v in trainer.last_metrics.values())
