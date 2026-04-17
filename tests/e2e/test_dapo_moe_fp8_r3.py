"""DAPO + MoE + FP8 + R3 convergence shell (multi-GPU, slow)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer import rl_trainer as rl_trainer_mod
from lumenrl.trainer.rl_trainer import RLTrainer


pytestmark = [pytest.mark.multigpu, pytest.mark.moe, pytest.mark.fp8, pytest.mark.slow]


class _RisingRewardStub(rl_trainer_mod.StubRewardWorker):
    history: list[float] = []

    def compute_rewards(self, batch: DataProto) -> DataProto:
        b = batch.batch_size
        device = batch.tensors["old_log_probs"].device
        step = len(type(self).history) + 1
        batch.tensors["rewards"] = torch.linspace(0.15, 0.25, b, device=device, dtype=torch.float32) + 0.03 * float(
            step
        )
        batch.meta.setdefault(
            "response_lengths",
            [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)],
        )
        type(self).history.append(float(batch.tensors["rewards"].mean().item()))
        return batch


def test_dapo_convergence(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    path = e2e_config_dir / "dapo_moe_fp8_r3.yaml"
    assert path.is_file()
    cfg = LumenRLConfig.from_yaml(path)
    assert cfg.algorithm.name.lower() == "dapo"
    assert cfg.moe.r3.enabled is True

    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    _RisingRewardStub.history.clear()
    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _RisingRewardStub)

    run_cfg = LumenRLConfig.from_yaml(
        path,
        overrides=[
            "num_training_steps=20",
            "cluster.num_nodes=1",
            "cluster.gpus_per_node=2",
            f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'dapo_moe')}",
        ],
    )

    trainer = RLTrainer(run_cfg)
    try:
        trainer.setup()
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()

    h = _RisingRewardStub.history
    assert len(h) >= 8
    k = min(4, len(h) // 2)
    assert sum(h[-k:]) / k > sum(h[:k]) / k + 1e-6
