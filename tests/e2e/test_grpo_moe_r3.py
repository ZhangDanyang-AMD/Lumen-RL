"""MoE + R3 GRPO end-to-end structure (multi-GPU, slow)."""

from __future__ import annotations

import os
import statistics
from pathlib import Path

import pytest
import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer import rl_trainer as rl_trainer_mod
from lumenrl.trainer.callbacks import Callback, LoggingCallback
from lumenrl.trainer.rl_trainer import RLTrainer


pytestmark = [pytest.mark.multigpu, pytest.mark.moe, pytest.mark.slow]


class _StableRewardStub(rl_trainer_mod.StubRewardWorker):
    def compute_rewards(self, batch: DataProto) -> DataProto:
        b = batch.batch_size
        device = batch.tensors["old_log_probs"].device
        batch.tensors["rewards"] = torch.ones(b, device=device, dtype=torch.float32) * 0.5
        batch.meta.setdefault(
            "response_lengths",
            [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)],
        )
        return batch


def test_moe_r3_prevents_collapse(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    path = e2e_config_dir / "grpo_moe_r3.yaml"
    assert path.is_file()
    cfg = LumenRLConfig.from_yaml(path)
    assert cfg.moe.r3.enabled is True

    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _StableRewardStub)

    run_cfg = LumenRLConfig.from_yaml(
        path,
        overrides=[
            "num_training_steps=100",
            "cluster.num_nodes=1",
            "cluster.gpus_per_node=2",
            f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'moe_r3')}",
        ],
    )

    trainer = RLTrainer(run_cfg)
    mins: list[float] = []
    try:
        trainer.setup()

        class _MinReward(Callback):
            def on_step_end(self, trainer: RLTrainer, step: int, metrics: dict[str, float]) -> None:
                b = trainer._build_rollout_batch()
                b = trainer.rollout_wg.dispatch_and_call("rollout", b)  # type: ignore[union-attr]
                b = trainer.reward_wg.dispatch_and_call("compute_rewards", b)  # type: ignore[union-attr]
                mins.append(float(b.tensors["rewards"].min().item()))

        trainer.callbacks.append(_MinReward())
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()

    assert mins, "expected captured rewards"
    assert min(mins) > 0.4


def test_moe_without_r3_is_unstable(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    path = e2e_config_dir / "grpo_moe_r3.yaml"
    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    class _NoisyRewardStub(rl_trainer_mod.StubRewardWorker):
        def compute_rewards(self, batch: DataProto) -> DataProto:
            b = batch.batch_size
            device = batch.tensors["old_log_probs"].device
            batch.tensors["rewards"] = torch.randn(b, device=device, dtype=torch.float32) * 0.4
            batch.meta.setdefault(
                "response_lengths",
                [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)],
            )
            return batch

    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _NoisyRewardStub)

    run_cfg = LumenRLConfig.from_yaml(
        path,
        overrides=[
            "num_training_steps=24",
            "cluster.num_nodes=1",
            "cluster.gpus_per_node=2",
            "moe.r3.enabled=false",
            "algorithm.grpo.kl_coeff=0.05",
            f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'moe_nor3')}",
        ],
    )

    trainer = RLTrainer(run_cfg)
    kls: list[float] = []
    try:
        trainer.setup()

        class _KLTrace(LoggingCallback):
            def __init__(self) -> None:
                super().__init__(interval=1)

            def on_step_end(self, trainer: RLTrainer, step: int, metrics: dict[str, float]) -> None:
                super().on_step_end(trainer, step, metrics)
                kls.append(float(metrics.get("kl", 0.0)))

        trainer.callbacks.append(_KLTrace())
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()

    assert len(kls) >= 4
    assert statistics.pstdev(kls) > 1e-6
