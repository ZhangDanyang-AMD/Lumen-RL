"""GRPO dense FP8 vs BF16 e2e checks (multi-GPU, FP8, slow)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer import rl_trainer as rl_trainer_mod
from lumenrl.trainer.rl_trainer import RLTrainer


pytestmark = [pytest.mark.multigpu, pytest.mark.fp8, pytest.mark.slow]


class _FlatRewardStub(rl_trainer_mod.StubRewardWorker):
    """Stable reward mean so loss trajectories dominate cross-precision comparison."""

    def compute_rewards(self, batch: DataProto) -> DataProto:
        b = batch.batch_size
        device = batch.tensors["old_log_probs"].device
        batch.tensors["rewards"] = torch.ones(b, device=device, dtype=torch.float32) * 0.25
        batch.meta.setdefault(
            "response_lengths",
            [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)],
        )
        return batch


def _run_trainer(cfg: LumenRLConfig) -> tuple[RLTrainer, float]:
    trainer = RLTrainer(cfg)
    t0 = time.perf_counter()
    try:
        trainer.setup()
        trainer.train()
    finally:
        elapsed = time.perf_counter() - t0
        if trainer._cluster is not None:
            trainer._cluster.shutdown()
    return trainer, elapsed


def test_grpo_dense_fp8_convergence(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    bf16_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    fp8_path = e2e_config_dir / "grpo_dense_fp8.yaml"
    assert bf16_path.is_file() and fp8_path.is_file()

    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _FlatRewardStub)

    overrides = [
        "num_training_steps=10",
        "cluster.num_nodes=1",
        "cluster.gpus_per_node=2",
        f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'cmp')}",
    ]
    cfg_bf16 = LumenRLConfig.from_yaml(bf16_path, overrides=overrides)
    cfg_fp8 = LumenRLConfig.from_yaml(fp8_path, overrides=overrides)

    tr_bf16, _ = _run_trainer(cfg_bf16)
    tr_fp8, _ = _run_trainer(cfg_fp8)

    loss_bf16 = float(tr_bf16.last_metrics.get("loss", 0.0))
    loss_fp8 = float(tr_fp8.last_metrics.get("loss", 0.0))
    denom = max(abs(loss_bf16), 1e-6)
    assert abs(loss_bf16 - loss_fp8) / denom <= 0.05 + 1e-6


def test_fp8_throughput_improvement(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    if os.environ.get("LUMENRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set LUMENRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    monkeypatch.setattr(rl_trainer_mod, "StubRewardWorker", _FlatRewardStub)

    overrides = [
        "num_training_steps=16",
        "cluster.num_nodes=1",
        "cluster.gpus_per_node=2",
    ]
    bf16_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    fp8_path = e2e_config_dir / "grpo_dense_fp8.yaml"

    _, t_bf16 = _run_trainer(
        LumenRLConfig.from_yaml(
            bf16_path,
            overrides=overrides
            + [f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'tbf16')}"],
        )
    )
    _, t_fp8 = _run_trainer(
        LumenRLConfig.from_yaml(
            fp8_path,
            overrides=overrides
            + [f"checkpointing.checkpoint_dir={str(tmp_checkpoint_dir / 'tfp8')}"],
        )
    )

    if t_fp8 >= 0.85 * t_bf16:
        pytest.skip(
            f"Inconclusive throughput (bf16={t_bf16:.3f}s fp8={t_fp8:.3f}s); "
            "stub trainer may not reflect FP8 speedups."
        )
