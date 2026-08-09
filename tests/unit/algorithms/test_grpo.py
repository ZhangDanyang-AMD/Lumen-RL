from __future__ import annotations

import torch

from lumenrl.algorithms.grpo import GRPOAlgorithm
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto


def test_compute_advantages_group_relative() -> None:
    cfg = LumenRLConfig()
    cfg.algorithm.grpo.num_generations = 2
    algo = GRPOAlgorithm(cfg)
    rewards = torch.tensor([1.0, 3.0, 0.0, 4.0])
    batch = DataProto(tensors={"rewards": rewards})
    out = algo.compute_advantages(batch)
    adv = out.tensors["advantages"].view(-1, 2)
    assert torch.allclose(adv.sum(dim=1), torch.zeros(2), atol=1e-5)


def test_compute_loss_shape() -> None:
    cfg = LumenRLConfig()
    algo = GRPOAlgorithm(cfg)
    b, t = 2, 5
    batch = DataProto(
        tensors={
            "log_probs": torch.randn(b, t, requires_grad=True),
            "old_log_probs": torch.randn(b, t),
            "advantages": torch.randn(b),
        }
    )
    loss, metrics = algo.compute_loss(batch)
    assert loss.ndim == 0
    assert "loss_total" in metrics
    assert "loss_pg" in metrics


def test_compute_loss_uses_miles_asymmetric_sequence_mean() -> None:
    cfg = LumenRLConfig()
    cfg.algorithm.grpo.clip_ratio = 0.2
    cfg.algorithm.clip_ratio_high = 0.28
    cfg.algorithm.loss_agg_mode = "seq-mean-token-mean"
    algo = GRPOAlgorithm(cfg)
    ratios = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    log_probs = ratios.log().requires_grad_(True)
    batch = DataProto(
        tensors={
            "log_probs": log_probs,
            "old_log_probs": torch.zeros_like(log_probs),
            "advantages": torch.tensor([1.0, -1.0]),
            "response_mask": torch.tensor([[1, 0], [1, 1]]),
        },
        meta={"global_batch_size": 4, "dp_size": 2},
    )

    loss, metrics = algo.compute_loss(batch)

    torch.testing.assert_close(loss, torch.tensor((-1.28 + 0.8) / 2))
    assert metrics["loss_pg"] == float(loss.detach())
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))
