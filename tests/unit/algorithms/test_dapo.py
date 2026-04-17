from __future__ import annotations

import torch

from lumenrl.algorithms.dapo import DAPOAlgorithm
from lumenrl.algorithms.loss_functions import asymmetric_clip_loss
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto


def test_compute_advantages_with_dynamic_sampling() -> None:
    cfg = LumenRLConfig()
    cfg.algorithm.dapo.num_generations = 2
    cfg.algorithm.dapo.dynamic_sampling = True
    cfg.algorithm.dapo.overlong_reward_shaping = False
    algo = DAPOAlgorithm(cfg)
    rewards = torch.tensor([1.0, 1.0, 1.0, 2.0])
    batch = DataProto(tensors={"rewards": rewards})
    out = algo.compute_advantages(batch)
    mask = out.tensors["dapo_sample_mask"]
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(mask, expected)


def test_asymmetric_clip() -> None:
    cfg = LumenRLConfig()
    algo = DAPOAlgorithm(cfg)
    dapo = cfg.algorithm.dapo
    b, t = 2, 3
    logp = torch.randn(b, t, requires_grad=True)
    old_logp = torch.zeros(b, t)
    adv = torch.ones(b, t)
    batch = DataProto(
        tensors={
            "log_probs": logp,
            "old_log_probs": old_logp,
            "advantages": adv,
            "dapo_sample_mask": torch.ones(b, t),
        }
    )
    sm = torch.ones_like(logp)
    expected = asymmetric_clip_loss(
        logp,
        old_logp,
        adv,
        float(dapo.clip_ratio_low),
        float(dapo.clip_ratio_high),
        mask=sm,
    )
    loss, _ = algo.compute_loss(batch)
    assert torch.allclose(loss, expected)
