from __future__ import annotations

import torch

from lumenrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    entropy_bonus,
    kl_penalty,
    policy_gradient_loss,
)


def test_policy_gradient_loss_gradient_flows() -> None:
    b, t = 2, 4
    logp = torch.randn(b, t, requires_grad=True)
    old_logp = torch.randn(b, t).detach()
    adv = torch.randn(b, t)
    loss = policy_gradient_loss(logp, old_logp, adv, clip_ratio=0.2)
    loss.backward()
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all()


def test_asymmetric_clip_loss() -> None:
    logp = torch.tensor([[0.5, 0.5]])
    old_logp = torch.zeros(1, 2)
    adv = torch.ones(1, 2)
    loss = asymmetric_clip_loss(logp, old_logp, adv, clip_low=0.2, clip_high=0.28)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kl_penalty_nonneg() -> None:
    logp = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[0.5, 0.5]])
    kl = kl_penalty(logp, ref)
    assert float(kl) >= 0.0


def test_entropy_bonus() -> None:
    logp = torch.tensor([[-1.0, -0.5]])
    ent = entropy_bonus(logp)
    assert ent.ndim == 0
    assert float(ent) >= 0.0
