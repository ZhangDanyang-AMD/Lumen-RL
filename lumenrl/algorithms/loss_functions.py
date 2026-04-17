"""Shared PyTorch loss helpers for policy optimization algorithms."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def policy_gradient_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_ratio: float,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Clipped surrogate policy-gradient (PPO-style) objective as a loss to minimize.

    Computes ``-mean(min(r_t * A, clip(r_t) * A))`` where ``r_t = exp(logp - logp_old)``.

    Args:
        logprobs: Current log-probabilities, shape broadcastable with ``old_logprobs``.
        old_logprobs: Behavior-policy log-probabilities (same shape as ``logprobs``).
        advantages: Advantage estimates (same shape as ``logprobs``).
        clip_ratio: Symmetric clipping radius ``epsilon``; ratio is clamped to
            ``[1 - clip_ratio, 1 + clip_ratio]``.
        mask: Optional boolean/float mask of the same shape as ``logprobs`` for
            token-level masking (1 = include).

    Returns:
        Scalar tensor: clipped surrogate loss (negative of the PPO objective).
    """
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    pg = -torch.minimum(surr1, surr2)
    if mask is not None:
        w = mask.to(dtype=pg.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (pg * w).sum() / denom
    return pg.mean()


def asymmetric_clip_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Policy-gradient surrogate with asymmetric ratio clipping (DAPO-style).

    The importance ratio ``exp(logp - logp_old)`` is clamped to ``[clip_low, clip_high]``
    instead of a symmetric epsilon band.

    Args:
        logprobs: Current log-probabilities.
        old_logprobs: Reference log-probabilities.
        advantages: Advantage tensor (broadcastable).
        clip_low: Lower bound for the clipped ratio (e.g. ``0.8``).
        clip_high: Upper bound for the clipped ratio (e.g. ``1.2``).
        mask: Optional token mask for masked mean reduction.

    Returns:
        Scalar loss tensor to minimize.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    clipped = torch.clamp(ratio, clip_low, clip_high)
    surr1 = ratio * advantages
    surr2 = clipped * advantages
    pg = -torch.minimum(surr1, surr2)
    if mask is not None:
        w = mask.to(dtype=pg.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (pg * w).sum() / denom
    return pg.mean()


def value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_ratio: float,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Clipped value-function regression loss (PPO-style).

    Args:
        values: Current value predictions ``V(s)``.
        old_values: Old value predictions ``V_old(s)`` from behavior critic.
        returns: TD(lambda) or Monte-Carlo return targets.
        clip_ratio: Epsilon for clipping ``values`` around ``old_values``.
        mask: Optional mask for token-level values.

    Returns:
        Scalar MSE-style clipped value loss.
    """
    values_clipped = old_values + torch.clamp(
        values - old_values, -clip_ratio, clip_ratio
    )
    loss_unclipped = F.mse_loss(values, returns, reduction="none")
    loss_clipped = F.mse_loss(values_clipped, returns, reduction="none")
    vf = torch.maximum(loss_unclipped, loss_clipped)
    if mask is not None:
        w = mask.to(dtype=vf.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (vf * w).sum() / denom
    return vf.mean()


def kl_penalty(logprobs: Tensor, ref_logprobs: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
    """Token-level KL penalty ``mean(ref_logp - logp)`` (non-negative when ref is teacher).

    This matches the common RLHF surrogate ``KL(pi || pi_ref)`` Monte Carlo estimate
    along the sampled trajectory.

    Args:
        logprobs: Log-probs under the trainable policy.
        ref_logprobs: Log-probs under the reference policy.
        mask: Optional mask for valid tokens.

    Returns:
        Scalar KL penalty tensor.
    """
    kl = ref_logprobs - logprobs
    if mask is not None:
        w = mask.to(dtype=kl.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (kl * w).sum() / denom
    return kl.mean()


def entropy_bonus(logprobs: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
    """Surrogate entropy term from sampled-action log-probabilities.

    For a sampled action ``a`` with ``logp = log pi(a|s)``, the quantity
    ``-exp(logp) * logp`` equals ``-p(a) log p(a)`` for that draw and is a
    nonnegative scalar that can be used as a per-token entropy surrogate when
    full logits are unavailable.

    Args:
        logprobs: Log-probabilities of actions that were sampled from the policy.
        mask: Optional mask for valid tokens.

    Returns:
        Scalar tensor; **subtract** ``coeff * entropy_bonus(...)`` from the loss
        to encourage larger per-token entropy mass (typical PPO sign convention).
    """
    ent = -(torch.exp(logprobs) * logprobs)
    if mask is not None:
        w = mask.to(dtype=ent.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (ent * w).sum() / denom
    return ent.mean()
