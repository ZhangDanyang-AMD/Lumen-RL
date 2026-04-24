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
    clip_ratio_c: float = 0.0,
) -> Tensor:
    """Policy-gradient surrogate with asymmetric ratio clipping (DAPO-style).

    ``clip_low`` and ``clip_high`` are **epsilon** values: the importance ratio
    ``exp(logp - logp_old)`` is clamped to ``[1 - clip_low, 1 + clip_high]``.

    When ``clip_ratio_c > 0``, applies DAPO dual-clip: for negative advantages,
    the loss is capped at ``-advantages * clip_ratio_c``, preventing domination
    by highly negative advantages and preserving gradient signal for positive
    advantage tokens. Matches verl's implementation.

    Args:
        logprobs: Current log-probabilities.
        old_logprobs: Reference log-probabilities.
        advantages: Advantage tensor (broadcastable).
        clip_low: Lower epsilon; ratio lower bound is ``1 - clip_low`` (e.g. 0.2 → 0.8).
        clip_high: Upper epsilon; ratio upper bound is ``1 + clip_high`` (e.g. 0.28 → 1.28).
        mask: Optional token mask for masked mean reduction.
        clip_ratio_c: Dual-clip bound C (DAPO). 0 disables. Typical value: 3.0.

    Returns:
        Scalar loss tensor to minimize.
    """
    neg_approx_kl = logprobs - old_logprobs
    neg_approx_kl = torch.clamp(neg_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(neg_approx_kl)
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped
    pg = torch.maximum(pg_losses1, pg_losses2)

    if clip_ratio_c > 0.0:
        pg_losses3 = -advantages * clip_ratio_c
        pg_clipped = torch.minimum(pg_losses3, pg)
        pg = torch.where(advantages < 0, pg_clipped, pg)

    if mask is not None:
        w = mask.to(dtype=pg.dtype)
        pg = torch.where(w.bool(), pg, torch.zeros_like(pg))
        denom = torch.clamp(w.sum(), min=1.0)
        return pg.sum() / denom
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


def opd_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    mask: Optional[Tensor] = None,
    kl_direction: str = "reverse",
    temperature: float = 1.0,
    position_weights: Optional[Tensor] = None,
) -> Tensor:
    """Full-vocabulary KL divergence for On-Policy Distillation.

    Unlike :func:`kl_penalty` which operates on sampled-token log-probs, this
    computes the exact KL over the entire vocabulary at each token position.

    Args:
        student_logits: Raw logits from the student model, ``[B, T, V]``.
        teacher_logits: Raw logits from the teacher model, ``[B, T, V]`` (detached).
        mask: Optional ``[B, T]`` mask (1 = include token in loss).
        kl_direction: ``"reverse"`` for ``D_KL(student || teacher)`` (DeepSeek-V4),
            ``"forward"`` for ``D_KL(teacher || student)`` (TorchSpec Eagle3).
        temperature: Softmax temperature applied to both distributions.
        position_weights: Optional ``[T]`` weights (e.g. ``0.8 ** i``) for
            position-dependent weighting.

    Returns:
        Scalar KL divergence loss tensor.
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)

    if kl_direction == "reverse":
        # D_KL(student || teacher) = sum_v p_s(v) * [log p_s(v) - log p_t(v)]
        student_probs = student_log_probs.exp()
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    elif kl_direction == "forward":
        # D_KL(teacher || student) = sum_v p_t(v) * [log p_t(v) - log p_s(v)]
        teacher_probs = teacher_log_probs.exp()
        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    else:
        raise ValueError(f"Unknown kl_direction: {kl_direction!r}. Use 'reverse' or 'forward'.")

    if position_weights is not None:
        T_len = kl_per_token.shape[1]
        pw = position_weights[:T_len].to(dtype=kl_per_token.dtype, device=kl_per_token.device)
        kl_per_token = kl_per_token * pw.unsqueeze(0)

    if mask is not None:
        w = mask.to(dtype=kl_per_token.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (kl_per_token * w).sum() / denom
    return kl_per_token.mean()


def hidden_state_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    *,
    mask: Optional[Tensor] = None,
    loss_type: str = "mse",
    projection: Optional[torch.nn.Module] = None,
) -> Tensor:
    """Hidden-state matching loss for speculative decoding draft model training.

    Args:
        student_hidden: Student hidden states, ``[B, T, D_s]``.
        teacher_hidden: Teacher hidden states, ``[B, T, D_t]``.
        mask: Optional ``[B, T]`` mask.
        loss_type: ``"mse"`` for mean squared error, ``"cosine"`` for cosine distance.
        projection: Optional linear layer to project ``D_s`` to ``D_t`` if they differ.

    Returns:
        Scalar loss tensor.
    """
    if projection is not None:
        student_hidden = projection(student_hidden)

    if loss_type == "mse":
        per_token = F.mse_loss(student_hidden.float(), teacher_hidden.float(), reduction="none").mean(dim=-1)
    elif loss_type == "cosine":
        cos_sim = F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)
        per_token = 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Use 'mse' or 'cosine'.")

    if mask is not None:
        w = mask.to(dtype=per_token.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (per_token * w).sum() / denom
    return per_token.mean()


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
