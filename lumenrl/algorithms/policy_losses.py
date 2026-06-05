# Copyright 2025 The LumenRL Authors.
# Derived from verl (verl-project/verl):
#   verl/trainer/ppo/core_algos.py L50-85 (registry, PolicyLossFn type)
#   verl/trainer/ppo/core_algos.py L1138-1199 (agg_loss)
#   verl/trainer/ppo/core_algos.py L1278-2487 (11 registered policy losses)
#
# Licensed under the Apache License, Version 2.0.
"""Policy loss registry with pluggable loss functions and aggregation modes."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from lumenrl.utils.torch_functional import masked_mean, masked_sum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias and registry  (verl/trainer/ppo/core_algos.py L50-85)
# ---------------------------------------------------------------------------

PolicyLossFn = Callable[
    [
        Tensor,                 # old_log_prob   [B, T]
        Tensor,                 # log_prob       [B, T]
        Tensor,                 # advantages     [B, T]
        Tensor,                 # response_mask  [B, T]
        str,                    # loss_agg_mode
        dict[str, Any],         # config dict (clip_ratio, clip_ratio_low, etc.)
        Optional[Tensor],       # rollout_is_weights (optional)
    ],
    tuple[Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Decorator that registers a policy loss function by name."""
    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func
    return decorator


def get_policy_loss_fn(name: str) -> PolicyLossFn:
    if name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {name!r}. "
            f"Supported: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[name]


# ---------------------------------------------------------------------------
# agg_loss  (verl/trainer/ppo/core_algos.py L1138-1199)
# ---------------------------------------------------------------------------

def agg_loss(
    loss_mat: Tensor,
    loss_mask: Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> Tensor:
    """Aggregate a per-token loss matrix into a scalar.

    Modes:
      - ``token-mean``: global token-level mean  (default PPO)
      - ``seq-mean-token-sum``: sum tokens per seq, then mean across seqs
      - ``seq-mean-token-sum-norm``: like above, divided by seq length
      - ``seq-mean-token-mean``: mean tokens per seq, then mean across seqs
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = int(loss_mask.sum().item())
        return masked_sum(loss_mat, loss_mask) / max(batch_num_tokens, 1) * dp_size

    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = int(seq_mask.sum().item())
        loss = masked_sum(seq_losses, seq_mask) / max(global_batch_size, 1) * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss = loss / loss_mask.shape[-1]
        return loss

    if loss_agg_mode == "seq-mean-token-mean":
        seq_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_count + 1e-8)
        seq_mask = (seq_count > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = int(seq_mask.sum().item())
        return masked_sum(seq_losses, seq_mask) / max(global_batch_size, 1) * dp_size

    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode!r}")


# ---------------------------------------------------------------------------
# Helper: extract clip config from dict
# ---------------------------------------------------------------------------

def _clip_cfg(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    cr = cfg.get("clip_ratio", 0.2)
    cr_low = cfg.get("clip_ratio_low", cr) or cr
    cr_high = cfg.get("clip_ratio_high", cr) or cr
    cr_c = cfg.get("clip_ratio_c", 3.0)
    return cr, cr_low, cr_high, cr_c


# ---------------------------------------------------------------------------
# Registered losses  (verl/trainer/ppo/core_algos.py L1278-2487)
# ---------------------------------------------------------------------------

@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Standard PPO clip with dual-clip C.
    (verl/trainer/ppo/core_algos.py L1278-1369)
    """
    assert config is not None
    _, cr_low, cr_high, cr_c = _clip_cfg(config)

    neg_kl = log_prob - old_log_prob
    neg_kl = torch.clamp(neg_kl, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = masked_mean(-neg_kl, response_mask)

    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - cr_low, 1 + cr_high)
    clip1 = torch.maximum(pg1, pg2)
    clipfrac = masked_mean(torch.gt(pg2, pg1).float(), response_mask)

    pg3 = -advantages * cr_c
    clip2 = torch.minimum(pg3, clip1)
    clipfrac_lower = masked_mean(
        torch.gt(clip1, pg3).float() * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip2, clip1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": clipfrac_lower.detach().item(),
    }


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """DPPO-Binary-TV (arxiv 2602.04879).
    (verl/trainer/ppo/core_algos.py L1372-1450)
    """
    assert config is not None
    cr, cr_low, cr_high, _ = _clip_cfg(config)
    clip_c = config.get("clip_ratio_c", 20.0)

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = masked_mean(-neg_kl, response_mask)

    trunc_ratio = torch.clamp(ratio, max=clip_c).detach()

    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_pos = (prob - old_prob) <= cr_high
    valid_neg = (prob - old_prob) >= -cr_low
    valid = torch.where(advantages > 0, valid_pos, valid_neg).detach().float()

    pg_losses = -advantages * trunc_ratio * log_prob * valid
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))

    clipfrac = masked_mean((1.0 - valid).float(), response_mask)
    clipfrac_lower = masked_mean((ratio > clip_c).float() * valid, response_mask)
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": clipfrac_lower.detach().item(),
    }


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """DPPO-Binary-KL (arxiv 2602.04879).
    (verl/trainer/ppo/core_algos.py L1453-1535)
    """
    assert config is not None
    cr, cr_low, cr_high, _ = _clip_cfg(config)
    clip_c = config.get("clip_ratio_c", 20.0)

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = masked_mean(-neg_kl, response_mask)
    trunc_ratio = torch.clamp(ratio, max=clip_c).detach()

    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_pos = (binary_kl <= cr_high) | (prob <= old_prob)
    valid_neg = (binary_kl <= cr_low) | (prob >= old_prob)
    valid = torch.where(advantages > 0, valid_pos, valid_neg).detach().float()

    pg_losses = -advantages * trunc_ratio * log_prob * valid
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))

    clipfrac = masked_mean((1.0 - valid).float(), response_mask)
    clipfrac_lower = masked_mean((ratio > clip_c).float() * valid, response_mask)
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": clipfrac_lower.detach().item(),
    }


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "seq-mean-token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """GSPO sequence-level importance ratio (arxiv 2507.18071).
    (verl/trainer/ppo/core_algos.py L1538-1611)
    """
    assert config is not None
    _, cr_low, cr_high, _ = _clip_cfg(config)

    neg_kl = log_prob - old_log_prob
    seq_len = torch.sum(response_mask, dim=-1).clamp(min=1)
    neg_kl_seq = torch.sum(neg_kl * response_mask, dim=-1) / seq_len

    log_sir = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    log_sir = torch.clamp(log_sir, max=10.0)
    sir = torch.exp(log_sir)

    pg1 = -advantages * sir
    pg2 = -advantages * torch.clamp(sir, 1 - cr_low, 1 + cr_high)
    pg_losses = torch.maximum(pg1, pg2)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, "seq-mean-token-mean",
                    dp_size=config.get("dp_size", 1),
                    global_batch_size=config.get("global_batch_size"))

    clipfrac = masked_mean(torch.gt(pg2, pg1).float(), response_mask)
    ppo_kl = masked_mean(-neg_kl, response_mask)
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "seq-mean-token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """SAPO smoothed gating (arxiv 2511.20347).
    (verl/trainer/ppo/core_algos.py L1614-1696)
    """
    assert config is not None
    tau_pos = config.get("tau_pos", 10.0)
    tau_neg = config.get("tau_neg", 10.0)

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)

    tau_pos_t = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg_t = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)
    taus = torch.where(advantages > 0, tau_pos_t, tau_neg_t)
    gates = torch.sigmoid(taus * (ratio - 1.0)) * (4.0 / taus)

    pg_losses = -gates * advantages
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, "seq-mean-token-mean",
                    dp_size=config.get("dp_size", 1),
                    global_batch_size=config.get("global_batch_size"))

    ppo_kl = masked_mean(-neg_kl, response_mask)
    return loss, {
        "actor/pg_clipfrac": 0.0,
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """GPG pure policy gradient.
    (verl/trainer/ppo/core_algos.py L1699-1732)
    """
    assert config is not None
    pg_losses = -log_prob * advantages
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))
    return loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Clip-Cov covariance clipping.
    (verl/trainer/ppo/core_algos.py L1735-1837)
    """
    assert config is not None
    _, cr_low, cr_high, _ = _clip_cfg(config)
    clip_cov_ratio = config.get("clip_cov_ratio", 0.0002)
    clip_cov_ub = config.get("clip_cov_ub", 5.0)
    clip_cov_lb = config.get("clip_cov_lb", 1.0)

    neg_kl = log_prob - old_log_prob
    ratio = torch.exp(neg_kl)
    ppo_kl = masked_mean(-neg_kl, response_mask)

    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - cr_low, 1 + cr_high)
    clip_by_origin = (pg2 > pg1) & (response_mask > 0)

    corr = torch.ones_like(advantages)
    cov_all = (advantages - masked_mean(advantages, response_mask)) * (
        log_prob - masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)
    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx), device=top_k_idx.device)
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)
    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_losses = torch.maximum(pg1, pg2) * corr
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    clipfrac = masked_mean((corr == 0).float(), response_mask)
    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """KL-Cov covariance + KL penalty on high-cov tokens.
    (verl/trainer/ppo/core_algos.py L1840-1917)
    """
    assert config is not None
    kl_cov_ratio = config.get("kl_cov_ratio", 0.0002)
    ppo_kl_coef = config.get("ppo_kl_coef", 1.0)

    neg_kl = log_prob - old_log_prob
    abs_kl = neg_kl.abs()
    ratio = torch.exp(neg_kl)
    ppo_kl_abs = masked_mean(abs_kl, response_mask)

    pg1 = -advantages * ratio
    pg_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg1.clone()

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))
    if k != 0:
        cov = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_num = max(1, int(len(cov) * kl_cov_ratio))
        large_idx = torch.topk(cov, k_num, largest=True).indices
        if len(large_idx) > 0:
            large_idx = all_valid_idx[large_idx]
            cols = advantages.shape[1]
            pg_losses[large_idx // cols, large_idx % cols] = pg_kl[large_idx // cols, large_idx % cols]

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))
    return loss, {"actor/ppo_kl": ppo_kl_abs.detach().item()}


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """GMPO geometric mean ratio (arxiv 2507.20673).
    (verl/trainer/ppo/core_algos.py L1920-2003)
    """
    assert config is not None
    _, cr_low, cr_high, _ = _clip_cfg(config)

    neg_kl = log_prob - old_log_prob
    ppo_kl = masked_mean(-neg_kl, response_mask)

    sgn = torch.sign(advantages)
    neg_kl_clamp = torch.clamp(neg_kl, -cr_low, cr_high)
    neg_kl_min = sgn * torch.min(sgn * neg_kl, sgn * neg_kl_clamp)

    mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((neg_kl_min * response_mask).sum(dim=-1) / (mask_sum + 1e-8))
    adv = (advantages * response_mask).sum(dim=-1) / (mask_sum + 1e-8)
    pg_losses = -adv * ratio

    if rollout_is_weights is not None:
        seq_is = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is

    loss = torch.mean(pg_losses)

    clipped = torch.ne(neg_kl, neg_kl_clamp)
    clipfrac = masked_mean((clipped * (advantages > 0)).float(), response_mask)
    clipfrac_lower = masked_mean((clipped * (advantages < 0)).float(), response_mask)
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": clipfrac_lower.detach().item(),
    }


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """CISPO stop-gradient clipped IS (arxiv 2506.13585).
    (verl/trainer/ppo/core_algos.py L2006-2064)
    """
    assert config is not None
    _, cr_low, cr_high, _ = _clip_cfg(config)

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = masked_mean(-neg_kl, response_mask)

    clipped_ratio = torch.clamp(ratio, 1 - cr_low, 1 + cr_high).detach()
    pg_losses = -clipped_ratio * advantages * log_prob

    clipfrac = masked_mean((ratio != clipped_ratio).float(), response_mask)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                    dp_size=config.get("dp_size", 1),
                    batch_num_tokens=config.get("batch_num_tokens"),
                    global_batch_size=config.get("global_batch_size"))
    return loss, {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: Tensor, log_prob: Tensor, advantages: Tensor,
    response_mask: Tensor, loss_agg_mode: str = "token-mean",
    config: dict[str, Any] | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Bypass mode: old_log_prob = rollout_log_prob, dispatches to REINFORCE or PPO-clip.
    (verl/trainer/ppo/core_algos.py L2351-2487)
    """
    assert config is not None
    loss_type = config.get("bypass_loss_type", "ppo_clip")

    if loss_type == "reinforce":
        pg_losses = -advantages * log_prob
        if rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
        loss = agg_loss(pg_losses, response_mask, loss_agg_mode,
                        dp_size=config.get("dp_size", 1),
                        batch_num_tokens=config.get("batch_num_tokens"),
                        global_batch_size=config.get("global_batch_size"))
        neg_kl = log_prob - old_log_prob
        kl = masked_mean(-neg_kl, response_mask)
        return loss, {"actor/ppo_kl": kl.detach().item()}

    # Default: PPO-clip with old_log_prob = rollout_log_prob
    return compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask,
        loss_agg_mode, config, rollout_is_weights=None,
    )
