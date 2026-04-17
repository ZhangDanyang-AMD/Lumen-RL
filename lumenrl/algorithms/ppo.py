"""Proximal Policy Optimization (PPO) with GAE and value loss."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.loss_functions import kl_penalty, policy_gradient_loss, value_loss
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.core.types import AlgorithmName

logger = logging.getLogger(__name__)


def _last_token_indices(attention_mask: Tensor) -> Tensor:
    """Return index of the last valid token per row (0-based)."""
    lengths = attention_mask.long().sum(dim=-1)
    return torch.clamp(lengths - 1, min=0)


def _scalar_rewards_to_token_rewards(
    rewards: Tensor, attention_mask: Tensor
) -> Tensor:
    """Scatter final scalar rewards onto the last valid timestep."""
    b, t = attention_mask.shape
    device = attention_mask.device
    dtype = rewards.dtype
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    out = torch.zeros(b, t, device=device, dtype=dtype)
    rows = torch.arange(b, device=device)
    last = _last_token_indices(attention_mask)
    out[rows, last] = rewards.to(dtype=dtype)
    return out


def _gae_returns(
    token_rewards: Tensor,
    values: Tensor,
    attention_mask: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Vectorized GAE-Lambda advantages and TD(lambda) returns."""
    # mask: 1 for valid post-prompt tokens (use attention_mask as proxy)
    m = attention_mask.to(dtype=values.dtype)
    # Approximate "next value" by rolling values along time
    next_values = torch.roll(values, shifts=-1, dims=-1)
    # Zero bootstrap after last valid token
    lengths = attention_mask.long().sum(dim=-1)
    batch_idx = torch.arange(values.shape[0], device=values.device)
    next_values[batch_idx, lengths - 1] = 0.0

    delta = token_rewards + gamma * next_values * m - values
    b, seq = delta.shape
    adv = torch.zeros_like(delta)
    last_gae = torch.zeros(b, device=values.device, dtype=values.dtype)
    for t in range(seq - 1, -1, -1):
        last_gae = delta[:, t] + gamma * lam * m[:, t] * last_gae
        adv[:, t] = last_gae
    returns = adv + values
    return adv * m, returns * m


class PPOAlgorithm(BaseAlgorithm):
    """Standard PPO with clipped surrogate, optional KL to reference, and value loss."""

    def compute_advantages(self, batch: DataProto) -> DataProto:
        if "values" not in batch.tensors:
            raise KeyError("PPO requires critic 'values' tensor [B, T].")
        if "attention_mask" not in batch.tensors:
            raise KeyError("PPO requires 'attention_mask' for GAE masking.")
        if "rewards" not in batch.tensors:
            raise KeyError("PPO requires 'rewards' (scalar or token-level).")

        values = batch.tensors["values"]
        mask = batch.tensors["attention_mask"].to(dtype=torch.float32)
        rewards = batch.tensors["rewards"]

        if rewards.dim() == 1 or (rewards.dim() == 2 and rewards.shape[-1] == 1):
            token_rewards = _scalar_rewards_to_token_rewards(
                rewards.squeeze(-1) if rewards.dim() == 2 else rewards, mask.bool()
            )
        else:
            token_rewards = rewards.to(dtype=values.dtype)
            if token_rewards.shape != values.shape:
                raise ValueError(
                    f"Token rewards shape {tuple(token_rewards.shape)} != values {tuple(values.shape)}."
                )

        cfg = self._config.algorithm.ppo
        adv, ret = _gae_returns(
            token_rewards,
            values,
            mask,
            gamma=float(cfg.discount),
            lam=float(cfg.gae_lambda),
        )
        batch.tensors["advantages"] = adv
        batch.tensors["returns"] = ret
        mb = mask.bool()
        adv_mean = float((adv * mask).sum() / mask.sum().clamp(min=1.0))
        adv_std = float(torch.std(adv[mb])) if int(mask.sum()) > 0 else 0.0
        logger.debug("PPO GAE: adv_mean=%.6f adv_std=%.6f", adv_mean, adv_std)
        return batch

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        required = ("log_probs", "old_log_probs", "advantages", "values", "returns")
        for k in required:
            if k not in batch.tensors:
                raise KeyError(f"PPO loss missing tensor '{k}'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        values = batch.tensors["values"]
        rets = batch.tensors["returns"]
        mask = batch.tensors.get("attention_mask")
        if mask is None:
            mask_t = torch.ones_like(adv)
        else:
            mask_t = mask.to(dtype=adv.dtype)

        clip = self._config.algorithm.ppo.clip_ratio
        pg = policy_gradient_loss(logp, old_logp, adv, clip, mask=mask_t)

        old_values = batch.tensors.get("old_values", values.detach())
        vf = value_loss(values, old_values, rets, clip, mask=mask_t)

        loss = pg + 0.5 * vf
        metrics: dict[str, Any] = {
            "loss_pg": float(pg.detach().cpu()),
            "loss_vf": float(vf.detach().cpu()),
        }

        kl_c = self._config.algorithm.ppo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask_t.bool())
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        # Optional value clip diagnostics
        with torch.no_grad():
            metrics["explained_variance"] = float(
                1.0
                - F.mse_loss(values[mask_t.bool()], rets[mask_t.bool()])
                / torch.var(rets[mask_t.bool()].float()).clamp_min(1e-8)
                if bool(mask_t.sum() > 0)
                else 0.0
            )

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.PPO.value, PPOAlgorithm)
