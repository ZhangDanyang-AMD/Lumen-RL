# Copyright 2025 The LumenRL Authors (originally derived from verl).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pluggable advantage estimator registry.

Each estimator has signature ``(batch: DataProto, config: LumenRLConfig) -> DataProto``
and populates ``batch.tensors["advantages"]`` (and optionally ``batch.tensors["returns"]``).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADV_ESTIMATOR_REGISTRY: dict[str, Callable[[DataProto, LumenRLConfig], DataProto]] = {}


def register_adv_est(name: str) -> Callable:
    """Decorator that registers an advantage estimator function by *name*."""

    def wrapper(fn: Callable[[DataProto, LumenRLConfig], DataProto]) -> Callable:
        if name in ADV_ESTIMATOR_REGISTRY:
            logger.warning("Overwriting advantage estimator '%s'.", name)
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masked_whiten(values: Tensor, mask: Tensor, shift_mean: bool = True) -> Tensor:
    """Whiten *values* using only the entries where *mask* is nonzero."""
    valid = values[mask.bool()]
    if valid.numel() < 2:
        return values
    mean = valid.mean() if shift_mean else 0.0
    std = valid.std().clamp(min=1e-8)
    return (values - mean) / std * mask.float()


def _response_mask_from_batch(batch: DataProto) -> Tensor | None:
    """Return ``response_mask`` if available, else fall back to ``attention_mask``."""
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


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
    m = attention_mask.to(dtype=values.dtype)
    next_values = torch.roll(values, shifts=-1, dims=-1)
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


def _apply_overlong_shaping(
    rewards: Tensor,
    batch: DataProto,
    max_len: int,
    penalty: float,
) -> Tensor:
    """Subtract a linear penalty when sequence length exceeds *max_len*."""
    if penalty <= 0.0:
        return rewards
    lengths = batch.meta.get("response_lengths")
    if lengths is None:
        return rewards
    lens_t = torch.as_tensor(lengths, device=rewards.device, dtype=torch.float32)
    if lens_t.shape[0] != rewards.shape[0]:
        return rewards
    over = torch.clamp(lens_t - float(max_len), min=0.0)
    shaped = rewards.to(dtype=torch.float32) - penalty * over
    return shaped.to(dtype=rewards.dtype)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

@register_adv_est("grpo")
def compute_grpo_advantage(batch: DataProto, config: LumenRLConfig) -> DataProto:
    """Group-based mean/std normalization (GRPO)."""
    if "rewards" not in batch.tensors:
        raise KeyError("GRPO requires tensor key 'rewards' on the batch.")
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    cfg = config.algorithm.grpo
    g = cfg.num_generations
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )
    grouped = rewards.view(-1, g)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
    adv = (grouped - mean) / std
    adv_flat = adv.reshape(-1)
    batch.tensors["advantages"] = adv_flat
    logger.debug(
        "GRPO advantages: mean=%.6f std=%.6f",
        adv_flat.mean().item(),
        adv_flat.std().item(),
    )
    return batch


@register_adv_est("dapo")
def compute_dapo_advantage(batch: DataProto, config: LumenRLConfig) -> DataProto:
    """DAPO: group-relative advantages with dynamic sampling and overlong shaping."""
    if "rewards" not in batch.tensors:
        raise KeyError("DAPO requires tensor key 'rewards' on the batch.")
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    cfg = config.algorithm.dapo
    g = cfg.num_generations
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )

    if cfg.overlong_reward_shaping:
        rewards = _apply_overlong_shaping(
            rewards,
            batch,
            max_len=int(config.policy.max_total_sequence_length),
            penalty=float(batch.meta.get("overlong_penalty", 1.0)),
        )

    grouped = rewards.view(-1, g)
    std = grouped.std(dim=1)

    if cfg.dynamic_sampling:
        keep = std > 1e-6
        if not torch.any(keep):
            logger.warning("DAPO dynamic sampling removed all groups; keeping all.")
            keep = torch.ones_like(keep, dtype=torch.bool)
        row_mask = keep.unsqueeze(-1).expand(-1, g).reshape(-1)
    else:
        row_mask = torch.ones(
            rewards.shape[0], dtype=torch.bool, device=rewards.device
        )

    mean = grouped.mean(dim=1, keepdim=True)
    std_safe = grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
    adv = (grouped - mean) / std_safe
    adv_flat = adv.reshape(-1)

    batch.tensors["advantages"] = adv_flat
    batch.tensors["dapo_sample_mask"] = row_mask.to(dtype=torch.float32)
    if adv_flat.numel() > 0:
        logger.info(
            "NaN-DEBUG DAPO advantages: active_frac=%.4f, adv nan=%d inf=%d "
            "min=%.4f max=%.4f mean=%.4f, rewards min=%.4f max=%.4f mean=%.4f, "
            "std min=%.6f max=%.6f",
            float(row_mask.float().mean().cpu()),
            adv_flat.isnan().sum().item(),
            adv_flat.isinf().sum().item(),
            adv_flat.min().item(),
            adv_flat.max().item(),
            adv_flat.mean().item(),
            rewards.min().item(),
            rewards.max().item(),
            rewards.mean().item(),
            std.min().item(),
            std.max().item(),
        )
    else:
        logger.warning(
            "DAPO advantages: adv_flat is empty (batch_size=%d, num_gen=%d)",
            rewards.shape[0],
            g,
        )
    return batch


@register_adv_est("gae")
def compute_gae_advantage(batch: DataProto, config: LumenRLConfig) -> DataProto:
    """Generalized Advantage Estimation (GAE-Lambda) for PPO."""
    if "values" not in batch.tensors:
        raise KeyError("GAE requires critic 'values' tensor [B, T].")
    if "rewards" not in batch.tensors:
        raise KeyError("GAE requires 'rewards' (scalar or token-level).")

    values = batch.tensors["values"]
    # Prefer response_mask, fall back to attention_mask
    mask = _response_mask_from_batch(batch)
    if mask is None:
        raise KeyError("GAE requires 'response_mask' or 'attention_mask' for masking.")
    mask_f = mask.to(dtype=torch.float32)
    rewards = batch.tensors["rewards"]

    if rewards.dim() == 1 or (rewards.dim() == 2 and rewards.shape[-1] == 1):
        token_rewards = _scalar_rewards_to_token_rewards(
            rewards.squeeze(-1) if rewards.dim() == 2 else rewards, mask
        )
    else:
        token_rewards = rewards.to(dtype=values.dtype)
        if token_rewards.shape != values.shape:
            raise ValueError(
                f"Token rewards shape {tuple(token_rewards.shape)} != "
                f"values {tuple(values.shape)}."
            )

    cfg = config.algorithm.ppo
    adv, ret = _gae_returns(
        token_rewards,
        values,
        mask_f,
        gamma=float(cfg.discount),
        lam=float(cfg.gae_lambda),
    )
    batch.tensors["advantages"] = adv
    batch.tensors["returns"] = ret
    mb = mask.bool()
    adv_mean = float((adv * mask_f).sum() / mask_f.sum().clamp(min=1.0))
    adv_std = float(torch.std(adv[mb])) if int(mask_f.sum()) > 0 else 0.0
    logger.debug("GAE: adv_mean=%.6f adv_std=%.6f", adv_mean, adv_std)
    return batch


@register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_advantage(
    batch: DataProto, config: LumenRLConfig
) -> DataProto:
    """REINFORCE++ with discounted cumulative returns and masked whitening."""
    if "rewards" not in batch.tensors:
        raise KeyError("REINFORCE++ requires tensor key 'rewards' on the batch.")

    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    response_mask = _response_mask_from_batch(batch)
    if response_mask is None:
        raise KeyError(
            "REINFORCE++ requires 'response_mask' or 'attention_mask'."
        )
    response_mask_f = response_mask.to(dtype=torch.float32)

    gamma = float(config.algorithm.ppo.discount)

    # Scatter scalar rewards to last response token
    token_rewards = _scalar_rewards_to_token_rewards(rewards, response_mask)

    # Compute discounted returns by reverse iteration over response tokens
    b, t = token_rewards.shape
    returns = torch.zeros_like(token_rewards)
    running = torch.zeros(b, device=token_rewards.device, dtype=token_rewards.dtype)
    for step in range(t - 1, -1, -1):
        running = token_rewards[:, step] + gamma * running * response_mask_f[:, step]
        returns[:, step] = running

    # Apply masked whitening (zero-mean, unit-var within response_mask)
    advantages = _masked_whiten(returns, response_mask_f, shift_mean=True)

    batch.tensors["advantages"] = advantages
    logger.debug(
        "REINFORCE++ advantages: mean=%.6f std=%.6f",
        advantages[response_mask].mean().item() if response_mask.any() else 0.0,
        advantages[response_mask].std().item() if response_mask.any() else 0.0,
    )
    return batch


@register_adv_est("rloo")
def compute_rloo_advantage(batch: DataProto, config: LumenRLConfig) -> DataProto:
    """Leave-one-out baseline (RLOO) advantage estimation."""
    if "rewards" not in batch.tensors:
        raise KeyError("RLOO requires tensor key 'rewards' on the batch.")

    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    # Determine num_generations from meta or config
    g = batch.meta.get("num_generations", config.algorithm.grpo.num_generations)
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )

    response_mask = _response_mask_from_batch(batch)
    if response_mask is None:
        raise KeyError("RLOO requires 'response_mask' or 'attention_mask'.")
    response_mask_f = response_mask.to(dtype=torch.float32)

    # Group by prompt: [num_prompts, g]
    grouped = rewards.view(-1, g)
    # Leave-one-out: for each sample, baseline = (group_sum - self) / (N - 1)
    # advantage = reward - baseline = reward * N/(N-1) - group_mean * N/(N-1)
    #           = N/(N-1) * (reward - group_mean)
    #
    # Equivalently: score - (sum - score) / (N - 1)
    #             = score - sum/(N-1) + score/(N-1)
    #             = score * N/(N-1) - sum/(N-1)
    group_sum = grouped.sum(dim=1, keepdim=True)
    n = float(g)
    # baseline_i = (group_sum - score_i) / (N - 1)
    # advantage_i = score_i - baseline_i
    advantages_grouped = rewards.view(-1, g) - (group_sum - grouped) / (n - 1.0)
    adv_flat = advantages_grouped.reshape(-1)

    # Scatter to token level and apply masked whitening
    # Expand [B] -> [B, T] using response_mask
    adv_tokens = adv_flat.unsqueeze(-1) * response_mask_f
    advantages = _masked_whiten(adv_tokens, response_mask_f, shift_mean=True)

    batch.tensors["advantages"] = advantages
    logger.debug(
        "RLOO advantages: mean=%.6f std=%.6f",
        adv_flat.mean().item(),
        adv_flat.std().item(),
    )
    return batch
