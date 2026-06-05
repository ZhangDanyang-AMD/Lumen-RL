"""Token-level rollout correction (TIS / MIS) for FP8 vs BF16 log-probability mismatch.

Extended with IS weight computation and rejection sampling.
Derived from verl/trainer/ppo/rollout_corr_helper.py and
verl/trainer/ppo/ray_trainer.py L1481-1567.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from lumenrl.core.config import LumenRLConfig, QuantizationConfig, RolloutCorrectionConfig
from lumenrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


def token_level_tis(
    bf16_logprobs: Tensor,
    fp8_logprobs: Tensor,
    advantages: Tensor,
    clip: float = 1.5,
) -> Tensor:
    """Truncated importance sampling correction at token granularity.

    Uses the closed-form clipped density ratio between the BF16 (reference) and FP8
    (rollout) log-probabilities:

    .. math::

        \\rho_t = \\exp(\\log \\pi_\\text{bf16} - \\log \\pi_\\text{fp8}),\\quad
        \\tilde{\\rho}_t = \\mathrm{clip}(\\rho_t, e^{-c}, e^{c}),\\quad
        \\tilde{A}_t = \\tilde{\\rho}_t A_t

    where ``clip=c`` symmetrically clamps the multiplicative ratio to ``[1/c, c]``.
    """
    log_ratio = bf16_logprobs - fp8_logprobs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 / clip, clip)
    return clipped * advantages


def token_level_mis(
    bf16_logprobs: Tensor,
    fp8_logprobs: Tensor,
    advantages: Tensor,
) -> Tensor:
    """Self-normalizing multiplicative importance sampling correction.

    Unlike TIS, MIS avoids hard clipping and instead normalizes the raw
    density ratio so that the per-token mean weight equals 1.  This
    preserves the expected gradient direction while absorbing FP8/BF16
    distributional shift:

    .. math::

        \\rho_t = \\exp(\\log \\pi_\\text{bf16} - \\log \\pi_\\text{fp8}),\\quad
        \\bar{\\rho}_t = \\rho_t / \\mathrm{mean}(\\rho),\\quad
        \\tilde{A}_t = \\bar{\\rho}_t A_t
    """
    log_ratio = bf16_logprobs - fp8_logprobs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    normalized = ratio / ratio.mean().clamp(min=1e-8)
    return normalized * advantages


def _resolve_rollout_correction_config(config: Any) -> RolloutCorrectionConfig:
    if isinstance(config, RolloutCorrectionConfig):
        return config
    if isinstance(config, QuantizationConfig):
        return config.rollout_correction
    if isinstance(config, LumenRLConfig):
        return config.quantization.rollout_correction
    raise TypeError(
        "config must be RolloutCorrectionConfig, QuantizationConfig, or LumenRLConfig, "
        f"got {type(config)!r}"
    )


def _pick_fp8_logprobs(batch: DataProto) -> Tensor:
    if "fp8_logprobs" in batch.tensors:
        return batch["fp8_logprobs"]
    if "fp8_log_probs" in batch.tensors:
        return batch["fp8_log_probs"]
    raise KeyError("DataProto must contain 'fp8_logprobs' or 'fp8_log_probs'.")


def _pick_bf16_logprobs(batch: DataProto) -> Tensor:
    for key in ("bf16_logprobs", "old_log_probs", "ref_log_probs"):
        if key in batch.tensors:
            return batch[key]
    raise KeyError("DataProto must contain 'bf16_logprobs', 'old_log_probs', or 'ref_log_probs'.")


def apply_rollout_correction(batch: DataProto, config: Any) -> DataProto:
    """Apply configured token-level correction to advantages in-place on a new DataProto."""
    rcfg = _resolve_rollout_correction_config(config)
    if not rcfg.enabled:
        return batch

    bf16_lp = _pick_bf16_logprobs(batch)
    fp8_lp = _pick_fp8_logprobs(batch)
    if "advantages" not in batch.tensors:
        raise KeyError("DataProto must contain 'advantages' for rollout correction.")
    advantages = batch["advantages"]

    method = rcfg.method.lower().strip()
    if method == "tis":
        corrected = token_level_tis(bf16_lp, fp8_lp, advantages, clip=rcfg.clip)
    elif method in {"mis", "multiplicative"}:
        corrected = token_level_mis(bf16_lp, fp8_lp, advantages)
    else:
        raise ValueError(f"Unknown rollout correction method: {rcfg.method!r}")

    out = DataProto(tensors=dict(batch.tensors), meta=dict(batch.meta))
    out["advantages"] = corrected
    out.meta["rollout_correction"] = {"method": method, "clip": rcfg.clip}
    logger.debug("apply_rollout_correction: method=%s", method)
    return out


# ---------------------------------------------------------------------------
# Extended rollout correction: IS weights & rejection sampling
# (verl/trainer/ppo/rollout_corr_helper.py, verl/trainer/ppo/ray_trainer.py L1481-1567)
# ---------------------------------------------------------------------------

def compute_rollout_is_weights(
    old_log_probs: Tensor,
    rollout_log_probs: Tensor,
    response_mask: Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Compute importance sampling weights between current and rollout policies.

    Derived from verl/trainer/ppo/rollout_corr_helper.py
    ``compute_rollout_correction_and_rejection_mask()``.

    Args:
        old_log_probs: Log-probs from current training policy [B, T].
        rollout_log_probs: Log-probs from rollout policy [B, T].
        response_mask: Valid token mask [B, T].
        rollout_is: Aggregation level — ``"token"`` or ``"sequence"``.
        rollout_is_threshold: Upper truncation threshold.
        rollout_is_batch_normalize: If True, normalize weights to mean=1.

    Returns:
        ``(is_weights, metrics)`` where ``is_weights`` is [B, T].
    """
    log_ratio = old_log_probs - rollout_log_probs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    mask_f = response_mask.float()

    if rollout_is == "token":
        raw_weights = torch.exp(log_ratio)
        weights = torch.clamp(raw_weights, max=rollout_is_threshold)
    elif rollout_is == "sequence":
        seq_len = mask_f.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (log_ratio * mask_f).sum(dim=-1) / seq_len
        seq_weights = torch.exp(seq_log_ratio)
        seq_weights = torch.clamp(seq_weights, max=rollout_is_threshold)
        weights = seq_weights.unsqueeze(-1).expand_as(response_mask)
    else:
        weights = torch.ones_like(response_mask, dtype=torch.float32)

    if rollout_is_batch_normalize and weights.numel() > 0:
        w_mean = (weights * mask_f).sum() / mask_f.sum().clamp(min=1)
        weights = weights / w_mean.clamp(min=1e-8)

    metrics = {
        "rollout_correction/is_weight_mean": float((weights * mask_f).sum() / mask_f.sum().clamp(min=1)),
        "rollout_correction/is_weight_max": float(weights.max()),
    }
    return weights * mask_f, metrics


def apply_rejection_sampling(
    response_mask: Tensor,
    is_weights: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    """Zero out response_mask entries where IS weight exceeds threshold.

    Derived from verl/trainer/ppo/rollout_corr_helper.py rejection sampling logic.

    Args:
        response_mask: Original response mask [B, T].
        is_weights: Importance sampling weights [B, T].
        threshold: Rejection threshold.  0 = disabled.

    Returns:
        Modified response mask with rejected tokens zeroed.
    """
    if threshold <= 0.0:
        return response_mask
    reject = is_weights > threshold
    modified = response_mask.clone()
    modified[reject] = 0
    return modified
