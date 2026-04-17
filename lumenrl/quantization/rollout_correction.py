"""Token-level rollout correction (TIS / MIS) for FP8 vs BF16 log-probability mismatch."""

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
    """Multiplicative importance sampling correction without truncation."""
    log_ratio = bf16_logprobs - fp8_logprobs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    return ratio * advantages


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
