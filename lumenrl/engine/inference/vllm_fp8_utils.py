"""vLLM online ``fp8_per_block`` support for LumenRL rollout (verl-free).

This is a self-contained port of the *online per-block FP8* pieces of verl's
``verl/utils/vllm/vllm_fp8_utils.py`` + ``vllm_rollout/utils.py``, with **no
dependency on ``verl``**. It only imports from ``vllm`` / ``torch``.

Strict reproduction of verl's ``rollout.quantization=fp8_per_block``:

* vLLM natively registers the ``fp8_per_block`` *online* quantization method
  (``OnlineQuantizationConfig``); the BF16 training weights are re-quantized to
  per-128-block FP8 inside vLLM on every weight load.
* On ROCm the online per-block weight post-processing needs a guard so it does
  not double-convert an already-FP8 (e4m3fnuz) tensor -> ``apply_vllm_fp8_per_block_patches``.
* Because online quant rebuilds per-layer FP8 scales, each RL weight sync must
  wrap ``model.load_weights`` with vLLM's layerwise-reload lifecycle
  (``prepare_*`` before the first bucket, ``finalize_*`` after the last).

Only what ``fp8_per_block`` needs is ported here (no static-FP8
``load_quanted_weights``, no LoRA, no QAT / NVFP4).
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Module-level idempotency guard: the ROCm per-block patches are process-global
# and must only be started once per worker process.
_PER_BLOCK_PATCHES: list = []


def is_online_quant_model(vllm_config) -> bool:
    """Return True if vLLM is using an online quantization (e.g. ``fp8_per_block``)."""
    try:
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )
    except ImportError:
        return False

    quant_config = getattr(vllm_config, "quant_config", None)
    return isinstance(quant_config, OnlineQuantizationConfig)


def process_fp8_weight_block_strategy_rocm_safe(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ROCm guard for online ``fp8_per_block`` weight post-processing.

    On AMD, online ``fp8_per_block`` may already emit the platform FP8 dtype
    (e4m3fnuz). Only normalize ``e4m3fn -> e4m3fnuz`` when the tensor is still
    ``e4m3fn`` to avoid a second (corrupting) conversion.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _maybe_pad_fp8_weight,
    )
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
    )
    from vllm.platforms import current_platform

    if current_platform.is_fp8_fnuz() and weight.dtype == torch.float8_e4m3fn:
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale
        )

    weight = _maybe_pad_fp8_weight(weight)
    return weight, weight_scale


def apply_vllm_fp8_per_block_patches() -> None:
    """Patch vLLM online ``fp8_per_block`` for ROCm FP8 weight processing.

    Idempotent + resilient: each target is patched independently so a vLLM
    version that lacks one of the symbols does not abort the others.
    """
    global _PER_BLOCK_PATCHES
    if _PER_BLOCK_PATCHES:
        logger.debug("vLLM fp8_per_block patches already applied")
        return

    targets = [
        "vllm.model_executor.layers.quantization.utils.fp8_utils.process_fp8_weight_block_strategy",
        "vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel.process_fp8_weight_block_strategy",
    ]
    applied = 0
    for target in targets:
        try:
            patcher = patch(target, process_fp8_weight_block_strategy_rocm_safe)
            patcher.start()
            _PER_BLOCK_PATCHES.append(patcher)
            applied += 1
        except (AttributeError, ModuleNotFoundError) as exc:
            logger.warning("skip fp8_per_block patch %s: %s", target, exc)
    logger.info("Applied vLLM fp8_per_block ROCm patches (%d/%d)", applied, len(targets))


def prepare_online_quantized_weights_for_loading(model) -> None:
    """Set up vLLM per-layer reload state BEFORE the first weight bucket."""
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

    initialize_layerwise_reload(model)


def finalize_online_quantized_weights_loading(model, model_config) -> None:
    """Finalize vLLM per-layer reload AFTER the last weight bucket.

    Rebuilds the per-block FP8 weight scales for the freshly loaded BF16 weights.
    """
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

    finalize_layerwise_reload(model, model_config)
