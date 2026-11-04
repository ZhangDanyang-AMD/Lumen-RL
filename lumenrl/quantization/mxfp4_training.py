"""Training-time MXFP4 setup delegated to Lumen."""

from __future__ import annotations

import os
from typing import Any

import torch.nn as nn

_DISABLE_WEIGHT_CACHE_ENV = "LUMEN_MXFP4_DISABLE_WEIGHT_CACHE"


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def setup_mxfp4_training(
    model: nn.Module,
    training_quant_config: Any,
) -> tuple[Any | None, nn.Module, bool]:
    """Apply Lumen MXFP4 before distributed wrapping.

    Returns ``(manager, model, cache_enabled)``. When MXFP4 is disabled this
    function returns immediately and does not mutate the cache environment.
    """
    if not _value(training_quant_config, "mxfp4", False):
        return None, model, False

    if _value(training_quant_config, "fp8"):
        raise ValueError(
            "quantization.training.fp8 and quantization.training.mxfp4 "
            "cannot be enabled together"
        )
    if _value(training_quant_config, "lumen_linear", False):
        raise ValueError(
            "quantization.training.lumen_linear must be false when MXFP4 is "
            "enabled; MXFP4 installs the selected Linear forwards itself"
        )

    disable_weight_cache = bool(
        _value(training_quant_config, "mxfp4_disable_weight_cache", False)
    )
    if disable_weight_cache:
        os.environ[_DISABLE_WEIGHT_CACHE_ENV] = "1"
    else:
        os.environ.pop(_DISABLE_WEIGHT_CACHE_ENV, None)

    # Lumen owns the selection schema and all matching/patching behavior.
    # Imports stay local so a BF16 run does not require Lumen to be installed.
    from lumen.config import LumenConfig

    selection = _value(
        training_quant_config,
        "mxfp4_quant_weight_selection",
        {"mode": "all_linear"},
    )

    lumen_config = LumenConfig(
        format="mxfp4",
        scaling="blockwise",
        block_size=32,
        quantize_activation=True,
        fp8_wgrad=True,
        quantize_grad=None,
        first_last_layers_bf16=False,
        lumen_norm=bool(_value(training_quant_config, "lumen_norm", False)),
        lumen_linear=False,
        hf_attn_patch=bool(_value(training_quant_config, "hf_attn_patch", False)),
        fused_mlp=bool(_value(training_quant_config, "fused_mlp", False)),
        fused_rope=bool(_value(training_quant_config, "fused_rope", False)),
        mxfp4_quant_weight_selection=selection,
    )
    manager, model = lumen_config.enable(model)
    return manager, model, not disable_weight_cache
