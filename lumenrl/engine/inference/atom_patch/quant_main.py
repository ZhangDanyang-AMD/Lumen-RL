###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in ATOM main-456ba2a dense MXFP4-2D recipe; no ATOM source edits.

Call ``install()`` after selecting the recipe and *before* constructing ATOM's
QuantizationConfig/Linear instances. No GPU library is imported by this module.
The GPU quantization and GEMM implementations remain AITER-owned public APIs.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os

logger = logging.getLogger(__name__)
_FORMAT = "mxfp4_2d"
_TRUE = frozenset(("1", "true", "yes", "on"))
_INSTALLED = False


def _aligned_spec_class():
    """Stable, spawn-pickleable spec identity without default-off ATOM import."""
    cached = globals().get("_AlignedLayerQuantConfig")
    if cached is not None:
        return cached
    from atom.quant_spec import LayerQuantConfig

    if "weight_block_2d" in LayerQuantConfig.__dataclass_fields__:
        return LayerQuantConfig
    if getattr(LayerQuantConfig, "__slots__", None):
        raise RuntimeError("mxfp4_2d cannot extend slotted ATOM quant specs")

    @dataclasses.dataclass(frozen=True)
    class _AlignedLayerQuantConfig(LayerQuantConfig):
        weight_block_2d: bool = False

    _AlignedLayerQuantConfig.__module__ = __name__
    _AlignedLayerQuantConfig.__qualname__ = "_AlignedLayerQuantConfig"
    globals()["_AlignedLayerQuantConfig"] = _AlignedLayerQuantConfig
    return _AlignedLayerQuantConfig


def __getattr__(name):
    if name == "_AlignedLayerQuantConfig":
        return _aligned_spec_class()
    raise AttributeError(name)


def _selected(config: dict) -> bool:
    """Only activate for a requested online target, never for ordinary MXFP4."""
    if not isinstance(config, dict):
        return False
    formats = [config.get("global_quant_config", "")]
    layers = config.get("layer_quant_config", {})
    if isinstance(layers, dict):
        formats.extend(layers.values())
    return any(
        isinstance(fmt, str) and fmt.strip().lower() == _FORMAT for fmt in formats
    )


def _is_2d(spec) -> bool:
    return bool(getattr(spec, "weight_block_2d", False))


def _check_env() -> None:
    if (
        os.environ.get("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", "0").strip().lower()
        not in _TRUE
    ):
        raise ValueError(
            "mxfp4_2d requires ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1 before ATOM import"
        )


def quantize_mxfp4_2d_weight(weight):
    """Pack 32x32 Even/RTN BF16 weights and expand E8M0 scales per row."""
    import torch

    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("mxfp4_2d requires a BF16 matrix")
    if any(size <= 0 or size % 32 for size in weight.shape):
        raise ValueError(f"mxfp4_2d requires N/K divisible by 32: {weight.shape}")
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant_blockscale
    except ImportError as exc:
        raise RuntimeError(
            "mxfp4_2d requires AITER dynamic_mxfp4_quant_blockscale"
        ) from exc
    packed, scales = dynamic_mxfp4_quant_blockscale(
        weight.contiguous(), scaling_mode="even"
    )
    return packed, scales.repeat_interleave(32, dim=0).contiguous()


def quantize_mxfp4_2d_activation(x):
    """Quantize BF16 activations as deterministic per-row 1x32 Even/RTN."""
    import torch

    if x.ndim < 2 or x.dtype != torch.bfloat16 or x.shape[-1] <= 0 or x.shape[-1] % 32:
        raise ValueError("mxfp4_2d activations require BF16 and K divisible by 32")
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError as exc:
        raise RuntimeError("mxfp4_2d requires AITER dynamic_mxfp4_quant") from exc
    return dynamic_mxfp4_quant(
        x.reshape(-1, x.shape[-1]).contiguous(), scaling_mode="even", use_sr=False
    )


def _gemm_2d(linear, layer, x, x_scale, otype):
    import torch

    if x_scale is not None:
        raise ValueError("mxfp4_2d requires unquantized BF16 activation input")
    _check_env()
    if not linear.use_fp4_non_shuffle_triton_gemm() or linear.gemm_afp4wfp4 is None:
        raise RuntimeError("mxfp4_2d requires AITER gemm_afp4wfp4 non-shuffle backend")
    x_packed, activation_scale = quantize_mxfp4_2d_activation(x)
    rows = x_packed.shape[0]
    block = linear.MXFP4_QUANT_BLOCK_SIZE
    y = torch.empty(
        ((rows + block - 1) // block * block, layer.output_size),
        dtype=otype,
        device=x.device,
    )
    # AITER consumes row-major packed bytes and row-expanded E8M0 scales.
    y = linear.gemm_afp4wfp4(
        x_packed.view(torch.uint8),
        layer.weight.view(torch.uint8),
        activation_scale.view(torch.uint8),
        layer.weight_scale.data.view(torch.uint8),
        otype,
        y,
    )
    return y[:rows, ...].reshape(*x.shape[:-1], layer.output_size)


def install() -> bool:
    """Install the recipe once; returns False when the explicit opt-in is absent.

    Set ``LUMEN_ATOM_MXFP4_2D=1`` and
    ``ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1`` before calling. This installer
    intentionally does not alter the shared ATOM import hook or sitecustomize.
    """
    global _INSTALLED
    if os.environ.get("LUMEN_ATOM_MXFP4_2D", "0").strip().lower() not in _TRUE:
        return False
    _check_env()
    if _INSTALLED:
        return True
    # ATOM linear selects the non-shuffle backend at import. Install before
    # building quant configs or linears; an already-imported module is safe
    # only if its dispatch flag and AITER backend match the selected recipe.

    import atom.config as config_module
    import torch
    from atom import quant_spec
    from atom.model_ops import linear
    from torch import nn

    parser_cls = quant_spec.QuarkOnlineParser
    config_cls = config_module.QuantizationConfig
    linear_cls = linear.LinearBase
    # Pinned ATOM main has an immutable spec without weight_block_2d. The
    # opt-in recipe must extend that spec before parsing its selected format.
    spec_cls = quant_spec.LayerQuantConfig
    if "weight_block_2d" not in spec_cls.__dataclass_fields__:
        aligned_spec_cls = _aligned_spec_class()
        quant_spec.LayerQuantConfig = aligned_spec_cls
        config_module.LayerQuantConfig = aligned_spec_cls
        if hasattr(linear, "LayerQuantConfig"):
            linear.LayerQuantConfig = aligned_spec_cls
    original_parse = parser_cls.parse
    original_init = config_cls.__init__
    original_stream_check = quant_spec.should_stream_online_quant
    if not callable(original_stream_check):
        raise RuntimeError("ATOM main online quant streaming guard unavailable")
    original_online = linear_cls.online_quantize_weight
    original_process = linear_cls.process_weights_after_loading
    original_forward = linear_cls.forward
    from atom.model_ops import layernorm, moe

    original_moe_online = moe.FusedMoE._online_quant
    original_norm_online = layernorm.RMSNorm.online_quantize_activation
    if not callable(original_moe_online) or not callable(original_norm_online):
        raise RuntimeError("ATOM main quantized activation/expert contract unavailable")

    if not linear.use_fp4_non_shuffle_triton_gemm() or linear.gemm_afp4wfp4 is None:
        raise RuntimeError(
            "mxfp4_2d requires AITER gemm_afp4wfp4 available during ATOM import"
        )

    @functools.wraps(original_parse)
    def parse(self, online_quant_config):
        if not _selected(online_quant_config):
            return original_parse(self, online_quant_config)
        # Keep ATOM's parser responsible for every other format and exclusion.
        normalized = dict(online_quant_config)
        global_format = normalized.get("global_quant_config")
        if isinstance(global_format, str) and global_format.strip().lower() == _FORMAT:
            normalized["global_quant_config"] = "mxfp4"
        layers = normalized.get("layer_quant_config")
        if isinstance(layers, dict):
            normalized["layer_quant_config"] = {
                name: (
                    "mxfp4"
                    if isinstance(fmt, str) and fmt.strip().lower() == _FORMAT
                    else fmt
                )
                for name, fmt in layers.items()
            }
        result = original_parse(self, normalized)
        if isinstance(global_format, str) and global_format.strip().lower() == _FORMAT:
            result.global_spec = dataclasses.replace(
                result.global_spec, weight_block_2d=True
            )
        if isinstance(layers, dict):
            result.layer_pattern_specs = [
                (
                    name,
                    (
                        dataclasses.replace(spec, weight_block_2d=True)
                        if isinstance(layers.get(name), str)
                        and layers[name].strip().lower() == _FORMAT
                        else spec
                    ),
                )
                for name, spec in result.layer_pattern_specs
            ]
        return result

    @functools.wraps(original_stream_check)
    def should_stream_online_quant(
        quant_config, prefix, source_quant_type, source_quant_dtype
    ):
        if quant_config is not None and getattr(quant_config, "online_quant", False):
            selected = quant_config.get_layer_quant_config(
                prefix, use_online_quant=True
            )
            if _is_2d(selected):
                # A 32x32 weight scale requires a complete BF16 projection.
                return False
        return original_stream_check(
            quant_config, prefix, source_quant_type, source_quant_dtype
        )

    @functools.wraps(original_init)
    def config_init(self, config=None, online_quant_config=None, model=None):
        if _selected(online_quant_config):
            _check_env()
        original_init(self, config, online_quant_config, model=model)
        if _selected(online_quant_config):
            # ATOM's hash includes online specs, which now encode this recipe.
            self.online_quant_config_raw = online_quant_config

    @functools.wraps(original_online)
    def online_quantize_weight(self):
        if self.quant_config is None or not self.quant_config.online_quant:
            return original_online(self)
        target = self.quant_config.get_layer_quant_config(
            self.prefix, use_online_quant=True
        )
        if not _is_2d(target):
            return original_online(self)
        _check_env()
        if self.tp_size != 1:
            raise ValueError("mxfp4_2d prototype requires TP=1")
        if self._stream_online_quant:
            raise ValueError("mxfp4_2d requires non-streamed full BF16 weights")
        if getattr(self, "native_a8_group_rows", None) is not None:
            raise ValueError("mxfp4_2d does not support native group32 source weights")
        if (
            self.quant_type.value != linear.QuantType.No.value
            or self.weight.dtype != torch.bfloat16
        ):
            raise ValueError("mxfp4_2d requires unquantized BF16 source weights")
        if any(size % 32 for size in self.output_partition_sizes):
            raise ValueError(
                "mxfp4_2d packed projection boundaries must align to 32 rows"
            )
        if self.weight.ndim != 2 or self.weight.shape != (
            self.output_size,
            self.input_size,
        ):
            raise ValueError("mxfp4_2d requires a dense two-dimensional projection")
        packed, scales = quantize_mxfp4_2d_weight(self.weight.data)
        if (
            target.quant_type.value != linear.QuantType.per_1x32.value
            or target.quant_dtype != torch.float4_e2m1fn_x2
        ):
            raise ValueError("mxfp4_2d online target must be per_1x32 FP4")
        self.weight = nn.Parameter(packed.view(target.quant_dtype), requires_grad=False)
        self.weight_scale = nn.Parameter(scales, requires_grad=False)
        self.weight.weight_loader_process = self.weight_loader_process
        self.weight_scale.weight_loader_process = self.weight_loader_process
        self.quant_type = target.quant_type
        self.params_dtype = target.quant_dtype
        self.scale_dtype = self.weight_scale.dtype
        self.layer_quant_config = target
        self._stream_online_quant = False
        self.fp4_n32k4_scale = False
        self.mxfp4_2d = True
        self._mxfp4_2d_shape = (self.output_size, self.input_size)
        self._online_quant_info = {
            "layer": self.prefix,
            "quant_type": self.quant_type.name,
            "quant_dtype": str(self.params_dtype),
        }
        if getattr(self, "input_scale", None) is not None:
            self.input_scale = None

    @functools.wraps(original_process)
    def process_weights_after_loading(self):
        # No shuffle may touch row-major 2D scales after online conversion.
        if getattr(self, "mxfp4_2d", False):
            return
        if (
            self.weight.numel()
            and self.quant_config is not None
            and self.quant_config.online_quant
        ):
            target = self.quant_config.get_layer_quant_config(
                self.prefix, use_online_quant=True
            )
            if _is_2d(target):
                self.online_quantize_weight()
                return
        return original_process(self)

    @functools.wraps(original_forward)
    def forward(self, x, x_scale=None, otype=linear.dtypes.bf16, out=None):
        if not getattr(self, "mxfp4_2d", False):
            return original_forward(self, x, x_scale, otype, out)
        if out is not None:
            raise ValueError("mxfp4_2d does not support out=")
        y = _gemm_2d(linear, self, x, x_scale, otype)
        if self.bias is not None:
            y = y + self.bias
        if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
            y = linear.tensor_model_parallel_all_reduce(y)
        return y

    @functools.wraps(original_moe_online)
    def moe_online(self, *args, **kwargs):
        if getattr(self, "online_quant", False):
            target = self.quant_config.get_layer_quant_config(
                self.layer_name, use_online_quant=True
            )
            if _is_2d(target):
                raise ValueError("mxfp4_2d supports dense Linear only")
        return original_moe_online(self, *args, **kwargs)

    @functools.wraps(original_norm_online)
    def norm_online(self, *args, **kwargs):
        if self.quant_config is not None and self.quant_config.online_quant:
            target = self.quant_config.get_layer_quant_config(
                self.prefix, use_online_quant=True
            )
            if _is_2d(target):
                raise ValueError("mxfp4_2d requires an unfused BF16 RMSNorm boundary")
        return original_norm_online(self, *args, **kwargs)

    # All compatibility checks finish before the first mutation. No silent
    # fallback for an explicitly selected quantization recipe.
    parser_cls.parse = parse
    config_cls.__init__ = config_init
    quant_spec.should_stream_online_quant = should_stream_online_quant
    linear.should_stream_online_quant = should_stream_online_quant
    linear_cls.online_quantize_weight = online_quantize_weight
    linear_cls.process_weights_after_loading = process_weights_after_loading
    linear_cls.forward = forward
    moe.FusedMoE._online_quant = moe_online
    layernorm.RMSNorm.online_quantize_activation = norm_online
    _INSTALLED = True
    logger.info("Installed default-off ATOM MXFP4-2D recipe")
    return True
