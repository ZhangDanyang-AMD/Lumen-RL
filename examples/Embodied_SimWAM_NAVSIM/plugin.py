"""Lumen kernel switch for SimWAM FlowGRPO.

Called after ``configure_grpo`` so the video expert, the LoRA-wrapped action
expert, the frozen reference expert and the VAE are all in the module tree.
Quantization stays off.

* ``hf_attn_patch`` — ``F.scaled_dot_product_attention``. A bool keep-mask
  (True = attend) is converted to an additive bias inside the patch.
* ``lumen_linear`` — every ``nn.Linear``. LoRA A/B are bare parameters, so
  the same ``lumen.ops.bf16_gemm.linear`` is installed on ``LoRALinear.forward``.
* ``aiter_conv`` — ``nn.Conv2d`` and ``nn.Conv3d`` go through AITER.
  ``CausalConv3d.forward`` is left in place; only ``_conv_forward`` changes.

Each patch is applied on its own, so one kernel failing to load does not turn
the others off.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_OFF = {"", "0", "false", "none", "off"}
_PATCHES = ("hf_attn_patch", "lumen_linear", "aiter_conv")


def _log_coverage(model) -> None:
    import torch.nn as nn

    n_linear = n_conv2d = n_conv3d = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            n_conv3d += 1
        elif isinstance(module, nn.Conv2d):
            n_conv2d += 1
        elif isinstance(module, nn.Linear):
            n_linear += 1
    parts = []
    for name in ("video_expert", "action_expert", "ref_action_expert", "vae"):
        child = getattr(model, name, None)
        if child is None:
            continue
        nparam = sum(p.numel() for p in child.parameters()) / 1e9
        parts.append(f"{name}={nparam:.2f}B")
    logger.info(
        "SimWAM coverage: %s | nn.Linear=%d nn.Conv2d=%d nn.Conv3d=%d",
        ", ".join(parts) or "unpacked", n_linear, n_conv2d, n_conv3d,
    )


def _patch_lora_gemm() -> None:
    """Route LoRA A/B through the same bf16 GEMM as ``nn.Linear``.

    ``lora_A`` / ``lora_B`` are bare parameters, so ``lumen_linear`` does not
    see them. Patching the class covers adapters created before or after.
    """
    from simwam.models.wan22.lora import LoRALinear

    if getattr(LoRALinear.forward, "_lumen_bf16", False):
        return

    def forward(self, x):
        from lumen.ops.bf16_gemm import linear

        out = self.base(x)
        dropped = self.lora_dropout(x)
        mid = linear(dropped, self.lora_A)
        lora = linear(mid, self.lora_B)
        return out + self.scaling * lora

    forward._lumen_bf16 = True
    LoRALinear.forward = forward


def install_grpo_hook() -> None:
    """Apply Lumen after ``configure_grpo`` without editing SimWAM sources.

    ``configure_grpo`` copies the frozen reference expert and wraps LoRA on
    the action expert. Patching any earlier would miss those modules.
    """
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.configure_grpo, "_lumen_hook", False):
        return
    original = ActionGRPOMixin.configure_grpo

    def configure_grpo(self, grpo_cfg):
        original(self, grpo_cfg)
        spec = os.environ.get("LUMEN_ENABLE", "").strip()
        if spec:
            enable_lumen(self, spec)

    configure_grpo._lumen_hook = True
    ActionGRPOMixin.configure_grpo = configure_grpo


def enable_lumen(model, spec: str):
    """Apply ``LumenConfig.enable`` once per requested patch."""
    requested = []
    for key in str(spec).split(","):
        key = key.strip()
        if not key or key.lower() in _OFF:
            continue
        requested.append(key)
    if not requested:
        return model

    from lumen.config import LumenConfig

    applied = []
    for key in requested:
        if key not in _PATCHES:
            logger.warning("Unknown LUMEN_ENABLE flag %r; ignored", key)
            continue
        try:
            _, model = LumenConfig(scaling="none", **{key: True}).enable(model)
            applied.append(key)
        except Exception as exc:
            logger.warning("%s failed (%s); that op stays on torch", key, exc)
    if "lumen_linear" in applied:
        try:
            _patch_lora_gemm()
        except Exception as exc:
            logger.warning("LoRA GEMM patch failed (%s); LoRA A/B stay on torch", exc)
    _log_coverage(model)
    logger.info("Lumen enabled on SimWAM: %s", applied)
    return model
