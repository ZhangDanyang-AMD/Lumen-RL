"""Megatron backend with optional Lumen feature integration."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore[assignment]

try:
    from lumen.config import LumenConfig
except ImportError:  # pragma: no cover - optional dependency
    LumenConfig = None  # type: ignore[assignment]

try:
    from lumen.ops.attention.hf_patch import patch_hf_sdpa
except ImportError:  # pragma: no cover - optional dependency
    patch_hf_sdpa = None  # type: ignore[assignment]

try:
    from megatron.core.transformer.moe import MoEConfig
except ImportError:  # pragma: no cover - optional dependency
    MoEConfig = None  # type: ignore[assignment]


class _MegatronStubLM(nn.Module):
    """Dense LM placeholder when Megatron-Core is not linked."""

    def __init__(self, hidden_size: int = 256, vocab_size: int = 32000) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        out, _ = self.decoder(x)
        return self.head(out)


def _to_dict(raw_cfg: Any) -> dict[str, Any]:
    if raw_cfg is None:
        return {}
    if is_dataclass(raw_cfg):
        return dict(asdict(raw_cfg))
    if isinstance(raw_cfg, dict):
        return dict(raw_cfg)
    return dict(vars(raw_cfg))


def _pick_megatron_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    root = config or {}
    return _to_dict(root.get("megatron") or root.get("megatron_cfg") or root)


def _patch_hf_attention_with_lumen() -> None:
    if patch_hf_sdpa is None:
        logger.debug("MegatronBackend: lumen hf attention patch unavailable.")
        return
    try:
        patch_hf_sdpa()
        logger.info("MegatronBackend: applied Lumen HF SDPA patch.")
    except Exception as exc:  # pragma: no cover - best-effort optional patch
        logger.warning("MegatronBackend: failed to apply Lumen HF patch (%s).", exc)


def _load_hf_model(model_name: str, torch_dtype: torch.dtype) -> nn.Module | None:
    if AutoModelForCausalLM is None:
        logger.warning(
            "MegatronBackend.build_model: transformers unavailable; cannot load model=%s.",
            model_name,
        )
        return None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("MegatronBackend.build_model: loaded HF model=%s.", model_name)
        return model
    except Exception as exc:  # pragma: no cover - dependency/runtime-specific
        logger.warning(
            "MegatronBackend.build_model: failed loading HF model=%s (%s); falling back.",
            model_name,
            exc,
        )
        return None


class MegatronBackend:
    """Build Megatron-path model and apply optional Lumen/MoE features."""

    @staticmethod
    def build_model(model_name: str, config: dict[str, Any] | None) -> nn.Module:
        """Instantiate a Megatron policy module with graceful fallback."""
        meg_cfg = _pick_megatron_cfg(config)
        hidden = int(meg_cfg.get("hidden_size", 256))
        vocab = int(meg_cfg.get("vocab_size", 32000))
        use_tiny_lm = bool(meg_cfg.get("use_tiny_lm", False))
        hf_attn_patch = bool(meg_cfg.get("hf_attn_patch", True))
        if hf_attn_patch:
            _patch_hf_attention_with_lumen()

        dtype_name = str(meg_cfg.get("dtype", "bfloat16")).lower()
        torch_dtype = torch.float16 if dtype_name in ("fp16", "float16") else torch.bfloat16

        if not use_tiny_lm and model_name:
            model = _load_hf_model(model_name, torch_dtype=torch_dtype)
            if model is not None:
                return model

        if use_tiny_lm:
            logger.info("MegatronBackend.build_model: using Tiny stub LM by configuration.")
        else:
            logger.warning(
                "MegatronBackend.build_model: using stub fallback (model_name=%s).",
                model_name or "<empty>",
            )
        return _MegatronStubLM(hidden_size=hidden, vocab_size=vocab)

    @staticmethod
    def apply_lumen_spec(model: nn.Module, megatron_config: dict[str, Any] | None) -> nn.Module:
        """Apply optional Lumen and MoE-related capabilities to the model."""
        megatron_config = _to_dict(megatron_config)
        num_experts = megatron_config.get("num_experts")
        if num_experts:
            logger.info(
                "MegatronBackend.apply_lumen_spec: MoE requested with num_experts=%s.",
                num_experts,
            )
            if MoEConfig is None:
                logger.warning(
                    "MegatronBackend.apply_lumen_spec: MoEConfig unavailable; running without Megatron MoE wiring."
                )
            else:
                logger.debug("MegatronBackend.apply_lumen_spec: MoEConfig import OK.")

        lumen_cfg = _to_dict(megatron_config.get("lumen"))
        fp8_enabled = bool(lumen_cfg.get("fp8", os.environ.get("LUMEN_FP8", "0") == "1"))
        fp8_pm = bool(
            lumen_cfg.get(
                "fp8_param_manager",
                os.environ.get("FP8_PARAM_MANAGER", "0") == "1",
            )
        )
        lumen_norm = bool(lumen_cfg.get("lumen_norm", os.environ.get("LUMEN_NORM", "0") == "1"))
        fused_mlp = bool(lumen_cfg.get("fused_mlp", os.environ.get("LUMEN_FUSED_MLP", "0") == "1"))
        fused_rope = bool(lumen_cfg.get("fused_rope", os.environ.get("LUMEN_FUSED_ROPE", "0") == "1"))
        fp8_weight_cache = bool(lumen_cfg.get("fp8_weight_cache", False))

        if not (fp8_enabled or fp8_pm or lumen_norm or fused_mlp or fused_rope):
            logger.debug("MegatronBackend.apply_lumen_spec: no Lumen feature requested.")
            return model

        if fp8_enabled and not torch.cuda.is_available():
            raise ValueError("MegatronBackend.apply_lumen_spec: FP8 requires CUDA-capable runtime.")

        if LumenConfig is None:
            logger.warning(
                "MegatronBackend.apply_lumen_spec: lumen package not installed; skipping requested features."
            )
            return model

        try:
            cfg = LumenConfig(
                fp8_param_manager=fp8_pm,
                lumen_norm=lumen_norm,
                fused_mlp=fused_mlp,
                fused_rope=fused_rope,
                fp8_weight_cache=fp8_weight_cache,
                hf_attn_patch=bool(lumen_cfg.get("hf_attn_patch", True)),
            )
            _manager, patched_model = cfg.enable(model)
            logger.info(
                "MegatronBackend.apply_lumen_spec: applied Lumen features (fp8=%s, fp8pm=%s, norm=%s, fused_mlp=%s, fused_rope=%s).",
                fp8_enabled,
                fp8_pm,
                lumen_norm,
                fused_mlp,
                fused_rope,
            )
            return patched_model
        except Exception as exc:  # pragma: no cover - depends on installed lumen/megatron stack
            logger.warning("MegatronBackend.apply_lumen_spec: Lumen enable failed (%s); using original model.", exc)
            return model
