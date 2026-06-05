"""FSDP2 training backend: real HF model loading + Lumen FP8 + FSDP2 sharding."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _TinyLM(nn.Module):
    """Compact causal LM for tests and offline development."""

    def __init__(self, vocab_size: int = 32000, dim: int = 256, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(dim, nhead=4, batch_first=True) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def _patch_hf_attention_with_lumen() -> None:
    """Patch HF sdpa attention to use AITER CK kernels via Lumen.

    Uses ``lumen.ops.attention.hf_patch`` which replaces the ``sdpa``
    entry in HF's ``ALL_ATTENTION_FUNCTIONS`` with an AITER-backed
    implementation that supports both forward and backward.

    Then installs a varlen wrapper on top for sequence packing support.
    """
    try:
        from lumen.ops.attention.hf_patch import patch_hf_sdpa
        patch_hf_sdpa()
    except ImportError:
        logger.warning("Lumen HF attention patch not available; using default SDPA/AOTriton.")
    except Exception as exc:
        logger.error("Lumen HF attention patch failed: %s", exc, exc_info=True)

    # Install varlen-aware wrapper for sequence packing (must come AFTER patch_hf_sdpa)
    try:
        from lumenrl.engine.training.packing import patch_attention_for_packing
        patch_attention_for_packing()
    except Exception as exc:
        logger.warning("Packing attention patch failed: %s", exc)


_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_dtype(value: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Resolve a string or torch.dtype config value to a torch.dtype."""
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        try:
            return _DTYPE_ALIASES[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype string: {value!r}") from exc
    raise TypeError(f"Cannot resolve dtype from {value!r}")


def _load_hf_model(model_name: str, torch_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Load a HuggingFace causal LM with gradient checkpointing.

    All ranks load the full model from disk to ensure identical weights
    before FSDP2 sharding. Model files on /dev/shm make this fast.
    """
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    rank = dist.get_rank() if dist.is_initialized() else 0

    _patch_hf_attention_with_lumen()

    logger.info("[rank %d] Loading HF model: %s (dtype=%s)", rank, model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    logger.info("[rank %d] Model ready: %s (%d params)", rank, model_name, sum(p.numel() for p in model.parameters()))
    return model


def _apply_lumen_fp8(model: nn.Module, quant_config: dict[str, Any]) -> nn.Module:
    """Apply Lumen optimizations to the model before FSDP2 sharding.

    Supports both FP8 and BF16 Lumen features (norm, fused_mlp, fused_rope).
    """
    if not quant_config:
        return model

    fp8_enabled = quant_config.get("fp8") or os.environ.get("LUMEN_FP8", "0") == "1"
    fp8_pm = quant_config.get("fp8_param_manager") or os.environ.get("FP8_PARAM_MANAGER", "0") == "1"
    lumen_norm = quant_config.get("lumen_norm") or os.environ.get("LUMEN_NORM", "0") == "1"
    fused_mlp = quant_config.get("fused_mlp") or os.environ.get("LUMEN_FUSED_MLP", "0") == "1"
    fused_rope = quant_config.get("fused_rope") or os.environ.get("LUMEN_FUSED_ROPE", "0") == "1"

    if not (fp8_enabled or fp8_pm or lumen_norm or fused_mlp or fused_rope):
        return model

    try:
        from lumen.config import LumenConfig

        kwargs = dict(
            fp8_param_manager=bool(fp8_pm),
            lumen_norm=bool(lumen_norm),
            fused_mlp=bool(fused_mlp),
            fused_rope=bool(fused_rope),
            hf_attn_patch=True,
            fp8_weight_cache=quant_config.get("fp8_weight_cache", False),
        )
        if fp8_enabled:
            # FP8 mode: enable quantized linear and related features
            kwargs.update(
                scaling=os.environ.get("LUMEN_FP8_SCALING", "delayed"),
                format=os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3"),
                block_size=int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128")),
                fp8_activation_store=os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0") == "1",
                fp8_param_gather=os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1",
            )
        else:
            # BF16 mode: disable FP8 quantized linear to avoid unsupported GEMM errors
            kwargs.update(
                quantize_activation=False,
                fp8_wgrad=False,
            )
        cfg = LumenConfig(**kwargs)
        _manager, model = cfg.enable(model)
        logger.info("Lumen optimizations applied (fp8=%s, fp8pm=%s, norm=%s, fused_mlp=%s, fused_rope=%s)",
                     fp8_enabled, fp8_pm, lumen_norm, fused_mlp, fused_rope)
    except ImportError:
        logger.warning("lumen package not installed; skipping Lumen optimizations.")
    except Exception as exc:
        logger.error("Lumen enable() failed: %s", exc, exc_info=True)

    return model


def _apply_fsdp2_sharding(
    model: nn.Module,
    fsdp_config: dict[str, Any],
    fp8_linear: bool = False,
) -> nn.Module:
    """Apply PyTorch FSDP2 ``fully_shard`` to the model.

    All ranks must have identical, fully-loaded parameters on their
    local CUDA device before this call. FSDP2 will then shard them.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        logger.warning("torch.distributed not initialized; returning unsharded model.")
        return model

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_device = torch.device(f"cuda:{local_rank}")

    model.to(local_device)

    param_dtype = _resolve_dtype(fsdp_config.get("param_dtype"), torch.bfloat16)
    reduce_dtype = _resolve_dtype(fsdp_config.get("reduce_dtype"), torch.float32)
    if fp8_linear:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )

    reshard = fsdp_config.get("reshard_after_forward", True)
    offload_policy = None
    if fsdp_config.get("param_offload", False):
        from torch.distributed._composable.fsdp import CPUOffloadPolicy
        offload_policy = CPUOffloadPolicy(pin_memory=True)

    for module in model.modules():
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            for layer in module.layers:
                fully_shard(
                    layer,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard,
                    offload_policy=offload_policy,
                )

    fully_shard(
        model,
        mp_policy=mp_policy,
        reshard_after_forward=reshard,
        offload_policy=offload_policy,
    )

    logger.info("[rank %d] FSDP2 fully_shard applied (fp8_reduce=%s, offload=%s)",
                rank, fp8_linear, fsdp_config.get("param_offload", False))
    return model


def set_requires_gradient_sync(model: nn.Module, requires_sync: bool) -> None:
    """Toggle gradient sync on all FSDP2 units in *model*.

    For gradient accumulation: set to ``False`` on intermediate
    micro-batches so that FSDP2 skips the reduce-scatter in backward,
    letting gradients accumulate locally.  Set back to ``True`` on the
    last micro-batch of each accumulation group so that the final
    backward performs the reduce-scatter.
    """
    from torch.distributed._composable.fsdp import FSDPModule
    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(requires_sync, recurse=True)
    else:
        for mod in model.modules():
            if isinstance(mod, FSDPModule):
                mod.set_requires_gradient_sync(requires_sync, recurse=True)
                break


def set_reshard_after_forward(model: nn.Module, reshard: bool) -> None:
    """Toggle ``reshard_after_forward`` on all FSDP2 units in *model*.

    During generation, setting this to ``False`` keeps parameters
    all-gathered across decode steps, eliminating one all-gather per
    layer per token.  Restore to ``True`` before training so that
    memory stays bounded.

    Uses the official ``FSDPModule.set_reshard_after_forward`` API
    (PyTorch >= 2.6).
    """
    from torch.distributed._composable.fsdp import FSDPModule
    if isinstance(model, FSDPModule):
        model.set_reshard_after_forward(reshard, recurse=True)
    else:
        for mod in model.modules():
            if isinstance(mod, FSDPModule):
                mod.set_reshard_after_forward(reshard, recurse=True)
                break


class FSDP2Backend:
    """Build a model and optionally wrap it with PyTorch FSDP2."""

    @staticmethod
    def build_model(model_name: str, config: dict[str, Any] | None = None) -> nn.Module:
        """Construct the trainable policy network.

        If ``model_name`` points to a real HF model (local path or hub ID),
        loads it. Otherwise falls back to ``_TinyLM`` for testing.
        """
        cfg = config or {}

        # Apply Liger kernel monkey-patches BEFORE model loading
        if cfg.get("use_liger", False):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                logger.info("Applied Liger kernel optimizations.")
            except ImportError:
                logger.warning("use_liger=True but liger-kernel not installed. Skipping.")

        if cfg.get("use_tiny_lm", False) or not model_name:
            arch = cfg.get("tiny_lm", {})
            vocab = int(arch.get("vocab_size", 32000))
            dim = int(arch.get("dim", 256))
            n_layers = int(arch.get("n_layers", 2))
            logger.info("FSDP2Backend.build_model: TinyLM (vocab=%d, dim=%d)", vocab, dim)
            return _TinyLM(vocab_size=vocab, dim=dim, n_layers=n_layers)

        model_dtype = _resolve_dtype(cfg.get("model_dtype"), torch.bfloat16)
        return _load_hf_model(model_name, torch_dtype=model_dtype)

    @staticmethod
    def apply_lumen_optimizations(model: nn.Module, quant_config: dict[str, Any] | None) -> nn.Module:
        """Apply Lumen FP8/norm optimizations before FSDP2 wrapping."""
        return _apply_lumen_fp8(model, quant_config or {})

    @staticmethod
    def apply_fsdp2(model: nn.Module, fsdp_config: dict[str, Any] | None) -> nn.Module:
        """Shard ``model`` with FSDP2 ``fully_shard``."""
        if not fsdp_config or not fsdp_config.get("enabled", True):
            return model
        fp8_linear = os.environ.get("LUMEN_FP8", "0") == "1"
        return _apply_fsdp2_sharding(model, fsdp_config, fp8_linear=fp8_linear)
