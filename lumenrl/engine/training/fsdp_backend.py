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

    # The Lumen/AITER attention patch replaces HF "sdpa" with an AITER varlen
    # flash-attn kernel. It targets packed (varlen) inputs; on plain padded
    # batches (e.g. the Ray controller path, which does not pack) its backward
    # can fault on ROCm. Allow opting out to fall back to pure torch SDPA,
    # matching the vanilla verl BF16 baseline.
    if os.environ.get("LUMEN_DISABLE_HF_ATTN_PATCH", "0") == "1":
        logger.info("[rank %d] HF attention patch disabled; using pure torch SDPA.", rank)
    else:
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
    # ATOM rollout op-alignment (verl FP8 recipe: LUMEN_ROLLOUT=ATOM). Patches
    # norm/sdpa/linear/mlp with AITER kernels so the training-side forward matches
    # the AITER-based vLLM rollout. Works in pure BF16 (no FP8 quant).
    lumen_rollout = quant_config.get("rollout") or os.environ.get("LUMEN_ROLLOUT", "")

    if not (fp8_enabled or fp8_pm or lumen_norm or fused_mlp or fused_rope or lumen_rollout):
        return model

    # Alignment mode: delegate the exact FP8 model modification to verl's
    # ``_maybe_apply_lumen`` (same LumenConfig.enable + lm_head BF16 restore),
    # so the native actor's FP8 training path is byte-identical to verl's
    # validated recipe. Enabled with LUMENRL_FP8_VIA_VERL=1 (needs verl importable
    # + MODEL_NAME env set). Keeps the native verl-free default for BF16/rollout.
    if os.environ.get("LUMENRL_FP8_VIA_VERL", "0") == "1":
        try:
            from verl.utils.fsdp_utils import _maybe_apply_lumen

            _maybe_apply_lumen(model, forward_only=False)
            logger.info("FP8 applied via verl _maybe_apply_lumen (aligned recipe).")
            return model
        except Exception as exc:
            logger.error("verl _maybe_apply_lumen delegation failed: %s", exc, exc_info=True)
            return model

    try:
        from lumen.config import LumenConfig

        # Respect LUMEN_DISABLE_HF_ATTN_PATCH: the native Ray path runs a validated
        # packed single-sequence pure-SDPA forward and installs its own packing
        # attention wrapper. Letting Lumen re-patch SDPA on top corrupts the
        # padded/packed attention layout (garbage logits -> exploding grad_norm /
        # rollout_corr/kl). Keep FP8 *linear* + norm while leaving attention to
        # the native forward unless the user explicitly opts into Lumen attention.
        hf_attn_patch = os.environ.get("LUMEN_DISABLE_HF_ATTN_PATCH", "0") != "1"
        kwargs = dict(
            fp8_param_manager=bool(fp8_pm),
            lumen_norm=bool(lumen_norm),
            fused_mlp=bool(fused_mlp),
            fused_rope=bool(fused_rope),
            hf_attn_patch=hf_attn_patch,
            fp8_weight_cache=quant_config.get("fp8_weight_cache", False),
        )
        if lumen_rollout:
            kwargs["rollout"] = lumen_rollout
        if fp8_enabled:
            # FP8 mode: enable quantized linear and related features. Attention
            # FP8 knobs (fp8_attn/attn_quant_type/attn_backend) are forwarded so
            # that ``LUMEN_FP8_ATTN=mha`` etc. take effect — matching verl's
            # ``_maybe_apply_lumen`` (per_block_fp8 recipe uses mha/blockwise).
            kwargs.update(
                scaling=os.environ.get("LUMEN_FP8_SCALING", "delayed"),
                format=os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3"),
                block_size=int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128")),
                fp8_activation_store=os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0") == "1",
                fp8_param_gather=os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1",
                fp8_attn=os.environ.get("LUMEN_FP8_ATTN", "none"),
                attn_quant_type=os.environ.get("LUMEN_FP8_QUANT_TYPE", "blockwise"),
                attn_backend=os.environ.get("LUMEN_ATTN_BACKEND", "auto"),
            )
        else:
            # BF16 mode: no FP8 quantized linear (scaling="none"); ATOM/norm/attn
            # patches still apply with AITER BF16 kernels.
            kwargs.update(
                scaling="none",
                quantize_activation=False,
                fp8_wgrad=False,
            )
        cfg = LumenConfig(**kwargs)
        _manager, model = cfg.enable(model)

        # verl-aligned (fsdp_utils._maybe_apply_lumen): restore lm_head to BF16.
        # The lm_head FP8 blockscale GEMM overflows INT32 on the huge vocab GEMM
        # (vocab ~152k x hidden 4096) -> garbage logits -> exploding grad_norm /
        # near-zero entropy / policy collapse. Keep every other Linear in FP8 but
        # run lm_head in BF16. Only relevant when FP8 quantized linear is active.
        if fp8_enabled:
            import torch.nn as nn

            restored = 0
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and "lm_head" in name:
                    if getattr(mod, "_quant_enabled", False):
                        mod._quant_enabled = False
                    mod.forward = nn.Linear.forward.__get__(mod, nn.Linear)
                    restored += 1
            if restored:
                logger.info("Restored %d lm_head Linear(s) to BF16 (CK blockscale INT32 overflow)", restored)

        logger.info(
            "Lumen optimizations applied (fp8=%s, fp8pm=%s, norm=%s, fused_mlp=%s, "
            "fused_rope=%s, rollout=%s)",
            fp8_enabled, fp8_pm, lumen_norm, fused_mlp, fused_rope, lumen_rollout or "(none)",
        )
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
    """Apply PyTorch FSDP2 to the model.

    Strategy is controlled by ``fsdp_config["strategy"]``:
      - ``"full_shard"`` (default): Full FSDP2 sharding on all layers + root.
      - ``"replicate"``: Pure DDP-like gradient all-reduce via composable replicate.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        logger.warning("torch.distributed not initialized; returning unsharded model.")
        return model

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_device = torch.device(f"cuda:{local_rank}")

    model.to(local_device)

    strategy = fsdp_config.get("strategy", "full_shard").lower()

    if strategy == "replicate":
        from torch.distributed._composable.replicate import replicate

        replicate(model, device_mesh=dist.init_device_mesh("cuda", (dist.get_world_size(),)))
        logger.info(
            "[rank %d] FSDP2 replicate applied (pure DDP, reduce_dtype=float32)",
            rank,
        )
        return model

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    param_dtype = _resolve_dtype(fsdp_config.get("param_dtype"), torch.bfloat16)
    reduce_dtype = _resolve_dtype(fsdp_config.get("reduce_dtype"), torch.float32)
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

    logger.info("[rank %d] FSDP2 full_shard applied (fp8_reduce=%s, offload=%s)",
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
