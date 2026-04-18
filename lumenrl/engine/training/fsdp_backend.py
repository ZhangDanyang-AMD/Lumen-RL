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


def _load_hf_model(model_name: str, torch_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Load a HuggingFace causal LM with gradient checkpointing.

    In distributed mode, only rank 0 loads from disk. Other ranks load
    with empty weights (meta device), then rely on FSDP2's broadcast
    to receive the actual parameters.
    """
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoConfig

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if is_distributed and rank != 0:
        logger.info("[rank %d] Loading HF model config only (rank 0 has weights): %s", rank, model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
    else:
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
    """Apply Lumen FP8 optimizations to the model before FSDP2 sharding.

    This replicates what ``lumen.rl.verl.verl_entry.patch_verl_fsdp_workers``
    did via monkey-patching, but called directly without any VERL dependency.
    """
    if not quant_config:
        return model

    fp8_enabled = quant_config.get("fp8") or os.environ.get("LUMEN_FP8", "0") == "1"
    fp8_pm = quant_config.get("fp8_param_manager") or os.environ.get("FP8_PARAM_MANAGER", "0") == "1"
    lumen_norm = quant_config.get("lumen_norm") or os.environ.get("LUMEN_NORM", "0") == "1"

    if not (fp8_enabled or fp8_pm or lumen_norm):
        return model

    try:
        from lumen.config import LumenConfig

        cfg = LumenConfig(
            linear_fp8=bool(fp8_enabled),
            fp8_param_manager=bool(fp8_pm),
            lumen_norm=bool(lumen_norm),
            scaling=os.environ.get("LUMEN_FP8_SCALING", "delayed"),
            format=os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3"),
            block_size=int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128")),
            fp8_activation_store=os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0") == "1",
            fp8_param_gather=os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1",
            fp8_weight_cache=quant_config.get("fp8_weight_cache", False),
        )
        _manager, model = cfg.enable(model)
        logger.info("Lumen FP8 optimizations applied (fp8=%s, fp8pm=%s, norm=%s)",
                     fp8_enabled, fp8_pm, lumen_norm)
    except ImportError:
        logger.warning("lumen package not installed; skipping FP8 optimizations.")
    except Exception as exc:
        logger.error("Lumen FP8 enable() failed: %s", exc, exc_info=True)

    return model


def _apply_fsdp2_sharding(
    model: nn.Module,
    fsdp_config: dict[str, Any],
    fp8_linear: bool = False,
) -> nn.Module:
    """Apply PyTorch FSDP2 ``fully_shard`` to the model.

    Non-rank-0 processes may have meta-device parameters. FSDP2's
    ``fully_shard`` will materialize them during the first all-gather.
    We provide a ``param_init_fn`` that allocates empty tensors on the
    local device so FSDP2 can then populate them from rank 0.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        logger.warning("torch.distributed not initialized; returning unsharded model.")
        return model

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_device = torch.device(f"cuda:{local_rank}")

    has_meta = any(p.device.type == "meta" for p in model.parameters())
    if has_meta:
        logger.info("[rank %d] Materializing meta parameters on %s", rank, local_device)
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                materialized = torch.empty(
                    param.shape, dtype=param.dtype, device=local_device,
                )
                param_module_name = ".".join(name.split(".")[:-1])
                param_name = name.split(".")[-1]
                mod = model
                for part in param_module_name.split("."):
                    if part:
                        mod = getattr(mod, part)
                setattr(mod, param_name, nn.Parameter(materialized, requires_grad=param.requires_grad))

        for name, buf in model.named_buffers():
            if buf.device.type == "meta":
                materialized = torch.empty(buf.shape, dtype=buf.dtype, device=local_device)
                buf_module_name = ".".join(name.split(".")[:-1])
                buf_name = name.split(".")[-1]
                mod = model
                for part in buf_module_name.split("."):
                    if part:
                        mod = getattr(mod, part)
                mod.register_buffer(buf_name, materialized)
    elif rank == 0:
        model.to(local_device)

    if fp8_linear:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
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

        if cfg.get("use_tiny_lm", False) or not model_name:
            arch = cfg.get("tiny_lm", {})
            vocab = int(arch.get("vocab_size", 32000))
            dim = int(arch.get("dim", 256))
            n_layers = int(arch.get("n_layers", 2))
            logger.info("FSDP2Backend.build_model: TinyLM (vocab=%d, dim=%d)", vocab, dim)
            return _TinyLM(vocab_size=vocab, dim=dim, n_layers=n_layers)

        return _load_hf_model(model_name, torch_dtype=torch.bfloat16)

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
