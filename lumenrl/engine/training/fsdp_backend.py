"""FSDP2 training backend helpers."""

from __future__ import annotations

import logging
from typing import Any

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


class FSDP2Backend:
    """Build a model and optionally wrap it with PyTorch FSDP2."""

    @staticmethod
    def build_model(model_name: str, config: dict[str, Any] | None) -> nn.Module:
        """Construct the trainable policy network."""
        del model_name  # resolved via config in full integrations
        cfg = config or {}
        arch = cfg.get("tiny_lm", {})
        vocab = int(arch.get("vocab_size", 32000))
        dim = int(arch.get("dim", 256))
        n_layers = int(arch.get("n_layers", 2))
        logger.info("FSDP2Backend.build_model: constructing TinyLM vocab=%d dim=%d", vocab, dim)
        return _TinyLM(vocab_size=vocab, dim=dim, n_layers=n_layers)

    @staticmethod
    def apply_lumen_optimizations(model: nn.Module, quant_config: dict[str, Any] | None) -> nn.Module:
        """Apply Lumen-specific fusion / quantization recipes where available."""
        if not quant_config:
            return model
        try:
            import lumen  # type: ignore[import-untyped]

            _ = lumen
            logger.info("FSDP2Backend.apply_lumen_optimizations: lumen present (no-op hook).")
        except ImportError:
            logger.warning(
                "FSDP2Backend.apply_lumen_optimizations: `lumen` not installed; skipping."
            )
        return model

    @staticmethod
    def apply_fsdp2(model: nn.Module, fsdp_config: dict[str, Any] | None) -> nn.Module:
        """Shard ``model`` with FSDP2 when torch.distributed.fsdp is available."""
        if not fsdp_config or not fsdp_config.get("enabled", True):
            return model
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            mesh = fsdp_config.get("device_mesh")
            if mesh is None:
                logger.warning(
                    "FSDP2Backend.apply_fsdp2: no device_mesh in fsdp_config; returning unsharded model."
                )
                return model
            wrapped = FSDP(model, device_mesh=mesh)
            logger.info("FSDP2Backend.apply_fsdp2: wrapped model with FSDP.")
            return wrapped
        except ImportError:
            logger.warning(
                "FSDP2Backend.apply_fsdp2: FSDP not available in this PyTorch build; "
                "returning dense module."
            )
            return model
        except Exception as exc:
            logger.warning("FSDP2Backend.apply_fsdp2 failed (%s); using dense module.", exc)
            return model
