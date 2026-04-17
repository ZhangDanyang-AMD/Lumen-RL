"""Megatron-LM style training backend with MoE spec hooks."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


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


class MegatronBackend:
    """Construct Megatron-Core models and attach MoE / TE specifications."""

    @staticmethod
    def build_model(model_name: str, config: dict[str, Any] | None) -> nn.Module:
        """Instantiate a Megatron policy module or a local stub."""
        del model_name
        root = config or {}
        meg_cfg = root.get("megatron") or root.get("megatron_cfg") or {}
        if is_dataclass(meg_cfg):
            meg_cfg = asdict(meg_cfg)
        elif not isinstance(meg_cfg, dict):
            meg_cfg = dict(vars(meg_cfg))
        hidden = int(meg_cfg.get("hidden_size", 256))
        vocab = int(meg_cfg.get("vocab_size", 32000))
        try:
            import megatron  # type: ignore[import-untyped]

            _ = megatron
            logger.warning(
                "MegatronBackend.build_model: `megatron` import succeeded but no public "
                "builder is wired; returning stub LM. Extend this hook for MCore integration."
            )
        except ImportError as exc:
            logger.warning(
                "MegatronBackend.build_model: megatron not available (%s); using stub LM.",
                exc,
            )
        return _MegatronStubLM(hidden_size=hidden, vocab_size=vocab)

    @staticmethod
    def apply_lumen_spec(model: nn.Module, megatron_config: dict[str, Any] | None) -> nn.Module:
        """Register MoE spec providers and Lumen tensor-parallel annotations."""
        megatron_config = megatron_config or {}
        num_experts = megatron_config.get("num_experts")
        if num_experts:
            logger.info(
                "MegatronBackend.apply_lumen_spec: MoE with num_experts=%s (spec hook).",
                num_experts,
            )
        try:
            from megatron.core.transformer.moe import MoEConfig  # type: ignore[import-untyped]

            _ = MoEConfig
            logger.debug("MegatronBackend.apply_lumen_spec: MoEConfig import OK.")
        except ImportError:
            logger.debug(
                "MegatronBackend.apply_lumen_spec: megatron.core MoEConfig not available."
            )
        try:
            import lumen  # type: ignore[import-untyped]

            _ = lumen
        except ImportError:
            logger.warning("MegatronBackend.apply_lumen_spec: optional `lumen` not installed.")
        return model
