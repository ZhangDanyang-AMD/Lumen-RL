"""FP8 rollout and training configuration derived from LumenRL quant settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from lumenrl.core.config import QuantizationConfig

logger = logging.getLogger(__name__)

RecipeKind = Literal["blockwise", "tensorwise"]


@dataclass(frozen=True)
class FP8Config:
    """Unified FP8 knobs for rollout, KV cache, and training paths."""

    precision: str = "bf16"
    recipe: RecipeKind = "blockwise"
    use_deep_gemm: bool = True
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0
    use_weight_pow2_scale: bool = False
    use_activation_pow2_scale: bool = False

    def is_fp8_enabled(self) -> bool:
        """Return True when FP8 numerics are active for rollout or training."""
        p = self.precision.lower().strip()
        if p in {"fp8", "float8", "e4m3", "e5m2", "hybrid_fp8"}:
            return True
        return "fp8" in p

    @classmethod
    def from_config(cls, quant_config: QuantizationConfig) -> FP8Config:
        """Build an :class:`FP8Config` from a structured :class:`QuantizationConfig`."""
        if not isinstance(quant_config, QuantizationConfig):
            raise TypeError(f"quant_config must be QuantizationConfig, got {type(quant_config)!r}")

        rollout = quant_config.rollout
        training = quant_config.training

        precision = rollout.precision
        if training.fp8:
            precision = training.fp8

        recipe: RecipeKind = "blockwise"
        if training.fp8_recipe in ("blockwise", "tensorwise"):
            recipe = training.fp8_recipe  # type: ignore[assignment]

        cfg = cls(
            precision=precision,
            recipe=recipe,
            use_deep_gemm=rollout.use_deep_gemm,
            num_first_layers_in_bf16=rollout.num_first_layers_in_bf16,
            num_last_layers_in_bf16=rollout.num_last_layers_in_bf16,
            use_weight_pow2_scale=False,
            use_activation_pow2_scale=False,
        )
        logger.debug("Built FP8Config: %s", cfg)
        return cfg
