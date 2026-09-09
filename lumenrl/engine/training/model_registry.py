"""Architecture registry for the Megatron training path.

``megatron_native_engine`` used to answer "which model is this?" with two
booleans -- ``_is_dsv4`` and ``_is_moe`` -- computed inline from the HF config.
That works for exactly the three families already in tree (DSv4, Qwen3-MoE,
Qwen3 dense) and cannot express a fourth: every new architecture needs another
boolean, another arm on every branch, and another pair of imports at the top of
the engine.

This module turns that into data. One :class:`ModelSpec` per family says how to
recognise it, how to derive its dims, and which routing conventions it needs;
the engine asks the registry instead of asking itself.

Resolution is ordered -- the first spec whose ``detect`` returns True wins -- so
the more specific families are registered before the general ones, and a
catch-all dense entry is last. Detection takes the engine config as well as the
HF config because ``num_experts`` may be overridden there, which can turn an
otherwise-dense config into a MoE run.

Adding an architecture should mean adding a :class:`ModelSpec`, not editing the
engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "ModelCaps",
    "ModelSpec",
    "ModelRegistry",
    "MODEL_REGISTRY",
    "hf_num_experts",
    "resolve_head_dim",
]


def hf_num_experts(hf: Mapping[str, Any]) -> int:
    """Routed-expert count declared by an HF config, across the spellings in use.

    Qwen3-MoE uses ``num_experts``, DeepSeek ``n_routed_experts``, Mixtral-style
    configs ``num_local_experts``. Returns 0 for a dense config.
    """
    return int(
        hf.get("num_experts")
        or hf.get("n_routed_experts")
        or hf.get("num_local_experts")
        or 0
    )


def resolve_head_dim(hf: Mapping[str, Any]) -> int:
    """``head_dim`` if the config states it, else hidden / heads."""
    return int(hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]))


@dataclass(frozen=True)
class ModelCaps:
    """What the engine may do with this family.

    These replace questions the engine currently asks about itself. ``has_experts``
    is the old ``_is_moe``; the other two record why DSv4 takes different paths
    rather than leaving that knowledge in an ``if`` at the call site.
    """

    has_experts: bool = False
    # DSv4 ships block-quantized FP8 weights the HF-safetensors bridge cannot read,
    # so it supplies no dims and neither loads nor weight-syncs through that path.
    supports_hf_bridge: bool = True
    # DSv4 attention derives token positions from the tensor length alone, so
    # bin-packing several sequences into one microbatch reads as one long sequence.
    supports_dynamic_batch: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """How to recognise one model family and what the engine needs from it."""

    name: str
    detect: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    caps: ModelCaps = field(default_factory=ModelCaps)
    # Returns the dims dataclass the HF weight bridge consumes, or None for a
    # family that does not use that bridge.
    build_dims: Optional[Callable[[Mapping[str, Any]], Any]] = None
    # Routing conventions that belong to the architecture rather than to the run.
    # engine_config still overrides these; they are the family's default.
    routing_defaults: Mapping[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Ordered first-match registry of :class:`ModelSpec`."""

    def __init__(self) -> None:
        self._specs: list[ModelSpec] = []

    def register(self, spec: ModelSpec) -> ModelSpec:
        """Append a spec. Order is priority: register specific before general."""
        if any(s.name == spec.name for s in self._specs):
            raise ValueError(f"ModelSpec {spec.name!r} is already registered")
        self._specs.append(spec)
        return spec

    def resolve(
        self, hf: Mapping[str, Any], engine_config: Optional[Mapping[str, Any]] = None
    ) -> ModelSpec:
        """First spec whose ``detect`` accepts this config.

        Raises if nothing matches, which can only happen if the catch-all entry
        was removed -- better a loud error here than a silently wrong dense path.
        """
        ec = engine_config or {}
        for spec in self._specs:
            if spec.detect(hf, ec):
                return spec
        arch = hf.get("architectures") or hf.get("model_type") or "<unknown>"
        raise LookupError(
            f"No ModelSpec matches architecture {arch!r}. "
            f"Registered: {[s.name for s in self._specs]}"
        )

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]


MODEL_REGISTRY = ModelRegistry()
