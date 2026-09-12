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

    Each field replaces a question the engine used to answer by testing which
    model it was holding. The point is that a new architecture declares its
    answers here instead of adding another ``if`` at every call site.
    """

    # Weights load and weight-sync through the HF safetensors bridge. DSv4 ships
    # block-quantized FP8 the bridge cannot read, so it supplies no dims and takes
    # a dist-checkpoint path in both directions.
    supports_hf_bridge: bool = True
    # Several sequences may be bin-packed into one microbatch. DSv4 attention
    # derives token positions from the tensor length alone, so a packed microbatch
    # would read as one long sequence.
    supports_dynamic_batch: bool = True
    # The family supplies its own TransformerConfig and layer spec (``build_config``
    # / ``build_layer_spec``) rather than the generic TE block spec.
    builds_own_config: bool = False
    # Forward must always go through the pipeline path, even at PP=CP=1.
    requires_pipeline_forward: bool = False
    # The model reads a packed stream as a single sequence (ignores cu_seqlens),
    # which constrains sequence alignment and CP microbatch layout.
    packed_stream_is_single_sequence: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """How to recognise one model family and what the engine needs from it.

    The optional callables are hooks: ``None`` means "use the engine's generic
    path". A family that needs a bespoke construction supplies one instead of the
    engine growing a branch for it.
    """

    name: str
    detect: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    caps: ModelCaps = field(default_factory=ModelCaps)
    # Returns the dims dataclass the HF weight bridge consumes, or None for a
    # family that does not use that bridge.
    build_dims: Optional[Callable[[Mapping[str, Any]], Any]] = None
    # Routing conventions that belong to the architecture rather than to the run.
    # engine_config still overrides these; they are the family's default.
    routing_defaults: Mapping[str, Any] = field(default_factory=dict)
    # ``(hf, ec, **parallel) -> TransformerConfig`` for a family the generic
    # builder cannot describe.
    build_config: Optional[Callable[..., Any]] = None
    # ``(tfcfg, ec) -> ModuleSpec`` for a family whose layers are heterogeneous.
    build_layer_spec: Optional[Callable[..., Any]] = None
    # ``(tfcfg) -> int`` extra sequence-length alignment the family requires.
    sequence_alignment: Optional[Callable[[Any], int]] = None
    # ``(engine) -> Iterable[(name, tensor)]`` -- the rollout-ready weight stream.
    # Which parameter gather to use and how to rename on the way out are both
    # family decisions, so they travel together in one hook rather than as two
    # branches in the engine. The engine passes itself because the gathers are its
    # own helpers; specs live in the same package, so this coupling stays internal.
    export_weights: Optional[Callable[[Any], Any]] = None
    # Override the expert check for a family whose config does not state the count
    # in any spelling ``hf_num_experts`` knows. ``None`` uses the count, which is
    # what every current family wants.
    has_experts: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = None

    def resolve_has_experts(
        self, hf: Mapping[str, Any], ec: Mapping[str, Any]
    ) -> bool:
        """Whether this run has routed experts, honouring the config override.

        Deliberately keyed on the effective COUNT rather than on a per-family
        declaration: engine_config's ``num_experts`` can turn a dense checkpoint
        into a MoE run and back, so a static flag would be wrong half the time.
        This is byte-for-byte the condition the engine used before the registry.
        """
        if self.has_experts is not None:
            return bool(self.has_experts(hf, ec))
        return int(ec.get("num_experts") or hf_num_experts(hf) or 0) > 1


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
