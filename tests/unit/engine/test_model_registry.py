"""Resolution behaviour of the architecture registry.

These pin the properties ``megatron_native_engine`` relies on: order is
priority, the engine_config override participates in detection, and the three
in-tree families resolve to the specs that reproduce the pre-registry branches.
"""

import pytest

from lumenrl.engine.training import model_specs  # noqa: F401  (registers specs)
from lumenrl.engine.training.model_registry import (
    MODEL_REGISTRY,
    ModelCaps,
    ModelRegistry,
    ModelSpec,
    hf_num_experts,
    resolve_head_dim,
)

QWEN3_DENSE_CFG = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 12288,
    "vocab_size": 151936,
}

QWEN3_MOE_CFG = {
    **QWEN3_DENSE_CFG,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}


# --- helpers -----------------------------------------------------------------

@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({}, 0),
        ({"num_experts": 128}, 128),
        ({"n_routed_experts": 256}, 256),      # DeepSeek spelling
        ({"num_local_experts": 8}, 8),         # Mixtral spelling
    ],
)
def test_hf_num_experts_handles_every_spelling(cfg, expected):
    assert hf_num_experts(cfg) == expected


def test_resolve_head_dim_prefers_explicit():
    assert resolve_head_dim({**QWEN3_DENSE_CFG, "head_dim": 128}) == 128


def test_resolve_head_dim_falls_back_to_hidden_over_heads():
    assert resolve_head_dim(QWEN3_DENSE_CFG) == 4096 // 32


# --- resolution --------------------------------------------------------------

def test_dense_config_resolves_to_dense_spec():
    spec = MODEL_REGISTRY.resolve(QWEN3_DENSE_CFG, {})
    assert spec.name == "qwen3_dense"
    assert spec.caps.has_experts is False


def test_moe_config_resolves_to_moe_spec():
    spec = MODEL_REGISTRY.resolve(QWEN3_MOE_CFG, {})
    assert spec.name == "qwen3_moe"
    assert spec.caps.has_experts is True


def test_engine_config_num_experts_override_promotes_dense_to_moe():
    """The pre-registry engine let engine_config override the HF expert count."""
    spec = MODEL_REGISTRY.resolve(QWEN3_DENSE_CFG, {"num_experts": 64})
    assert spec.name == "qwen3_moe"


def test_single_expert_is_not_moe():
    """``_is_moe`` was ``num_experts > 1``; one expert stays dense."""
    spec = MODEL_REGISTRY.resolve({**QWEN3_DENSE_CFG, "num_experts": 1}, {})
    assert spec.name == "qwen3_dense"


def test_dense_spec_builds_dims_matching_the_old_inline_construction():
    spec = MODEL_REGISTRY.resolve(QWEN3_DENSE_CFG, {})
    dims = spec.build_dims(QWEN3_DENSE_CFG)
    assert (dims.num_layers, dims.hidden, dims.num_heads) == (36, 4096, 32)
    assert (dims.num_kv_groups, dims.head_dim) == (8, 128)
    assert (dims.ffn, dims.vocab) == (12288, 151936)


def test_moe_routing_default_is_pre_softmax_false():
    """Qwen3-MoE's norm_topk_prob maps to Megatron pre_softmax=False."""
    spec = MODEL_REGISTRY.resolve(QWEN3_MOE_CFG, {})
    assert spec.routing_defaults["moe_router_pre_softmax"] is False


def test_dense_spec_declares_no_routing_defaults():
    spec = MODEL_REGISTRY.resolve(QWEN3_DENSE_CFG, {})
    assert spec.routing_defaults == {}


# --- registry mechanics ------------------------------------------------------

def test_order_is_priority_first_match_wins():
    reg = ModelRegistry()
    reg.register(ModelSpec(name="specific", detect=lambda hf, ec: "flag" in hf))
    reg.register(ModelSpec(name="catchall", detect=lambda hf, ec: True))
    assert reg.resolve({"flag": 1}).name == "specific"
    assert reg.resolve({}).name == "catchall"


def test_duplicate_registration_is_rejected():
    reg = ModelRegistry()
    reg.register(ModelSpec(name="dup", detect=lambda hf, ec: True))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ModelSpec(name="dup", detect=lambda hf, ec: True))


def test_no_match_raises_rather_than_defaulting():
    """A missing catch-all must fail loudly, not silently take a dense path."""
    reg = ModelRegistry()
    reg.register(ModelSpec(name="never", detect=lambda hf, ec: False))
    with pytest.raises(LookupError, match="No ModelSpec matches"):
        reg.resolve({"architectures": ["MysteryForCausalLM"]})


def test_registered_families_are_ordered_specific_before_general():
    """DSv4 and MoE must both precede the dense catch-all."""
    names = MODEL_REGISTRY.names
    assert names[-1] == "qwen3_dense"
    assert names.index("deepseek_v4") < names.index("qwen3_moe")


def test_caps_defaults_are_the_permissive_ones():
    caps = ModelCaps()
    assert caps.supports_hf_bridge is True
    assert caps.supports_dynamic_batch is True
    assert caps.has_experts is False
