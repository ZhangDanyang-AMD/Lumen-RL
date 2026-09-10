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


def _all_specs():
    """The registered specs, by identity rather than by name lookup."""
    return [MODEL_REGISTRY.resolve(cfg, ec) for cfg, ec in (
        ({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}, {}),
        (QWEN3_MOE_CFG, {}),
        (QWEN3_DENSE_CFG, {}),
    )]


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


# --- capabilities and hooks (phase 2) ----------------------------------------

def test_dsv4_declares_the_capabilities_the_engine_used_to_hard_code():
    """Each of these replaced an ``if self._is_dsv4`` at a call site."""
    spec = next(s for s in _all_specs() if s.name == "deepseek_v4")
    caps = spec.caps
    assert caps.supports_hf_bridge is False          # dist-checkpoint load + sync
    assert caps.supports_dynamic_batch is False      # attention ignores cu_seqlens
    assert caps.builds_own_config is True            # heterogeneous layers
    assert caps.requires_pipeline_forward is True    # topology check before forward
    assert caps.packed_stream_is_single_sequence is True


def test_generic_families_take_the_generic_paths():
    for name in ("qwen3_moe", "qwen3_dense"):
        spec = next(s for s in _all_specs() if s.name == name)
        assert spec.caps.supports_hf_bridge is True
        assert spec.caps.supports_dynamic_batch is True
        assert spec.caps.builds_own_config is False
        assert spec.caps.requires_pipeline_forward is False
        assert spec.build_config is None, f"{name} must use the generic builder"
        assert spec.build_layer_spec is None
        assert spec.sequence_alignment is None


def test_dsv4_supplies_construction_hooks():
    spec = next(s for s in _all_specs() if s.name == "deepseek_v4")
    assert spec.build_config is not None
    assert spec.build_layer_spec is not None
    assert spec.sequence_alignment is not None


def test_resolve_has_experts_honours_the_config_override():
    """Replaces the engine's inline ``num_experts > 1``."""
    dense = MODEL_REGISTRY.resolve(QWEN3_DENSE_CFG, {})
    assert dense.resolve_has_experts(QWEN3_DENSE_CFG, {}) is False
    assert dense.resolve_has_experts(QWEN3_DENSE_CFG, {"num_experts": 8}) is True

    moe = MODEL_REGISTRY.resolve(QWEN3_MOE_CFG, {})
    assert moe.resolve_has_experts(QWEN3_MOE_CFG, {}) is True
    assert moe.resolve_has_experts(QWEN3_MOE_CFG, {"num_experts": 1}) is False


def test_caps_defaults_stay_permissive_for_new_families():
    """A new ModelSpec must opt in to special paths, never inherit them."""
    caps = ModelCaps()
    assert caps.builds_own_config is False
    assert caps.requires_pipeline_forward is False
    assert caps.packed_stream_is_single_sequence is False


# --- weight export hook (phase 3) --------------------------------------------

def test_every_family_supplies_a_weight_exporter():
    """The engine calls ``spec.export_weights(self)`` unconditionally now.

    A spec without one would raise TypeError at the first rollout weight sync,
    which is far from where the mistake was made.
    """
    for spec in _all_specs():
        assert spec.export_weights is not None, f"{spec.name} has no exporter"
        assert callable(spec.export_weights)


def test_exporters_are_distinct_per_family():
    """DSv4 renames to checkpoint names, MoE and dense use different gathers."""
    exporters = {s.name: s.export_weights for s in _all_specs()}
    assert len({id(f) for f in exporters.values()}) == 3


def test_dense_exporter_uses_the_plain_gather_and_te_naming(monkeypatch):
    """Pin the wiring without needing real tensors: patch the bridge, check inputs."""
    from lumenrl.engine.training import model_specs as ms

    seen = {}

    def fake_megatron_to_hf(named, dims, te=False):
        seen["gather"] = named
        seen["te"] = te
        return iter(())

    monkeypatch.setattr(ms, "megatron_to_hf", fake_megatron_to_hf)

    sentinel = object()

    class FakeEngine:
        _dims = None

        def _full_megatron_named_params(self):
            return sentinel

        def _full_megatron_named_params_moe(self):
            raise AssertionError("dense must not use the MoE gather")

    list(ms._export_dense(FakeEngine()))
    assert seen["gather"] is sentinel
    assert seen["te"] is True


def test_moe_exporter_uses_the_moe_gather(monkeypatch):
    from lumenrl.engine.training import model_specs as ms

    seen = {}

    def fake_megatron_to_hf_moe(named, dims):
        seen["gather"] = named
        return iter(())

    monkeypatch.setattr(ms, "megatron_to_hf_moe", fake_megatron_to_hf_moe)

    sentinel = object()

    class FakeEngine:
        _dims = None

        def _full_megatron_named_params(self):
            raise AssertionError("MoE must not use the plain gather")

        def _full_megatron_named_params_moe(self):
            return sentinel

    list(ms._export_moe(FakeEngine()))
    assert seen["gather"] is sentinel
