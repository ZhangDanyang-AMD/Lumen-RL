"""DeepSeek-V3 family registry entry.

Values are the real ``moonshotai/Kimi-K2-Base`` config, which reports
``DeepseekV3ForCausalLM`` -- one entry serves DeepSeek V3 / V3.1 / R1 and the
whole Kimi K2 line.
"""

import pytest

from lumenrl.engine.training import dsv3_megatron_bridge as dsv3
from lumenrl.engine.training import model_specs  # noqa: F401  (registers specs)
from lumenrl.engine.training.model_registry import MODEL_REGISTRY

# Real Kimi-K2-Base values.
K2 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 61,
    "hidden_size": 7168,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "intermediate_size": 18432,
    "vocab_size": 163840,
    "rms_norm_eps": 1e-06,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.827,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
}


# --- detection ----------------------------------------------------------------

def test_kimi_k2_resolves_to_the_deepseek_v3_entry():
    assert MODEL_REGISTRY.resolve(K2, {}).name == "deepseek_v3"


def test_dsv3_is_matched_before_the_generic_moe_entry():
    """A DSv3 config declares routed experts, so ordering is what saves it.

    If qwen3_moe claimed it, the model would be built with a plain
    TransformerConfig and fused QKV, which is wrong for MLA.
    """
    names = MODEL_REGISTRY.names
    assert names.index("deepseek_v3") < names.index("qwen3_moe")


def test_detection_is_by_architecture_not_by_mla_fields():
    """DSv4 also has kv_lora_rank; keying on that would misclaim it."""
    assert dsv3.is_dsv3(K2) is True
    assert dsv3.is_dsv3({**K2, "architectures": ["DeepseekV4ForCausalLM"]}) is False
    assert dsv3.is_dsv3({"kv_lora_rank": 512}) is False


# --- dims ---------------------------------------------------------------------

def test_dims_carry_the_mla_geometry():
    d = MODEL_REGISTRY.resolve(K2, {}).build_dims(K2)
    assert (d.num_layers, d.hidden, d.num_heads) == (61, 7168, 64)
    assert (d.q_lora_rank, d.kv_lora_rank) == (1536, 512)
    assert (d.qk_nope_head_dim, d.qk_rope_head_dim, d.v_head_dim) == (128, 64, 128)


def test_dims_carry_the_moe_geometry():
    d = dsv3.build_dsv3_dims(K2)
    assert (d.num_experts, d.moe_topk, d.n_shared_experts) == (384, 8, 1)
    assert d.moe_ffn == 2048
    assert d.shared_expert_ffn == 2048  # moe_ffn * n_shared
    assert d.first_k_dense_replace == 1
    assert d.routed_scaling_factor == pytest.approx(2.827)


# --- config -------------------------------------------------------------------

def test_config_is_an_mla_config_with_the_lora_ranks():
    cfg = dsv3.build_dsv3_config(K2, {})
    assert cfg.multi_latent_attention is True
    assert (cfg.q_lora_rank, cfg.kv_lora_rank) == (1536, 512)
    assert (cfg.qk_head_dim, cfg.qk_pos_emb_head_dim, cfg.v_head_dim) == (128, 64, 128)


def test_config_uses_deepseek_routing_not_qwen_routing():
    """sigmoid + noaux_tc bias + scaling factor; softmax would pick other experts."""
    cfg = dsv3.build_dsv3_config(K2, {})
    assert cfg.moe_router_score_function == "sigmoid"
    assert cfg.moe_router_enable_expert_bias is True
    assert cfg.moe_router_topk_scaling_factor == pytest.approx(2.827)
    assert cfg.moe_router_pre_softmax is False


def test_first_k_dense_replace_becomes_a_per_layer_pattern():
    cfg = dsv3.build_dsv3_config(K2, {})
    freq = cfg.moe_layer_freq
    assert len(freq) == 61
    assert freq[0] == 0 and set(freq[1:]) == {1}


def test_shared_expert_size_is_scaled_by_count():
    cfg = dsv3.build_dsv3_config(K2, {})
    assert cfg.moe_shared_expert_intermediate_size == 2048


def test_parallelism_and_dtypes_match_the_generic_path():
    cfg = dsv3.build_dsv3_config(K2, {}, tp=2, pp=4, cp=1, ep=8, etp=2, sp=True)
    assert cfg.tensor_model_parallel_size == 2
    assert cfg.pipeline_model_parallel_size == 4
    assert cfg.expert_model_parallel_size == 8
    assert cfg.expert_tensor_parallel_size == 2
    assert cfg.sequence_parallel is True
    # PP>1 needs dynamic P2P shapes for variable-length RL microbatches.
    assert cfg.variable_seq_lengths is True
    assert cfg.use_cpu_initialization is True


def test_recompute_is_forwarded():
    cfg = dsv3.build_dsv3_config(
        K2, {"recompute_granularity": "full", "recompute_num_layers": 2}
    )
    assert cfg.recompute_granularity == "full"
    assert cfg.recompute_method == "uniform"
    assert cfg.recompute_num_layers == 2


def test_engine_config_overrides_win_over_hf():
    cfg = dsv3.build_dsv3_config(
        K2, {"moe_router_topk": 4, "moe_grouped_gemm": False, "moe_aux_loss_coeff": 0.01}
    )
    assert cfg.moe_router_topk == 4
    assert cfg.moe_grouped_gemm is False
    assert cfg.moe_aux_loss_coeff == pytest.approx(0.01)


# --- weight export -------------------------------------------------------------

def test_export_uses_the_mla_bridge_and_includes_the_router_bias():
    """DSv3 must not fall back to the Qwen3 exporter.

    Qwen3 names do not exist on an MLA model, so that fallback would leave the
    rollout on a partly-uninitialised policy. The router bias is a buffer and has
    to be chained in explicitly; without it, noaux_tc selects different experts
    on the rollout side than the trainer used.
    """
    spec = MODEL_REGISTRY.resolve(K2, {})
    seen = {}

    class FakeEngine:
        module = object()
        _dims = None

        def _full_megatron_named_params_moe(self):
            seen["gather"] = "moe"
            return iter(())

        def _full_megatron_named_params(self):
            raise AssertionError("DSv3 must use the MoE gather")

    import lumenrl.engine.training.model_specs as ms

    orig = dsv3.dsv3_router_bias_buffers
    try:
        dsv3.dsv3_router_bias_buffers = lambda mod: iter(())  # type: ignore[assignment]
        ms.dsv3.dsv3_router_bias_buffers = dsv3.dsv3_router_bias_buffers
        list(spec.export_weights(FakeEngine()))
    finally:
        dsv3.dsv3_router_bias_buffers = orig  # type: ignore[assignment]
        ms.dsv3.dsv3_router_bias_buffers = orig

    assert seen["gather"] == "moe"


def test_dsv3_uses_the_stock_te_layer_spec():
    """MLA is a parameter of the stock builder, so no custom spec is needed."""
    assert MODEL_REGISTRY.resolve(K2, {}).build_layer_spec is None
