"""RoPE and expert-group settings must come from the HF config, not from defaults.

``MLATransformerConfig``'s defaults happen to be DeepSeek-V3's YaRN values, which
makes a builder that ignores ``rope_scaling`` look correct on V3 and quietly
wrong everywhere else. One default is wrong even on V3: ``mscale_all_dim``.

Measured on the real ``bzantium/tiny-deepseek-v3`` checkpoint in fp32, leaving
these at Megatron's defaults gave 11.4% mean relative logit error and 71.9% top-1
agreement against the HF reference; reading them from the config gives 1.4e-6 and
100%. See ``test_dsv3_real_checkpoint.py``.
"""

import pytest

from lumenrl.engine.training import dsv3_megatron_bridge as dsv3

BASE = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 4, "hidden_size": 128, "num_attention_heads": 4,
    "num_key_value_heads": 4, "intermediate_size": 256, "vocab_size": 512,
    "rms_norm_eps": 1e-6, "q_lora_rank": 32, "kv_lora_rank": 16,
    "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
    "moe_intermediate_size": 64, "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
}

# Real rope_scaling blocks, verbatim from the published configs.
V3_ROPE = {
    "beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0,
    "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn",
}
K2_ROPE = {
    "beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0, "mscale": 1.0,
    "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn",
}


# --- rope ---------------------------------------------------------------------

def test_yarn_parameters_are_read_from_rope_scaling():
    cfg = dsv3.build_dsv3_config({**BASE, "rope_scaling": V3_ROPE, "rope_theta": 10000}, {})
    assert cfg.rope_type == "yarn"
    assert cfg.rotary_base == pytest.approx(10000.0)
    assert cfg.rotary_scaling_factor == pytest.approx(40.0)
    assert cfg.original_max_position_embeddings == 4096
    assert (cfg.beta_fast, cfg.beta_slow) == (pytest.approx(32.0), pytest.approx(1.0))


def test_mscale_all_dim_is_taken_from_the_config_not_the_default():
    """The one default that is wrong even for DeepSeek-V3.

    ``MLASelfAttention`` derives ``softmax_scale`` from it as
    ``_yarn_get_mscale(factor, mscale_all_dim)**2 / sqrt(qk_dim)`` -- and does so
    whatever ``rope_type`` is. Megatron defaults to 0.0, every shipped DeepSeek
    config says 1.0, and at factor 40 that is a 1.87x error in the attention
    temperature.
    """
    cfg = dsv3.build_dsv3_config({**BASE, "rope_scaling": V3_ROPE}, {})
    assert cfg.mscale_all_dim == pytest.approx(1.0)


def test_kimi_k2_does_not_inherit_deepseek_v3s_rope():
    """K2 differs from the defaults on three values at once."""
    cfg = dsv3.build_dsv3_config(
        {**BASE, "rope_scaling": K2_ROPE, "rope_theta": 50000.0}, {}
    )
    assert cfg.rotary_base == pytest.approx(50000.0)     # default 10000
    assert cfg.rotary_scaling_factor == pytest.approx(32.0)  # default 40
    assert cfg.beta_fast == pytest.approx(1.0)           # default 32


def test_no_rope_scaling_means_plain_rope_and_a_unit_softmax_scale():
    """Without scaling, the default 'yarn' would apply a factor-40 stretch."""
    cfg = dsv3.build_dsv3_config(BASE, {})
    assert cfg.rope_type == "rope"
    assert cfg.rotary_scaling_factor == pytest.approx(1.0)
    assert cfg.mscale_all_dim == pytest.approx(0.0)


def test_unsupported_rope_scaling_is_refused():
    """MLA implements 'rope' and 'yarn'; anything else would be silently ignored."""
    with pytest.raises(ValueError, match="rope_scaling"):
        dsv3.build_dsv3_config({**BASE, "rope_scaling": {"type": "linear", "factor": 4}}, {})


# --- node-limited expert routing ----------------------------------------------

def test_expert_groups_are_wired_from_n_group_and_topk_group():
    """V3/R1 restrict top-k to 4 of 8 groups; ignoring that changes the experts."""
    cfg = dsv3.build_dsv3_config({**BASE, "n_group": 8, "topk_group": 4}, {})
    assert cfg.moe_router_num_groups == 8
    assert cfg.moe_router_group_topk == 4


def test_a_single_expert_group_is_left_unset():
    """K2 ships n_group=1, which constrains nothing and which Megatron rejects."""
    cfg = dsv3.build_dsv3_config({**BASE, "n_group": 1, "topk_group": 1}, {})
    assert cfg.moe_router_num_groups is None
    assert cfg.moe_router_group_topk is None


def test_engine_config_can_override_the_groups():
    cfg = dsv3.build_dsv3_config(
        {**BASE, "n_group": 8, "topk_group": 4},
        {"moe_router_num_groups": 4, "moe_router_group_topk": 2},
    )
    assert (cfg.moe_router_num_groups, cfg.moe_router_group_topk) == (4, 2)


def test_norm_topk_prob_false_is_refused():
    """Megatron's sigmoid router renormalises unconditionally, so there is no
    way to honour norm_topk_prob=False -- refuse rather than rescale silently."""
    with pytest.raises(ValueError, match="norm_topk_prob"):
        dsv3.build_dsv3_config({**BASE, "norm_topk_prob": False}, {})
