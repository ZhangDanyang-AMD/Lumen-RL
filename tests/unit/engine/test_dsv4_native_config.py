"""build_dsv4_config, the native Megatron path's DeepSeek-V4 TransformerConfig."""

from __future__ import annotations

import pytest

from lumenrl.engine.training.bridges import dsv4

# The DeepSeek-V4-Flash config.json fields build_dsv4_config reads.
HF = {
    "attention_bias": False, "n_shared_experts": 1, "rms_norm_eps": 1e-6,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                     "original_max_position_embeddings": 65536, "type": "yarn"},
    "rope_theta": 10000, "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc", "compress_ratios": [0, 0, 4, 128], "compress_rope_theta": 160000,
    "hc_eps": 1e-6, "hc_mult": 4, "hc_sinkhorn_iters": 20, "head_dim": 512, "hidden_size": 4096,
    "index_head_dim": 128, "index_n_heads": 64, "index_topk": 512, "moe_intermediate_size": 2048,
    "n_routed_experts": 256, "num_attention_heads": 64, "num_experts_per_tok": 6,
    "num_hash_layers": 3, "num_hidden_layers": 4, "o_groups": 8, "o_lora_rank": 1024,
    "q_lora_rank": 1024, "qk_rope_head_dim": 64, "sliding_window": 128, "swiglu_limit": 10.0,
    "vocab_size": 129280,
}


def build(monkeypatch, ec):
    monkeypatch.setattr(dsv4, "_require_dsv4_megatron", lambda: dict)
    return dsv4.build_dsv4_config(HF, ec, tp=1, pp=1, cp=1, ep=8, etp=1, sp=False)


def test_an_uneven_pipeline_split_reaches_megatron(monkeypatch):
    config = build(monkeypatch, {"num_layers_in_first_pipeline_stage": 22})

    assert config["num_layers_in_first_pipeline_stage"] == 22
    assert config["num_layers_in_last_pipeline_stage"] is None


@pytest.mark.parametrize(("ec", "rate"), [({}, 0.0), ({"moe_router_bias_update_rate": 1e-3}, 1e-3)])
def test_the_expert_bias_stays_frozen_unless_asked(monkeypatch, ec, rate):
    config = build(monkeypatch, ec)

    assert config["moe_router_enable_expert_bias"] is True
    assert config["moe_router_bias_update_rate"] == rate
