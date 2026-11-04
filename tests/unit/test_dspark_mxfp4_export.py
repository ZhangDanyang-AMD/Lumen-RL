from __future__ import annotations

import torch


def test_quantize_model_state_preserves_bf16_and_adds_e8m0_scale(
    load_k3_example_module,
) -> None:
    module = load_k3_example_module("scripts/export_mxfp4.py")
    assert (
        module.scale_key_for("layers.0.self_attn.q_b_proj.weight")
        == "layers.0.self_attn.q_b_proj.weight_scale"
    )
    selected = "layers.0.mlp.down_proj.weight"
    state = {
        selected: torch.randn(32, 64, dtype=torch.bfloat16),
        "layers.0.self_attn.q_a_proj.weight": torch.randn(
            32, 64, dtype=torch.bfloat16
        ),
    }

    def fake_quantize(weight: torch.Tensor):
        return (
            torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
            torch.ones(weight.shape[0], weight.shape[1] // 32, dtype=torch.uint8),
        )

    output, report, hits = module.quantize_model_state(
        state,
        ("layers.*.mlp.down_proj.weight",),
        fake_quantize,
    )

    assert output[selected].shape == (32, 32)
    assert output[selected].dtype == torch.uint8
    scale = output["layers.0.mlp.down_proj.weight_scale"]
    assert scale.shape == (32, 2)
    assert scale.dtype == torch.uint8
    assert output["layers.0.self_attn.q_a_proj.weight"].dtype == torch.bfloat16
    assert report[0]["logical_shape"] == [32, 64]
    assert hits == {"layers.*.mlp.down_proj.weight": 1}


def test_atom_config_selects_mxfp4_and_retains_a_projection_in_bf16(
    load_k3_example_module,
) -> None:
    module = load_k3_example_module("scripts/export_mxfp4.py")
    config = module.atom_quantization_config()

    assert config["quant_method"] == "quark"
    assert (
        config["global_quant_config"]["weight"]["dtype"]
        == "fp4_e2m1"
    )
    assert "context_proj" in config["exclude"]
    assert "layers.*.self_attn.fused_qkv_a_proj" in config["exclude"]
