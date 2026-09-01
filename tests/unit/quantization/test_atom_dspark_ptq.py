from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lumenrl.quantization.atom_dspark_native import (
    export_atom_native_dspark_checkpoint,
)
from lumenrl.quantization.atom_dspark_ptq import (
    build_mixed_quark_quant_config,
    build_quark_quant_config,
    build_quark_ptpc_config,
    compose_mixed_dspark_checkpoint,
    convert_dspark_checkpoint,
    expected_dspark_tensor_names,
    expected_quantized_weight_names,
    get_ptq_profile,
    select_mxfp4_weight_names,
    validate_dspark_checkpoint,
    weight_scale_name,
)


def _config() -> dict:
    return {
        "architectures": ["K3DSparkModel"],
        "num_hidden_layers": 5,
        "torch_dtype": "bfloat16",
    }


def _tiny_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in expected_dspark_tensor_names(5):
        if name.endswith(".bias") or "norm.weight" in name:
            tensors[name] = torch.randn(3, dtype=torch.bfloat16)
        else:
            tensors[name] = torch.randn(3, 32, dtype=torch.bfloat16)
    path.mkdir()
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(_config()))
    (path / "README.md").write_text("fixture\n")
    return tensors


def _reference_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.float().abs().amax(dim=1, keepdim=True) / fp8_max
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = (weight.float() / safe_scale).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn), scale


def _reference_mxfp4_pack(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight.shape[1] % 32 == 0
    packed = torch.zeros(
        weight.shape[0],
        weight.shape[1] // 2,
        dtype=torch.uint8,
    )
    scale = torch.ones(
        weight.shape[0],
        weight.shape[1] // 32,
        dtype=torch.uint8,
    )
    return packed, scale


def test_profile_selects_exact_phase_weight_sets() -> None:
    phase1 = expected_quantized_weight_names(5, get_ptq_profile("phase1"))
    phase2 = expected_quantized_weight_names(5, get_ptq_profile("phase2"))

    assert len(expected_dspark_tensor_names(5)) == 68
    assert len(phase1) == 30
    assert len(phase2) == 41
    assert "layers.4.mlp.down_proj.weight" in phase1
    assert "layers.0.self_attn.q_a_proj.weight" not in phase1
    assert "layers.0.self_attn.q_a_proj.weight" in phase2
    assert "context_proj.weight" not in phase1
    assert "context_proj.weight" in phase2
    assert not any("norm" in name or "markov_head" in name for name in phase2)


def test_quark_config_declares_offline_weights_and_dynamic_activations() -> None:
    phase1 = build_quark_ptpc_config(get_ptq_profile("phase1"))
    phase2 = build_quark_ptpc_config(get_ptq_profile("phase2"))

    assert phase1["global_quant_config"]["weight"] == {
        "qscheme": "per_channel",
        "dtype": "fp8_e4m3",
    }
    assert phase1["global_quant_config"]["input_tensors"] == {"is_dynamic": True}
    assert phase1["exclude"] == [
        "context_proj",
        "layers.*.self_attn.fused_qkv_a_proj",
    ]
    assert "exclude" not in phase2


def test_quark_mxfp4_config_declares_dynamic_a4w4() -> None:
    config = build_quark_quant_config(get_ptq_profile("phase1"), "mxfp4")

    assert config["global_quant_config"]["weight"] == {
        "qscheme": "per_group",
        "dtype": "fp4_e2m1",
    }
    assert config["global_quant_config"]["input_tensors"] == {"is_dynamic": True}
    assert config["exclude"] == [
        "context_proj",
        "layers.*.self_attn.fused_qkv_a_proj",
    ]


def test_mixed_selection_supports_projection_families_and_layers() -> None:
    mlp = select_mxfp4_weight_names(
        5,
        ["mlp_gate_up", "mlp_down"],
    )
    attention = select_mxfp4_weight_names(
        5,
        ["attn_q_b", "attn_kv_b", "attn_o"],
    )
    one_layer = select_mxfp4_weight_names(
        5,
        ["mlp_gate_up"],
        [2],
    )

    assert len(mlp) == 15
    assert len(attention) == 15
    assert one_layer == (
        "layers.2.mlp.gate_proj.weight",
        "layers.2.mlp.up_proj.weight",
    )
    assert set(mlp).isdisjoint(attention)
    with pytest.raises(ValueError, match="outside"):
        select_mxfp4_weight_names(5, ["attn_o"], [5])


def test_mixed_quark_config_overrides_only_selected_runtime_linears() -> None:
    config = build_mixed_quark_quant_config(
        5,
        ["mlp_gate_up", "mlp_down"],
    )

    assert config["global_quant_config"]["weight"]["dtype"] == "fp8_e4m3"
    assert set(config["layer_quant_config"]) == {
        "layers.*.mlp.gate_up_proj",
        "layers.*.mlp.down_proj",
    }
    for layer_config in config["layer_quant_config"].values():
        assert layer_config["weight"] == {
            "qscheme": "per_group",
            "dtype": "fp4_e2m1",
        }
        assert layer_config["input_tensors"] == {"is_dynamic": True}


def test_validate_checkpoint_is_fail_closed() -> None:
    names = set(expected_dspark_tensor_names(5))
    selected = validate_dspark_checkpoint(
        _config(),
        names,
        get_ptq_profile("phase1"),
    )
    assert len(selected) == 30

    names.remove("layers.3.mlp.down_proj.weight")
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_dspark_checkpoint(_config(), names, get_ptq_profile("phase1"))


def test_reference_quantizer_handles_zero_rows_and_reconstructs() -> None:
    weight = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [-4.0, -1.0, 2.0, 3.0]],
        dtype=torch.bfloat16,
    )
    quantized, scale = _reference_quantize(weight)
    reconstructed = quantized.float() * scale

    assert scale.shape == (2, 1)
    assert torch.isfinite(scale).all()
    assert scale[0].item() == 0
    assert (scale >= 0).all()
    torch.testing.assert_close(reconstructed, weight.float(), rtol=0.06, atol=0.03)


def test_convert_checkpoint_preserves_unselected_tensors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _tiny_checkpoint(source)

    manifest = convert_dspark_checkpoint(
        source,
        output,
        profile_name="phase1",
        device="cpu",
        quantize_fn=_reference_quantize,
    )

    assert manifest["selected_weight_count"] == 30
    assert (output / "README.md").read_text() == "fixture\n"
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["atom_dspark_ptq_profile"] == "phase1"
    assert output_config["quantization_config"]["global_quant_config"][
        "input_tensors"
    ] == {"is_dynamic": True}

    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as result:
        assert len(result.keys()) == 98
        norm = result.get_tensor("context_norm.weight")
        torch.testing.assert_close(norm, original["context_norm.weight"], rtol=0, atol=0)
        selected_name = "layers.0.self_attn.q_b_proj.weight"
        quantized = result.get_tensor(selected_name)
        scale = result.get_tensor(weight_scale_name(selected_name))
        assert quantized.dtype == torch.float8_e4m3fn
        assert scale.shape == (3, 1)
        torch.testing.assert_close(
            quantized.float() * scale,
            original[selected_name].float(),
            rtol=0.06,
            atol=0.03,
        )


def test_convert_checkpoint_records_mxfp4_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _tiny_checkpoint(source)

    manifest = convert_dspark_checkpoint(
        source,
        output,
        profile_name="phase1",
        quant_format="mxfp4",
        device="cpu",
        quantize_fn=_reference_mxfp4_pack,
    )

    assert manifest["quant_format"] == "mxfp4"
    assert manifest["selected_weight_count"] == 30
    assert manifest["activation_quantization"].endswith("_a4w4")
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["atom_dspark_ptq_format"] == "mxfp4"
    assert output_config["quantization_config"]["global_quant_config"]["weight"] == {
        "qscheme": "per_group",
        "dtype": "fp4_e2m1",
    }

    selected_name = "layers.0.self_attn.q_b_proj.weight"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as result:
        assert result.get_tensor(selected_name).shape == (3, 16)
        assert result.get_tensor(weight_scale_name(selected_name)).shape == (3, 1)


def test_compose_mixed_checkpoint_reuses_prequantized_tensors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fp8 = tmp_path / "fp8"
    mxfp4 = tmp_path / "mxfp4"
    output = tmp_path / "mixed"
    original = _tiny_checkpoint(source)
    convert_dspark_checkpoint(
        source,
        fp8,
        profile_name="phase1",
        device="cpu",
        quantize_fn=_reference_quantize,
    )
    convert_dspark_checkpoint(
        source,
        mxfp4,
        profile_name="phase1",
        quant_format="mxfp4",
        device="cpu",
        quantize_fn=_reference_mxfp4_pack,
    )

    manifest = compose_mixed_dspark_checkpoint(
        fp8,
        mxfp4,
        output,
        mxfp4_projections=["mlp_gate_up", "mlp_down"],
        selection_name="mlp_mxfp4",
    )

    assert manifest["mxfp4_weight_count"] == 15
    assert manifest["fp8_weight_count"] == 15
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["atom_dspark_ptq_format"] == "mixed_fp8_mxfp4"
    assert set(output_config["quantization_config"]["layer_quant_config"]) == {
        "layers.*.mlp.gate_up_proj",
        "layers.*.mlp.down_proj",
    }

    with (
        safe_open(fp8 / "model.safetensors", framework="pt", device="cpu") as fp8_model,
        safe_open(
            mxfp4 / "model.safetensors",
            framework="pt",
            device="cpu",
        ) as mxfp4_model,
        safe_open(output / "model.safetensors", framework="pt", device="cpu") as mixed,
    ):
        mlp_name = "layers.0.mlp.down_proj.weight"
        attn_name = "layers.0.self_attn.q_b_proj.weight"
        assert mixed.get_tensor(mlp_name).dtype == torch.uint8
        assert mixed.get_tensor(attn_name).dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            mixed.get_tensor(mlp_name),
            mxfp4_model.get_tensor(mlp_name),
        )
        torch.testing.assert_close(
            mixed.get_tensor(attn_name),
            fp8_model.get_tensor(attn_name),
        )
        torch.testing.assert_close(
            mixed.get_tensor("context_norm.weight"),
            original["context_norm.weight"],
        )


def test_dry_run_does_not_create_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _tiny_checkpoint(source)

    manifest = convert_dspark_checkpoint(
        source,
        output,
        profile_name="phase1",
        quant_format="mxfp4",
        dry_run=True,
    )

    assert manifest["selected_weight_count"] == 30
    assert manifest["quant_format"] == "mxfp4"
    assert not output.exists()


def test_native_export_merges_runtime_projections_and_context(tmp_path: Path) -> None:
    source = tmp_path / "source"
    portable = tmp_path / "portable"
    native = tmp_path / "native"
    original = _tiny_checkpoint(source)
    convert_dspark_checkpoint(
        source,
        portable,
        profile_name="phase1",
        device="cpu",
        quantize_fn=_reference_quantize,
    )

    manifest = export_atom_native_dspark_checkpoint(portable, native)

    assert manifest["architecture"] == "AtomK3DSparkModel"
    assert manifest["weight_layout"] == "logical_global_unshuffled"
    assert manifest["runtime_tp_slice"] is True
    assert manifest["runtime_preshuffle"] is True
    config = json.loads((native / "config.json").read_text())
    assert config["model_type"] == "atom_k3_dspark"
    assert config["architectures"] == ["AtomK3DSparkModel"]
    assert config["atom_native_checkpoint"]["fp8_dtype"] == "float8_e4m3fn"
    assert config["atom_native_checkpoint"]["context_kv_proj"]["shape"] == [
        576,
        7168,
    ]
    assert config["quantization_config"]["exclude"] == [
        "context_proj",
        "layers.*.self_attn.context_kv_proj",
        "layers.*.self_attn.fused_qkv_a_proj",
    ]
    assert "ATOM-native Kimi-K3 DSpark E4M3FN" in (
        native / "README.md"
    ).read_text()
    assert not (native / "atom_dspark_ptq_manifest.json").exists()

    layer = "layers.0"
    gate_name = f"{layer}.mlp.gate_proj.weight"
    up_name = f"{layer}.mlp.up_proj.weight"
    gate_up_name = f"{layer}.mlp.gate_up_proj.weight"
    q_a_name = f"{layer}.self_attn.q_a_proj.weight"
    kv_a_name = f"{layer}.self_attn.kv_a_proj_with_mqa.weight"
    fused_name = f"{layer}.self_attn.fused_qkv_a_proj.weight"
    context_name = f"{layer}.self_attn.context_kv_proj.weight"
    with (
        safe_open(portable / "model.safetensors", framework="pt", device="cpu") as src,
        safe_open(native / "model.safetensors", framework="pt", device="cpu") as dst,
    ):
        assert gate_name not in dst.keys()
        assert up_name not in dst.keys()
        assert q_a_name not in dst.keys()
        assert kv_a_name not in dst.keys()

        gate_up = dst.get_tensor(gate_up_name)
        gate_up_scale = dst.get_tensor(weight_scale_name(gate_up_name))
        assert gate_up.dtype == torch.float8_e4m3fn
        expected_bits = torch.cat(
            (src.get_tensor(gate_name), src.get_tensor(up_name)), dim=0
        ).view(torch.int8)
        expected_bits = expected_bits.clone()
        expected_bits[expected_bits == -128] = 0
        torch.testing.assert_close(
            gate_up.view(torch.int8), expected_bits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            gate_up_scale,
            torch.cat(
                (
                    src.get_tensor(weight_scale_name(gate_name)),
                    src.get_tensor(weight_scale_name(up_name)),
                ),
                dim=0,
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dst.get_tensor(fused_name),
            torch.cat((original[q_a_name], original[kv_a_name]), dim=0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dst.get_tensor(context_name), original[kv_a_name], rtol=0, atol=0
        )


def test_native_export_dry_run_does_not_create_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    portable = tmp_path / "portable"
    native = tmp_path / "native"
    _tiny_checkpoint(source)
    convert_dspark_checkpoint(
        source,
        portable,
        profile_name="phase1",
        device="cpu",
        quantize_fn=_reference_quantize,
    )

    manifest = export_atom_native_dspark_checkpoint(
        portable,
        native,
        dry_run=True,
    )

    assert manifest["format"] == "atom_k3_dspark_fp8"
    assert not native.exists()

