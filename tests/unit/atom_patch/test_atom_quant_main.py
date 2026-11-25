"""CPU/mock contract tests for opt-in ATOM main MXFP4-2D integration."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "lumenrl"
    / "engine"
    / "inference"
    / "atom_patch"
    / "quant_main.py"
)


def _module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "lumen_atom_quant_main_under_test", _SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("atom.config", "atom.quant_spec", "atom.model_ops.linear"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return module


def _fake_atom(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float4_e2m1fn_x2 = "fp4"
    torch.uint8 = "uint8"
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Parameter = lambda value, requires_grad=False: value
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch.nn)
    atom = types.ModuleType("atom")
    atom.__path__ = []
    model_ops = types.ModuleType("atom.model_ops")
    model_ops.__path__ = []
    quant_spec = types.ModuleType("atom.quant_spec")
    config = types.ModuleType("atom.config")
    linear = types.ModuleType("atom.model_ops.linear")

    @dataclass(frozen=True)
    class LayerQuantConfig:
        quant_type: object = None
        quant_dtype: object = None
        weight_block_2d: bool = False

    class QuarkOnlineParser:
        def parse(self, config):
            formats = config.get("layer_quant_config", {})
            if config.get("global_quant_config") not in ("", "mxfp4"):
                raise ValueError("unsupported global format")
            if any(fmt not in ("mxfp4", "mxfp8") for fmt in formats.values()):
                raise ValueError("unsupported layer format")
            return SimpleNamespace(
                global_spec=LayerQuantConfig(quant_type="fp4"),
                layer_pattern_specs=[
                    (name, LayerQuantConfig(quant_type=fmt))
                    for name, fmt in formats.items()
                ],
                exclude_layers=list(config.get("exclude_layer", [])),
            )

    quant_spec.LayerQuantConfig = LayerQuantConfig
    quant_spec.QuarkOnlineParser = QuarkOnlineParser
    config.QuantizationConfig = type(
        "QuantizationConfig",
        (),
        {"__init__": lambda self, config=None, online_quant_config=None, model=None: None},
    )
    quant_spec.should_stream_online_quant = lambda *args: True
    linear.should_stream_online_quant = quant_spec.should_stream_online_quant
    moe = types.ModuleType("atom.model_ops.moe")
    moe.FusedMoE = type("FusedMoE", (), {"_online_quant": lambda self: None})
    layernorm = types.ModuleType("atom.model_ops.layernorm")
    layernorm.RMSNorm = type(
        "RMSNorm", (), {"online_quantize_activation": lambda self: None}
    )
    linear.LinearBase = type(
        "LinearBase",
        (),
        {
            "online_quantize_weight": lambda self: "original_online",
            "process_weights_after_loading": lambda self: "original_process",
            "forward": lambda self, *args, **kwargs: "original_forward",
        },
    )
    linear.use_fp4_non_shuffle_triton_gemm = lambda: True
    linear.gemm_afp4wfp4 = lambda *args: None
    linear.dtypes = SimpleNamespace(bf16="bf16")
    linear.QuantType = SimpleNamespace(
        No=SimpleNamespace(value=0), per_1x32=SimpleNamespace(value=2)
    )
    for name, item in (
        ("atom", atom),
        ("atom.model_ops", model_ops),
        ("atom.quant_spec", quant_spec),
        ("atom.config", config),
        ("atom.model_ops.linear", linear),
        ("atom.model_ops.moe", moe),
        ("atom.model_ops.layernorm", layernorm),
    ):
        monkeypatch.setitem(sys.modules, name, item)
    atom.quant_spec = quant_spec
    atom.config = config
    atom.model_ops = model_ops
    model_ops.linear = linear
    model_ops.moe = moe
    model_ops.layernorm = layernorm
    return quant_spec, config, linear


def test_default_off_does_not_import_atom(monkeypatch):
    patch = _module(monkeypatch)
    monkeypatch.delenv("LUMEN_ATOM_MXFP4_2D", raising=False)
    assert patch.install() is False
    assert "atom.config" not in sys.modules


def test_opt_in_requires_nonshuffle_before_import(monkeypatch):
    patch = _module(monkeypatch)
    monkeypatch.setenv("LUMEN_ATOM_MXFP4_2D", "1")
    monkeypatch.delenv("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", raising=False)
    with pytest.raises(ValueError, match="NON_SHUFFLE"):
        patch.install()
    assert "atom.config" not in sys.modules


def test_installer_preserves_other_linear_paths_and_is_idempotent(monkeypatch):
    patch = _module(monkeypatch)
    monkeypatch.setenv("LUMEN_ATOM_MXFP4_2D", "1")
    monkeypatch.setenv("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", "1")
    _, _, linear = _fake_atom(monkeypatch)
    assert patch.install() is True
    first = linear.LinearBase.forward
    assert patch.install() is True
    assert linear.LinearBase.forward is first
    layer = SimpleNamespace(mxfp4_2d=False)
    assert linear.LinearBase.forward(layer, object()) == "original_forward"


def test_parser_retains_other_formats_exclusions_and_2d_marker(monkeypatch):
    patch = _module(monkeypatch)
    monkeypatch.setenv("LUMEN_ATOM_MXFP4_2D", "1")
    monkeypatch.setenv("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", "1")
    quant_spec, _, _ = _fake_atom(monkeypatch)
    patch.install()
    raw = {
        "global_quant_config": "mxfp4_2d",
        "layer_quant_config": {"dense.*": "mxfp4_2d", "other.*": "mxfp8"},
        "exclude_layer": ["lm_head"],
    }
    parsed = quant_spec.QuarkOnlineParser().parse(raw)
    assert parsed.global_spec.weight_block_2d
    assert parsed.layer_pattern_specs[0][1].weight_block_2d
    assert not parsed.layer_pattern_specs[1][1].weight_block_2d
    assert parsed.exclude_layers == ["lm_head"]
    assert raw["global_quant_config"] == "mxfp4_2d"
    with pytest.raises(ValueError, match="unsupported"):
        quant_spec.QuarkOnlineParser().parse({"global_quant_config": "bad"})


def test_quant_shape_refusal_and_aiter_dependency(monkeypatch):
    patch = _module(monkeypatch)
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="BF16"):
        patch.quantize_mxfp4_2d_weight(torch.ones((32, 32), dtype=torch.float32))
    with pytest.raises(ValueError, match="divisible by 32"):
        patch.quantize_mxfp4_2d_weight(torch.ones((31, 32), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible by 32"):
        patch.quantize_mxfp4_2d_activation(torch.ones((2, 31), dtype=torch.bfloat16))
