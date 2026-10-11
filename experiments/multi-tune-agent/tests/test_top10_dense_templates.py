from __future__ import annotations

import ast
from copy import deepcopy

import pytest

from multi_tune_agent.top10_dense_templates import (
    DenseTemplateError,
    render_dense_kernel,
    render_dense_runner,
)


MOE_FP8 = {
    "operator": "fused_moe",
    "mode": "fp8",
    "projection": "one_projection",
    "shape": {
        "TOKENS": 64,
        "MODEL": 4096,
        "INTER": 8192,
        "EXPERTS": 8,
        "TOPK": 2,
    },
    "input_dtype": "bf16",
    "weight_dtype": "fp8_e4m3fnuz",
    "output_dtype": "bf16",
}

MOE_SILU = {
    "operator": "fused_moe",
    "mode": "silu",
    "shape": {
        "TOKENS": 128,
        "MODEL": 4096,
        "INTER": 14336,
        "EXPERTS": 8,
        "TOPK": 2,
    },
    "input_dtype": "bf16",
    "weight_dtype": "bf16",
    "output_dtype": "bf16",
}

MOE_INT8 = {
    **MOE_FP8,
    "mode": "int8",
    "weight_dtype": "int8",
}

BLOCK_FP8 = {
    "operator": "blockscale_gemm",
    "mode": "fp8_block128",
    "shape": {"M": 128, "N": 4096, "K": 4096},
    "input_dtype": "fp8_e4m3fnuz",
    "weight_dtype": "fp8_e4m3fnuz",
    "output_dtype": "bf16",
    "scale": {"activation": "block128", "weight": "block128"},
}

BLOCK_INT8 = {
    "operator": "blockscale_gemm",
    "mode": "int8_block128",
    "shape": {"M": 16, "N": 2112, "K": 7168},
    "input_dtype": "int8",
    "weight_dtype": "int8",
    "output_dtype": "bf16",
    "scale_block_k": 128,
    "scale": {"activation": "block128", "weight": "block128"},
}


@pytest.mark.parametrize("language", ["hip", "triton"])
@pytest.mark.parametrize("contract", [MOE_FP8, MOE_INT8, MOE_SILU])
def test_fused_moe_is_routed_weighted_silu_one_projection(
    language: str, contract: dict
) -> None:
    source = render_dense_kernel("fused_moe", language, contract)

    assert "expert_ids" in source
    assert "routing_weights" in source
    assert "weight_scale" in source
    assert "one_projection" in source
    assert "silu" in source
    assert "torch.matmul" not in source
    assert "torch.nn" not in source
    assert "aiter" not in source.lower()
    assert (
        "#include <hip/hip_runtime.h>" in source
        if language == "hip"
        else "@triton.jit" in source
    )
    if contract["weight_dtype"] == "fp8_e4m3fnuz":
        assert "torch.float8_e4m3fnuz" in source
        if language == "hip":
            assert "fp8_e4m3fnuz_to_float" in source
    elif contract["weight_dtype"] == "int8":
        assert "torch.int8" in source
    compile(source, "kernel.py", "exec")


@pytest.mark.parametrize("language", ["hip", "triton"])
@pytest.mark.parametrize("contract", [BLOCK_FP8, BLOCK_INT8])
def test_blockscale_scales_each_k_block_before_accumulating(
    language: str, contract: dict
) -> None:
    source = render_dense_kernel("blockscale_gemm", language, contract)

    assert "SCALE_BLOCK_K" in source
    assert "K_BLOCKS" in source
    assert "a_scale" in source and "weight_scale" in source
    assert "result +=" in source
    assert "torch.matmul" not in source
    assert "aiter" not in source.lower()
    if contract["weight_dtype"] == "int8":
        assert "int32" in source
        assert "torch.int8" in source
    else:
        assert "torch.float8_e4m3fnuz" in source
    compile(source, "kernel.py", "exec")


@pytest.mark.parametrize(
    ("family", "contract"),
    [("fused_moe", MOE_FP8), ("blockscale_gemm", BLOCK_INT8)],
)
def test_runner_is_small_deterministic_and_independently_verifiable(
    family: str, contract: dict
) -> None:
    source = render_dense_runner(family, contract)
    tree = ast.parse(source)

    assert tree
    assert "fp32_int32_oracle" in source
    assert "torch.matmul" in source
    assert 'SUPPORTED_ARCH = "gfx942"' in source
    assert "gcnArchName" in source
    assert 'choices=("compile", "correctness", "performance")' in source
    assert "performance_report.json" in source
    assert '"execution_time_ms"' in source
    assert "torch.manual_seed(seed)" in source
    if family == "fused_moe":
        assert "'TOKENS': 8" in source
        assert "'MODEL': 32" in source
        assert "'INTER': 24" in source
        assert "expert_ids[token, route]" in source
        assert "routing[token, route]" in source
    else:
        assert "'M': 8" in source
        assert "'N': 16" in source
        assert "'K': 160" in source
        assert "for kb in range(blocks)" in source
        assert ".to(torch.int32).float()" in source


def test_rendering_is_deterministic_and_does_not_mutate_contract() -> None:
    contract = deepcopy(MOE_FP8)
    original = deepcopy(contract)
    assert render_dense_kernel("fused_moe", "hip", contract) == render_dense_kernel(
        "fused_moe", "hip", contract
    )
    assert render_dense_runner("fused_moe", contract) == render_dense_runner(
        "fused_moe", contract
    )
    assert contract == original


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_preshuffle_mode_reads_the_frozen_16x16_physical_layout(
    language: str,
) -> None:
    contract = deepcopy(BLOCK_FP8)
    contract["mode"] = "preshuffle"
    source = render_dense_kernel("blockscale_gemm", language, contract)
    runner = render_dense_runner("blockscale_gemm", contract)

    assert (
        "(n / 16) * (K / 32)" in source
        if language == "hip"
        else "(n // 16) * (K // 32)" in source
    )
    assert "MODE = 'preshuffle'" in runner
    assert ".permute(0, 2, 3, 1, 4)" in runner
    assert ".permute(0, 3, 1, 2, 4)" in runner


def test_arch_specific_nested_fp8_contract_is_normalized_exactly() -> None:
    contract = {
        "operator": "blockscale_gemm",
        "shape": {"M": 128, "N": 24576, "K": 1536},
        "dtype": {
            "input": "fp8_e4m3_arch_specific",
            "weight": "fp8_e4m3_arch_specific",
            "accum": "fp32",
            "output": "bf16",
        },
        "layout": "TN",
        "scale": {
            "activation_block": [1, 128],
            "weight_block": [128, 128],
        },
    }
    for language in ("hip", "triton"):
        source = render_dense_kernel("blockscale_gemm", language, contract)
        assert "torch.float8_e4m3fnuz" in source
        compile(source, "kernel.py", "exec")


@pytest.mark.parametrize(
    ("family", "language", "change", "message"),
    [
        ("fused_moe", "cuda", {}, "unsupported language"),
        (
            "fused_moe",
            "hip",
            {"projection": "two_projection"},
            "one-projection",
        ),
        (
            "blockscale_gemm",
            "triton",
            {"scale_block_k": 64},
            "scale_block_k=128",
        ),
        (
            "blockscale_gemm",
            "hip",
            {"weight_dtype": "bf16"},
            "weight dtype",
        ),
    ],
)
def test_unsupported_contract_drift_fails_closed(
    family: str, language: str, change: dict, message: str
) -> None:
    contract = deepcopy(MOE_SILU if family == "fused_moe" else BLOCK_FP8)
    contract.update(change)
    with pytest.raises(DenseTemplateError, match=message):
        render_dense_kernel(family, language, contract)


def test_shape_and_operator_mismatch_fail_closed() -> None:
    missing = deepcopy(MOE_SILU)
    del missing["shape"]["TOPK"]
    with pytest.raises(DenseTemplateError, match="TOPK"):
        render_dense_runner("fused_moe", missing)

    mismatch = deepcopy(BLOCK_FP8)
    mismatch["operator"] = "fused_moe"
    with pytest.raises(DenseTemplateError, match="does not match"):
        render_dense_kernel("blockscale_gemm", "hip", mismatch)
