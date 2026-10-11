from __future__ import annotations

import ast

import pytest

from multi_tune_agent.top10_sequence_templates import (
    SequenceTemplateError,
    render_sequence_kernel,
    render_sequence_runner,
)


ROPE = {
    "operator": "rope_kv_cache",
    "mode": "sbhd",
    "shape": {"S": 2048, "B": 2, "H": 32, "D": 128},
    "input_dtype": "bf16",
    "output_dtype": "bf16",
}
SAMPLING = {
    "operator": "sampling",
    "mode": "top_k_top_p",
    "shape": {"B": 99, "VOCAB": 128256},
    "input_dtype": "bf16",
    "output_dtype": "int32",
}


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_rope_kv_cache_is_custom_rotation_and_observable_mutation(language: str) -> None:
    source = render_sequence_kernel("rope_kv_cache", language, ROPE)
    ast.parse(source)
    assert "partner" in source
    assert "key_cache" in source and "value_cache" in source
    assert "slots" in source
    assert "q_out" in source and "k_out" in source
    assert "aiter" not in source.lower()
    assert all(
        operator not in source
        for operator in ("torch.empty", "torch.cat", "torch.index", "torch.matmul")
    )
    if language == "hip":
        assert "#include <hip/hip_runtime.h>" in source
        assert "--offload-arch=gfx942" in source
    else:
        assert "@triton.jit" in source
        assert "tl.store(key_cache" in source


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_sampling_is_stochastic_from_probs_with_exact_boundaries(language: str) -> None:
    source = render_sequence_kernel("sampling", language, SAMPLING)
    ast.parse(source)
    assert "uniforms" in source
    assert "top_k" in source and "top_p" in source
    assert "target >= prefix" in source
    assert "target < prefix + " in source
    assert "argmax" not in source
    assert "aiter" not in source.lower()
    assert all(
        operator not in source
        for operator in ("torch.argmax", "torch.multinomial", "torch.sort", "torch.topk")
    )
    if language == "hip":
        assert "p == previous_p && token > previous_token" in source
        assert "--offload-arch=gfx942" in source
    else:
        assert "(other_p == probability) & (other < token)" in source
        assert "@triton.jit" in source


@pytest.mark.parametrize(
    ("family", "contract"),
    [("rope_kv_cache", ROPE), ("sampling", SAMPLING)],
)
def test_runner_has_independent_gfx942_three_phase_gate(
    family: str, contract: dict,
) -> None:
    runner = render_sequence_runner(family, contract)
    ast.parse(runner)
    assert 'if arch != "gfx942"' in runner
    assert "independent_torch_oracle" in runner
    assert 'choices=("compile", "correctness", "performance")' in runner
    assert "performance_report.json" in runner
    assert "torch.manual_seed(9410)" in runner
    assert family in runner
    if family == "rope_kv_cache":
        assert "(8, 4, 32, 4)" in runner
        assert "expected_key_cache" in runner
        assert "expected_value_cache" in runner
        assert "torch.full" in runner
    else:
        assert "(8, 128)" in runner
        assert "torch.argsort" in runner
        assert "stable=True" in runner
        assert "torch.cumsum" in runner
        assert "degenerated to argmax" in runner


def test_fused_write_contract_uses_contract_derived_small_dimensions() -> None:
    contract = {
        "operator": "rope_kv_cache",
        "mode": "fused_write",
        "shape": {"TOKENS": 3, "HEADS": 2, "D": 16, "BLOCK": 2},
        "input_dtype": "bf16",
        "output_dtype": "bf16",
    }
    runner = render_sequence_runner("rope_kv_cache", contract)
    assert "(3, 2, 16, 2)" in runner


def test_top_p_mode_disables_the_top_k_cutoff() -> None:
    contract = {**SAMPLING, "mode": "top_p"}
    runner = render_sequence_runner("sampling", contract)
    assert "return probs, uniforms, 0, 0.75, out" in runner


@pytest.mark.parametrize(
    ("family", "language", "contract", "message"),
    [
        ("sampling", "cuda", SAMPLING, "language"),
        ("sampling", "hip", {**SAMPLING, "mode": "argmax"}, "mode"),
        ("rope_kv_cache", "triton", {**ROPE, "shape": {**ROPE["shape"], "D": 7}}, "odd"),
        ("sampling", "hip", {**SAMPLING, "operator": "rope_kv_cache"}, "operator"),
    ],
)
def test_unsupported_contracts_fail_closed(
    family: str, language: str, contract: dict, message: str
) -> None:
    with pytest.raises(SequenceTemplateError, match=message):
        render_sequence_kernel(family, language, contract)
