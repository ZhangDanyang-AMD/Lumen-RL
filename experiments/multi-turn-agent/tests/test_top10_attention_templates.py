from __future__ import annotations

import ast
import math
from typing import Any

import pytest

from multi_tune_agent.top10_attention_templates import (
    render_attention_kernel,
    render_attention_runner,
)


CONTRACTS: dict[str, dict[str, Any]] = {
    "mha": {
        "operator": "mha",
        "architecture": "gfx942",
        "shape": {"B": 2, "SQ": 7, "SK": 11, "HQ": 8, "HK": 2, "D": 16},
    },
    "mla": {
        "operator": "multi_latent_attention",
        "target_gpu": "gfx942",
        "shape": {"B": 2, "S": 9, "H": 4, "KV": 12, "ROPE": 6},
    },
    "paged_attention": {
        "operator": "paged_attention",
        "arch": "gfx942",
        "shape": {"B": 2, "HQ": 6, "HK": 2, "BLOCK": 4, "S": 10, "D": 8},
    },
}


def _runner_namespace(family: str) -> dict[str, Any]:
    source = render_attention_runner(family, CONTRACTS[family])
    namespace: dict[str, Any] = {
        "__name__": "rendered_runner",
        "__file__": "/tmp/rendered/scripts/task_runner.py",
    }
    exec(compile(source, "<runner>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("family", sorted(CONTRACTS))
@pytest.mark.parametrize("language", ["hip", "triton"])
def test_rendered_kernels_are_standalone_python(family: str, language: str) -> None:
    source = render_attention_kernel(family, language, CONTRACTS[family])
    ast.parse(source)
    lowered = source.lower()
    assert "aiter" not in lowered
    assert "torch.matmul" not in source
    assert "torch.softmax" not in source
    assert "gfx942" in source if language == "hip" else "@triton.jit" in source
    if language == "hip":
        assert "#include <hip/hip_runtime.h>" in source
        assert "hipLaunchKernelGGL" in source
        assert "--offload-arch=gfx942" in source
    else:
        assert f"def {family}(" in source
        assert "tl.exp" in source
        assert "tl.max" in source


def test_mha_kernel_encodes_causality_batch_heads_and_gqa() -> None:
    source = render_attention_kernel("mha", "triton", CONTRACTS["mha"])
    assert "b = row // (SQ * HQ)" in source
    assert "hk = hq // (HQ // HK)" in source
    assert "key_limit = qi + SK - SQ" in source
    assert "(s <= key_limit)" in source


def test_mla_kernel_uses_both_compressed_and_rope_segments() -> None:
    source = render_attention_kernel("mla", "triton", CONTRACTS["mla"])
    assert "q_latent" in source and "latent_kv" in source
    assert "q_rope" in source and "rope_k" in source
    assert "KV + ROPE" in source
    assert "weights[:, None] * kv" in source


def test_paged_kernel_follows_block_table_and_gqa_mapping() -> None:
    source = render_attention_kernel(
        "paged_attention", "triton", CONTRACTS["paged_attention"]
    )
    assert "physical = tl.load(block_table" in source
    assert "logical_block, offset = s // BLOCK, s % BLOCK" in source
    assert "hk = hq // (HQ // HK)" in source
    assert "length = tl.load(seq_lens + b)" in source


@pytest.mark.parametrize("family", sorted(CONTRACTS))
def test_runner_has_gfx942_gate_and_geak_performance_report(family: str) -> None:
    source = render_attention_runner(family, CONTRACTS[family])
    ast.parse(source)
    assert "torch.version.hip is None" in source
    assert 'arch != "gfx942"' in source
    assert "fp32_pytorch_oracle" in source
    assert '"performance_report.json"' in source
    assert '"test_cases"' in source
    assert '"execution_time_ms"' in source
    assert "SEED = 942_031" in source


def test_mha_oracle_matches_explicit_causal_gqa_loops() -> None:
    torch = pytest.importorskip("torch")
    namespace = _runner_namespace("mha")
    torch.manual_seed(3)
    q = torch.randn(1, 3, 4, 5, dtype=torch.bfloat16)
    k = torch.randn(1, 5, 2, 5, dtype=torch.bfloat16)
    v = torch.randn(1, 5, 2, 5, dtype=torch.bfloat16)
    actual = namespace["fp32_pytorch_oracle"]((q, k, v)).float()
    expected = torch.empty_like(actual)
    for qi in range(3):
        for hq in range(4):
            hk = hq // 2
            logits = torch.tensor(
                [
                    torch.dot(q[0, qi, hq].float(), k[0, s, hk].float())
                    / math.sqrt(5)
                    for s in range(qi + 3)
                ]
            )
            expected[0, qi, hq] = sum(
                torch.softmax(logits, 0)[s] * v[0, s, hk].float()
                for s in range(qi + 3)
            )
    torch.testing.assert_close(actual, expected.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_mla_oracle_matches_explicit_segmented_decode() -> None:
    torch = pytest.importorskip("torch")
    namespace = _runner_namespace("mla")
    torch.manual_seed(4)
    ql = torch.randn(1, 2, 3, dtype=torch.bfloat16)
    qr = torch.randn(1, 2, 2, dtype=torch.bfloat16)
    kv = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    rk = torch.randn(1, 4, 2, dtype=torch.bfloat16)
    actual = namespace["fp32_pytorch_oracle"]((ql, qr, kv, rk)).float()
    expected = torch.empty_like(actual)
    for h in range(2):
        scores = torch.tensor(
            [
                (
                    torch.dot(ql[0, h].float(), kv[0, s].float())
                    + torch.dot(qr[0, h].float(), rk[0, s].float())
                )
                / math.sqrt(5)
                for s in range(4)
            ]
        )
        expected[0, h] = sum(
            torch.softmax(scores, 0)[s] * kv[0, s].float() for s in range(4)
        )
    torch.testing.assert_close(actual, expected.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_paged_oracle_matches_explicit_noncontiguous_page_walk() -> None:
    torch = pytest.importorskip("torch")
    namespace = _runner_namespace("paged_attention")
    torch.manual_seed(5)
    q = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    kc = torch.randn(3, 2, 2, 3, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    table = torch.tensor([[2, 0]], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)
    actual = namespace["fp32_pytorch_oracle"](
        (q, kc, vc, table, lengths)
    ).float()
    expected = torch.empty_like(actual)
    locations = [(2, 0), (2, 1), (0, 0)]
    for hq in range(4):
        hk = hq // 2
        scores = torch.tensor(
            [
                torch.dot(q[0, hq].float(), kc[block, offset, hk].float())
                / math.sqrt(3)
                for block, offset in locations
            ]
        )
        expected[0, hq] = sum(
            torch.softmax(scores, 0)[s] * vc[block, offset, hk].float()
            for s, (block, offset) in enumerate(locations)
        )
    torch.testing.assert_close(actual, expected.to(torch.bfloat16).float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("family", "language", "contract"),
    [
        ("unknown", "triton", {"architecture": "gfx942"}),
        ("mha", "cuda", CONTRACTS["mha"]),
        ("mha", "triton", {**CONTRACTS["mha"], "architecture": "gfx90a"}),
    ],
)
def test_rejects_unsupported_family_language_or_architecture(
    family: str, language: str, contract: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        render_attention_kernel(family, language, contract)
