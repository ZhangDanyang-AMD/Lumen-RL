from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multi_tune_agent.trusted_template_reuse import (
    ReuseError,
    ReusePlanItem,
    _adapt_kernel,
    _adapt_runner,
    build_parser,
    build_reuse_plan,
    materialize_candidate,
)


PRODUCTION = Path(
    "/home/danyzhan/phase1_control/production-wave-300/production-cases.yaml"
)
REQUESTS = Path(
    "/home/danyzhan/phase1_control/splits/v3/generation-requests.yaml"
)
DEV_PRODUCTION = Path(
    "/home/danyzhan/phase1_control/dev-wave-v1/production-cases.yaml"
)
DEV_REQUESTS = Path(
    "/home/danyzhan/phase1_control/splits/v4/generation-requests.yaml"
)


@pytest.fixture(scope="module")
def real_plan():
    if not PRODUCTION.is_file() or not REQUESTS.is_file():
        pytest.skip("phase-one control manifests are unavailable")
    return build_reuse_plan(PRODUCTION, REQUESTS)


def test_real_train_plan_has_reviewed_partition(real_plan) -> None:
    assert real_plan.counts == {
        "exact": 30,
        "parameterizable": 77,
        "unsupported": 90,
    }
    assert len(real_plan.items) == 197
    assert all(
        item.request["seed_provenance"]["split_group"] == "train"
        for item in real_plan.items
    )


def test_materialization_is_untrusted_deterministic_and_non_mutating(
    real_plan, tmp_path: Path
) -> None:
    item = next(
        candidate
        for candidate in real_plan.items
        if candidate.category == "parameterizable"
        and candidate.request["recognized_contract"]["operator"]
        == "dynamic_per_token_quant"
    )
    assert item.template is not None
    source_hashes = {
        relative: hashlib.sha256((item.template.path / relative).read_bytes()).hexdigest()
        for relative in (
            "kernel.py",
            "config.yaml",
            "scripts/task_runner.py",
            "metadata.json",
        )
    }

    first = materialize_candidate(item, tmp_path / "candidates")
    second = materialize_candidate(item, tmp_path / "candidates")

    assert first.path == second.path
    assert first.valid
    metadata = json.loads((first.path / "metadata.json").read_text(encoding="utf-8"))
    assert "trust" not in metadata
    assert metadata["reuse_status"] == "untrusted_pending_gate"
    assert (
        metadata["provenance"]["case_seed"]
        == item.request["seed_provenance"]
    )
    assert metadata["provenance"]["template_source"]["case_id"] == item.template.task["id"]
    assert {
        relative: hashlib.sha256((item.template.path / relative).read_bytes()).hexdigest()
        for relative in source_hashes
    } == source_hashes


def test_materialization_identity_includes_request_id(
    real_plan, tmp_path: Path
) -> None:
    item = next(candidate for candidate in real_plan.items if candidate.category == "parameterizable")
    sibling = ReusePlanItem(
        request={**item.request, "id": f"{item.request['id']}-sibling"},
        category=item.category,
        template=item.template,
        reason=item.reason,
    )

    first = materialize_candidate(item, tmp_path / "candidates")
    second = materialize_candidate(sibling, tmp_path / "candidates")

    assert first.contract_hash == second.contract_hash
    assert first.path != second.path


def test_explicit_adapter_rejects_odd_split_dimension() -> None:
    with pytest.raises(ReuseError, match="even split-last dimension"):
        _adapt_runner(
            "shape = (256, 768)\n",
            operator="fused_silu_mul",
            source_shape=(256, 768),
            target_shape=(2, 7),
            epsilon=None,
        )


def test_explicit_adapter_does_not_replace_unrelated_numbers() -> None:
    adapted = _adapt_runner(
        "SHAPE = (32, 4096)\nWARMUP = 32\nEPSILON = 1e-6\n",
        operator="rms_norm",
        source_shape=(32, 4096),
        target_shape=(71, 3571),
        epsilon=1e-5,
    )
    assert "SHAPE = (71, 3571)" in adapted
    assert "WARMUP = 32" in adapted
    assert "EPSILON = 1e-05" in adapted


def test_rms_runner_updates_scalar_weight_constructor() -> None:
    adapted = _adapt_runner(
        "x = torch.randn(1, 4096)\nweight = torch.randn(4096)\nWARMUP = 4096\n",
        operator="rms_norm",
        source_shape=(32, 4096),
        target_shape=(71, 3571),
        epsilon=1e-5,
    )
    assert "torch.randn(1, 3571)" in adapted
    assert "torch.randn(3571)" in adapted
    assert "WARMUP = 4096" in adapted


def test_hip_rms_kernel_updates_verified_shape_anchors() -> None:
    source = """\
input.size(0) == 32 && input.size(1) == 4096
"input shape must be 32x4096"
weight.size(0) == 4096
"weight shape must be 4096"
"""
    adapted = _adapt_kernel(
        source,
        operator="rms_norm",
        language="hip",
        source_shape=(32, 4096),
        target_shape=(71, 3571),
    )
    assert "input.size(0) == 71 && input.size(1) == 3571" in adapted
    assert "weight.size(0) == 3571" in adapted


def test_triton_tensor_quant_updates_kernel_assertion() -> None:
    adapted = _adapt_kernel(
        """\
import triton.language as tl
def quant(input_tensor, magnitude, step):
    unrounded = magnitude / step
    rounded = tl.floor(unrounded).to(tl.int32)
    rounded += (unrounded - tl.floor(unrounded) > 0.5).to(tl.int32)
    subnormal_unrounded = magnitude * 1024.0
    subnormal_floor = tl.floor(subnormal_unrounded)
    subnormal_bits = subnormal_floor.to(tl.int32)
    subnormal_bits += (subnormal_unrounded - subnormal_floor > 0.5).to(tl.int32)
    assert input_tensor.shape == (32, 8192)
    block = 1024
""",
        operator="dynamic_per_tensor_quant",
        language="triton",
        source_shape=(32, 8192),
        target_shape=(93, 75),
    )
    assert "input_tensor.shape == (93, 75)" in adapted
    assert "block = 1024" in adapted
    assert "rounded_fraction == 0.5" in adapted
    assert "(rounded_floor & 1) == 1" in adapted
    assert "subnormal_fraction == 0.5" in adapted
    assert "(subnormal_bits & 1) == 1" in adapted
    assert "nextafter" not in adapted


def test_hip_fused_silu_flattens_and_restores_multidimensional_shape() -> None:
    source = """\
input.size(0) == 256 && input.size(1) == 768
"input shape must be 256x768"
def fused_silu_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    return _module().fused_silu_mul(input_tensor)
"""
    adapted = _adapt_kernel(
        source,
        operator="fused_silu_mul",
        language="hip",
        source_shape=(256, 768),
        target_shape=(2, 16, 128),
    )
    assert "input.size(0) == 32 && input.size(1) == 128" in adapted
    assert "matrix = input_tensor.reshape(-1, 128)" in adapted
    assert "output.reshape(*original_shape[:-1], original_shape[-1] // 2)" in adapted

    runner = _adapt_runner(
        "shape = (256, 768)\n"
        "a = input_tensor[:, :half]\n"
        "b = input_tensor[:, half:]\n",
        operator="fused_silu_mul",
        language="hip",
        source_shape=(256, 768),
        target_shape=(2, 16, 128),
        epsilon=None,
    )
    assert "input_tensor[..., :half]" in runner
    assert "input_tensor[..., half:]" in runner


def test_hip_gemm_updates_contract_and_launch_grid() -> None:
    source = """\
constexpr int M = 128;
constexpr int N = 32;
constexpr int K = 8192;
{128, 8192}
{32, 8192}
{128, 32}
"A must be 128x8192"
"B must be 32x8192"
gemm_kernel<<<16, 256, 0, stream>>>
"""
    adapted = _adapt_kernel(
        source,
        operator="gemm",
        language="hip",
        source_shape=(128, 32, 8192),
        target_shape=(32, 7168, 256),
    )
    assert "constexpr int M = 32;" in adapted
    assert "{7168, 256}" in adapted
    assert "gemm_kernel<<<896, 256, 0, stream>>>" in adapted


def test_hip_tensor_quant_removes_shape_sensitive_nextafter() -> None:
    adapted = _adapt_kernel(
        "*scale = nextafterf(quotient, INFINITY);\n",
        operator="dynamic_per_tensor_quant",
        language="hip",
        source_shape=(32, 8192),
        target_shape=(10, 128),
    )
    assert adapted == "*scale = quotient;\n"


def test_hip_static_quant_updates_only_verified_shape_anchors() -> None:
    adapted = _adapt_kernel(
        'input.sizes() == at::IntArrayRef({2048, 1024})\n'
        '"input shape must be 2048x1024"\n'
        "constexpr int threads = 256;\n",
        operator="static_per_tensor_quant",
        language="hip",
        source_shape=(2048, 1024),
        target_shape=(193, 75),
    )
    assert "at::IntArrayRef({193, 75})" in adapted
    assert '"input shape must be 193x75"' in adapted
    assert "constexpr int threads = 256;" in adapted


def test_real_dev_plan_uses_only_frozen_dev_requests() -> None:
    if not all(path.is_file() for path in (PRODUCTION, DEV_PRODUCTION, DEV_REQUESTS)):
        pytest.skip("phase-one Dev control manifests are unavailable")
    plan = build_reuse_plan(
        DEV_PRODUCTION,
        DEV_REQUESTS,
        request_split="dev",
        template_catalogs=(PRODUCTION,),
    )
    assert sum(plan.counts.values()) == 44
    assert plan.counts["exact"] >= 2
    assert plan.counts["unsupported"] <= 18
    assert all(
        item.request["seed_provenance"]["split_group"] == "dev"
        and item.request["seed_provenance"]["split_version"] == "v4"
        for item in plan.items
    )
    parameterizable_operators = {
        item.request["recognized_contract"]["operator"]
        for item in plan.items
        if item.category == "parameterizable"
    }
    assert parameterizable_operators <= {"softmax", "static_per_tensor_quant"}


def test_repeatable_case_selector_is_available() -> None:
    args = build_parser().parse_args(
        [
            "gate",
            "--production-catalog",
            "catalog.yaml",
            "--requests",
            "requests.yaml",
            "--case-id",
            "case-a",
            "--case-id",
            "case-b",
        ]
    )
    assert args.case_id == ["case-a", "case-b"]


def test_dev_acceleration_cli_separates_output_and_merge_catalogs() -> None:
    args = build_parser().parse_args(
        [
            "gate",
            "--production-catalog",
            "dev-production.yaml",
            "--template-catalog",
            "train-production.yaml",
            "--requests",
            "dev-requests.yaml",
            "--request-split",
            "dev",
            "--output-catalog",
            "accelerated.yaml",
            "--merge-catalog",
            "dev-production.yaml",
        ]
    )
    assert args.request_split == "dev"
    assert args.template_catalog == [Path("train-production.yaml")]
    assert args.output_catalog == Path("accelerated.yaml")
    assert args.merge_catalog == Path("dev-production.yaml")


def test_only_parameterizable_items_can_materialize(
    real_plan, tmp_path: Path
) -> None:
    exact = next(item for item in real_plan.items if item.category == "exact")
    with pytest.raises(ReuseError, match="only parameterizable"):
        materialize_candidate(
            ReusePlanItem(exact.request, "exact"),
            tmp_path,
        )
