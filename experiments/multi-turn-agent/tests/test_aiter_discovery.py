from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from geak_utils.aiter_discovery import (
    HIGH_CONFIDENCE_SCORE,
    AiterDiscoveryIndex,
    AiterQuery,
    discover_aiter,
)


def write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.resolve()


@pytest.fixture
def aiter_tree(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "aiter-checkout"
    paths = {
        "gemm": write(
            root,
            "aiter/ops/gemm/fp8_blockscale_gemm.py",
            "# supported on gfx942\n# per-block weight scale\n",
        ),
        "gemm_test": write(
            root,
            "op_tests/gemm/test_fp8_blockscale_gemm.py",
            "def reference(x, w):\n    return torch.matmul(x, w)\n",
        ),
        "gemm_bench": write(
            root, "benchmarks/gemm/benchmark_fp8_blockscale_gemm.py", ""
        ),
        "gemm_config": write(
            root, "configs/gemm/fp8_blockscale_gemm.yaml", "tile: 64\n"
        ),
        "attention": write(
            root, "aiter/ops/attention/flash_attention.py", "# gfx950\n"
        ),
        "attention_test": write(
            root,
            "op_tests/attention/test_flash_attention.py",
            "expected = F.scaled_dot_product_attention(q, k, v)\n",
        ),
        "moe": write(
            root, "aiter/ops/moe/fused_moe.py", "# generic fused MoE wrapper\n"
        ),
        "moe_test": write(
            root,
            "op_tests/moe/test_fused_moe.py",
            "def naive_moe_reference(x):\n    return x\n",
        ),
        "quant": write(
            root,
            "aiter/ops/quant/per_tensor_quant.py",
            "# per-tensor scale; gfx942\n",
        ),
        "quant_test": write(
            root,
            "op_tests/quant/test_per_tensor_quant.py",
            "def reference_quant(x):\n    return x\n",
        ),
    }
    return root, paths


def test_query_is_frozen_and_normalized() -> None:
    query = AiterQuery(
        operator=" Fused-Attention ",
        input_dtype="float16",
        weight_format=" FP-8 ",
        input_scale_granularity="blockwise",
        block_size=[16, 32],
        architecture="MI355X",
        shapes=[(2, 8, 128, 64)],
        backend=" ROCm ",
    )
    assert query.operator == "fused_attention"
    assert query.input_dtype == "fp16"
    assert query.weight_format == "fp_8"
    assert query.input_scale_granularity == "per_block"
    assert query.block_size == (16, 32)
    assert query.architecture == "gfx950"
    assert query.shapes == ((2, 8, 128, 64),)
    assert query.backend == "rocm"
    with pytest.raises(FrozenInstanceError):
        query.operator = "gemm"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("operator", "expected_key"),
    [
        ("gemm", "gemm"),
        ("attention", "attention"),
        ("moe", "moe"),
    ],
)
def test_discovers_arbitrary_operator_families(
    aiter_tree: tuple[Path, dict[str, Path]],
    operator: str,
    expected_key: str,
) -> None:
    root, paths = aiter_tree
    results = discover_aiter(root, AiterQuery(operator=operator))
    assert results
    assert results[0].wrapper_path == paths[expected_key]
    assert results[0].reference_paths
    assert results[0].high_confidence


def test_scoring_rewards_primary_evidence_more_than_minor_assets(
    aiter_tree: tuple[Path, dict[str, Path]],
) -> None:
    root, paths = aiter_tree
    candidate = AiterDiscoveryIndex(root).search(
        AiterQuery(
            operator="gemm",
            input_format="fp8",
            weight_scale_granularity="per_block",
            architecture="MI300X",
        )
    )[0]
    assert candidate.wrapper_path == paths["gemm"]
    assert candidate.test_paths == (paths["gemm_test"],)
    assert candidate.reference_paths == (paths["gemm_test"],)
    assert candidate.benchmark_paths == (paths["gemm_bench"],)
    assert candidate.config_paths == (paths["gemm_config"],)
    assert candidate.supported_architectures == ("gfx942",)
    assert candidate.score >= HIGH_CONFIDENCE_SCORE
    assert candidate.score <= 100
    assert candidate.reasons[:3] == (
        "operator tokens match wrapper path",
        "dedicated mirrored op test",
        "explicit independent reference in test",
    )


def test_results_are_deterministic_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "aiter"
    second = write(root, "aiter/ops/softmax/z_softmax.py")
    first = write(root, "aiter/ops/softmax/a_softmax.py")
    query = AiterQuery(operator="softmax")
    index = AiterDiscoveryIndex(root)
    results = index.search(query)
    assert results == index.search(query)
    assert [item.wrapper_path for item in results] == [first, second]
    assert isinstance(results, tuple)
    with pytest.raises(FrozenInstanceError):
        results[0].score = 100  # type: ignore[misc]


def test_architecture_and_scale_mismatches_are_hard_rejected(
    aiter_tree: tuple[Path, dict[str, Path]],
) -> None:
    root, _ = aiter_tree
    index = AiterDiscoveryIndex(root)
    assert not index.search(AiterQuery(operator="gemm", architecture="gfx950"))
    assert not index.search(
        AiterQuery(
            operator="quant",
            architecture="gfx942",
            input_scale_granularity="per_token",
        )
    )
    assert index.search(
        AiterQuery(
            operator="quant",
            architecture="gfx942",
            input_scale_granularity="per_tensor",
        )
    )


def test_index_ignores_symlinks_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "aiter"
    root.mkdir()
    outside = write(tmp_path, "outside/aiter/ops/gemm/evil_gemm.py")
    link = root / "aiter"
    try:
        link.symlink_to(outside.parents[3], target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    index = AiterDiscoveryIndex(root)
    assert outside not in index.paths
    assert not index.search(AiterQuery(operator="gemm"))


def test_no_result_and_minimum_score_behavior(tmp_path: Path) -> None:
    root = tmp_path / "aiter"
    write(root, "aiter/ops/gemm/plain_gemm.py")
    index = AiterDiscoveryIndex(root)
    assert index.search(AiterQuery(operator="attention")) == ()
    assert index.search(
        AiterQuery(operator="gemm"), minimum_score=HIGH_CONFIDENCE_SCORE
    ) == ()


def test_root_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AiterDiscoveryIndex(tmp_path / "missing")
    file_path = write(tmp_path, "not-a-root.txt")
    with pytest.raises(ValueError, match="not a directory"):
        AiterDiscoveryIndex(file_path)
