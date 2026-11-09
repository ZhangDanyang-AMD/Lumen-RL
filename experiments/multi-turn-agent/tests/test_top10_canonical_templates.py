from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from multi_tune_agent.top10_canonical_templates import (
    FAIL_CLOSED_REASONS,
    LOCKED_AITER_SHA,
    TOP10_FAMILIES,
    Top10CanonicalError,
    family_counts,
    load_top10_requests,
    materialize_request,
    merge_catalog,
    render_template,
    validate_gang_arguments,
    verify_locked_source,
)

REAL_V4_REQUESTS = Path(
    "/home/danyzhan/phase1_control/splits/v4/generation-requests.yaml"
)


def _document(tmp_path: Path) -> tuple[Path, Path]:
    aiter = tmp_path / "aiter"
    source = aiter / "op_tests" / "locked.py"
    source.parent.mkdir(parents=True)
    source.write_text("# locked source\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    requests = []
    for index, family in enumerate(TOP10_FAMILIES):
        language = "hip" if family == "all_reduce" else "triton"
        shapes = {
            "mha": {"B": 1, "SQ": 8, "SK": 8, "HQ": 2, "HK": 1, "D": 32},
            "mla": {"B": 1, "S": 8, "H": 2, "KV": 32, "ROPE": 8},
            "paged_attention": {
                "B": 1,
                "HQ": 2,
                "HK": 1,
                "BLOCK": 4,
                "S": 8,
                "D": 32,
            },
            "fused_moe": {
                "TOKENS": 8,
                "MODEL": 32,
                "INTER": 64,
                "EXPERTS": 4,
                "TOPK": 2,
            },
            "gemm": {"M": 8, "N": 32, "K": 32},
            "rms_norm": {"M": 8, "N": 32},
            "rope_kv_cache": {"S": 8, "B": 1, "H": 2, "D": 32},
            "blockscale_gemm": {"M": 8, "N": 32, "K": 128},
            "all_reduce": {"TOKENS": 8, "HIDDEN": 32},
            "sampling": {"B": 8, "VOCAB": 128},
        }
        modes = {
            "mha": "causal",
            "mla": "decode",
            "paged_attention": "decode",
            "fused_moe": "silu",
            "gemm": "bf16",
            "rms_norm": "plain",
            "rope_kv_cache": "sbhd",
            "blockscale_gemm": "fp8_block128",
            "all_reduce": "quick",
            "sampling": "top_p",
        }
        requests.append(
            {
                "id": f"top10-{index:02d}",
                "top10_family": family,
                "request": f"Implement {family}",
                "seed_provenance": {
                    "source_sha": LOCKED_AITER_SHA,
                    "source_test_path": "op_tests/locked.py",
                    "source_test_sha256": digest,
                    "split_group": "train",
                    "split_version": "v4",
                    "top10_family": family,
                },
                "recognized_contract": {
                    "top10_family": family,
                    "language": language,
                    "target_gpu": "gfx942",
                    "contract": {
                        "operator": family,
                        "mode": modes[family],
                        "shape": shapes[family],
                        "input_dtype": (
                            "fp8_e4m3fnuz"
                            if family == "blockscale_gemm"
                            else "bf16"
                        ),
                        "output_dtype": (
                            "int32" if family == "sampling" else "bf16"
                        ),
                        **(
                            {
                                "weight_dtype": "fp8_e4m3fnuz",
                                "scale": {
                                    "activation": "block128",
                                    "weight": "block128",
                                },
                            }
                            if family == "blockscale_gemm"
                            else {}
                        ),
                        **({"world_size": 2} if family == "all_reduce" else {}),
                    },
                },
            }
        )
    path = tmp_path / "generation-requests.yaml"
    path.write_text(yaml.safe_dump({"requests": requests}), encoding="utf-8")
    return path, aiter


def test_source_lock_checks_revision_and_artifact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, aiter = _document(tmp_path)
    requests = load_top10_requests(path)
    monkeypatch.setattr(
        "multi_tune_agent.top10_canonical_templates.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=LOCKED_AITER_SHA + "\n"),
    )
    verify_locked_source(aiter, requests)
    (aiter / "op_tests" / "locked.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(Top10CanonicalError, match="hash mismatch"):
        verify_locked_source(aiter, requests)


def test_real_split_v4_schema_and_source_locks_are_consumed() -> None:
    if not REAL_V4_REQUESTS.is_file():
        pytest.skip("real split-v4 generation requests are unavailable")
    requests = load_top10_requests(REAL_V4_REQUESTS)
    assert family_counts(requests) == {
        "all_reduce": 2,
        "blockscale_gemm": 4,
        "fused_moe": 4,
        "gemm": 2,
        "mha": 2,
        "mla": 4,
        "paged_attention": 2,
        "rms_norm": 4,
        "rope_kv_cache": 2,
        "sampling": 4,
        "total": 30,
    }
    assert all(
        request.seed_provenance["split_version"] == "v4"
        and request.seed_provenance["split_group"] in {"train", "dev"}
        for request in requests
    )
    aiter = Path("/home/danyzhan/aiter")
    if aiter.is_dir():
        verify_locked_source(aiter, requests)


def test_family_dispatch_is_exact(tmp_path: Path) -> None:
    path, _ = _document(tmp_path)
    requests = load_top10_requests(path)
    assert {request.family for request in requests} == set(TOP10_FAMILIES)
    assert family_counts(requests) == {
        **{family: 1 for family in TOP10_FAMILIES},
        "total": 10,
    }
    assert all(
        request.language == ("hip" if request.family == "all_reduce" else "triton")
        for request in requests
    )


def test_requested_hip_lane_dispatches_to_hip(tmp_path: Path) -> None:
    path, _ = _document(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    gemm = next(
        row for row in document["requests"] if row["top10_family"] == "gemm"
    )
    gemm["recognized_contract"]["language"] = "hip"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    request = next(item for item in load_top10_requests(path) if item.family == "gemm")
    kernel = render_template(request)["kernel.py"]
    assert "#include <hip/hip_runtime.h>" in kernel
    assert "gemm_kernel" in kernel


def test_generated_bundles_are_real_static_valid_and_untrusted(
    tmp_path: Path,
) -> None:
    path, _ = _document(tmp_path)
    for request in load_top10_requests(path):
        rendered = render_template(request)
        kernel = rendered["kernel.py"]
        runner = rendered["scripts/task_runner.py"]
        metadata = json.loads(rendered["metadata.json"])
        assert "trust" not in metadata
        assert metadata["materialization_status"] == "untrusted_pending_gpu_gate"
        assert "torch.testing.assert_close" in runner
        assert "expected" in runner
        assert all(
            token not in kernel.lower()
            for token in ("aiter", "hipblaslt", "composable_kernel")
        )
        if request.family == "all_reduce":
            assert "#include <rccl/rccl.h>" in kernel
            assert "ncclAllReduce" in kernel
            assert "--offload-arch=gfx942" in kernel
        else:
            assert "@triton.jit" in kernel
        first = materialize_request(request, tmp_path / "candidates")
        second = materialize_request(request, tmp_path / "candidates")
        assert first.valid and second.valid and first.path == second.path


def test_catalog_trust_is_fail_closed_and_merge_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "catalog.yaml"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    contract_hash = "a" * 64
    record = {
        "id": "case",
        "kernel_path": str(bundle),
        "contract_hash": contract_hash,
    }
    metadata_path = bundle / "metadata.json"
    metadata_path.write_text(
        json.dumps({"contract_hash": contract_hash}), encoding="utf-8"
    )
    with pytest.raises(Top10CanonicalError, match="untrusted"):
        merge_catalog(output, [record])
    assert not output.exists()

    metadata_path.write_text(
        json.dumps(
            {
                "contract_hash": contract_hash,
                "trust": {"trusted": True, "contract_hash": contract_hash},
            }
        ),
        encoding="utf-8",
    )
    merge_catalog(output, [record])
    merge_catalog(output, [record])
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"] == [record]
    assert not list(tmp_path.glob(".catalog.yaml.*.tmp"))


@pytest.mark.parametrize(
    ("gpu_ids", "world_size", "rank", "expected"),
    [
        ("1,2", 2, 0, (1, 2)),
        ("1,2,3,4", 4, 3, (1, 2, 3, 4)),
    ],
)
def test_gang_argument_validation_accepts_exact_gpu_sets(
    gpu_ids: str, world_size: int, rank: int, expected: tuple[int, ...]
) -> None:
    assert validate_gang_arguments(gpu_ids, world_size, rank) == expected


@pytest.mark.parametrize(
    ("gpu_ids", "world_size", "rank"),
    [
        ("1", 2, 0),
        ("1,1", 2, 0),
        ("1,2,3", 3, 0),
        ("1,2", 2, 2),
        ("-1,2", 2, 0),
    ],
)
def test_gang_argument_validation_rejects_unsafe_launches(
    gpu_ids: str, world_size: int, rank: int
) -> None:
    with pytest.raises(Top10CanonicalError):
        validate_gang_arguments(gpu_ids, world_size, rank)


def test_non_exact_contracts_have_explicit_fail_closed_reasons() -> None:
    assert FAIL_CLOSED_REASONS == {}
    assert all(reason.strip() for reason in FAIL_CLOSED_REASONS.values())
