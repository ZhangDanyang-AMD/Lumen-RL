from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from multi_tune_agent.fused_canonical_templates import (
    ACTIVATIONS,
    CanonicalFactoryError,
    LOCKED_AITER_SHA,
    OPERAND_KINDS,
    _selected,
    _trusted_completed_ids,
    coverage_counts,
    load_canonical_requests,
    materialize_candidate,
    merge_promoted_tasks,
)


REQUESTS = Path("/home/danyzhan/phase1_control/splits/v3/generation-requests.yaml")
AITER = Path("/home/danyzhan/aiter")


@pytest.fixture(scope="module")
def canonical_requests():
    if not REQUESTS.is_file() or not AITER.is_dir():
        pytest.skip("locked phase-one manifests are unavailable")
    return load_canonical_requests(REQUESTS, AITER)


def test_locked_source_backed_coverage_is_exact(canonical_requests) -> None:
    assert coverage_counts(canonical_requests) == {
        "fused_mul_add_hip": 6,
        "fused_mul_add_triton": 6,
        "gemm_activation_hip": 6,
        "gemm_activation_triton": 6,
        "total": 24,
    }
    assert len(canonical_requests) == 24
    assert all(
        item.request["seed_provenance"]["source_sha"] == LOCKED_AITER_SHA
        for item in canonical_requests
    )
    assert {
        tuple(item.kernel_contract.shapes[0])
        for item in canonical_requests
        if item.contract["operator"] == "gemm_activation"
        and item.kernel_contract.language == "hip"
    } == {
        (16, 1024, 1024),
        (128, 8192, 512),
        (256, 512, 8192),
        (1024, 1024, 1024),
        (5120, 5120, 5120),
        (8192, 8192, 8192),
    }
    assert {
        tuple(item.kernel_contract.shapes[0])
        for item in canonical_requests
        if item.contract["operator"] == "fused_mul_add"
        and item.kernel_contract.language == "hip"
    } == {
        (1,),
        (8,),
        (500,),
        (10000,),
        (32, 7168),
        (16, 50, 4186),
    }
def test_materializes_all_real_kernels_as_untrusted_deterministic_drafts(
    canonical_requests, tmp_path: Path
) -> None:
    root = tmp_path / "drafts"
    for item in canonical_requests:
        first = materialize_candidate(item, root)
        second = materialize_candidate(item, root)
        assert first.path == second.path
        assert first.valid
        metadata = json.loads((first.path / "metadata.json").read_text(encoding="utf-8"))
        assert "trust" not in metadata
        assert metadata["canonical_status"] == "untrusted_pending_gpu_gate"
        assert metadata["provenance"]["request_id"] == item.request_id
        kernel = (first.path / "kernel.py").read_text(encoding="utf-8")
        runner = (first.path / "scripts/task_runner.py").read_text(encoding="utf-8")
        assert "aiter" not in kernel.lower()
        assert "scaled_quant" not in kernel
        if item.kernel_contract.language == "hip":
            assert "#include <hip/hip_runtime.h>" in kernel
            assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in kernel
            assert "load_inline(" in kernel
            assert "cuda_sources=_SOURCE" in kernel
            assert "--offload-arch=gfx942" in kernel
        else:
            assert "@triton.jit" in kernel
        if item.contract["operator"] == "gemm_activation":
            assert all(name in runner for name in ACTIVATIONS)
            assert "torch.nn.functional.linear" in runner
            assert "torch.nn.functional.gelu" in runner
            assert "torch.nn.functional.silu" in runner
        else:
            assert all(name in runner for name in OPERAND_KINDS)
            assert "return (a * x.float() + b).to(torch.bfloat16)" in runner


def test_source_hashes_match_locked_manifest(canonical_requests) -> None:
    sources = {
        (
            item.request["seed_provenance"]["source_test_path"],
            item.request["seed_provenance"]["source_test_sha256"],
        )
        for item in canonical_requests
    }
    assert len(sources) == 2
    for relative, expected in sources:
        assert hashlib.sha256((AITER / relative).read_bytes()).hexdigest() == expected


def test_rejects_modified_source_provenance(tmp_path: Path) -> None:
    document = yaml.safe_load(REQUESTS.read_text(encoding="utf-8"))
    target = next(
        item
        for item in document["requests"]
        if item["recognized_contract"]["operator"] == "gemm_activation"
    )
    target["seed_provenance"]["source_sha"] = "0" * 40
    altered = tmp_path / "requests.yaml"
    altered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CanonicalFactoryError, match="source_sha"):
        load_canonical_requests(altered, AITER)


def test_seven_way_sharding_is_deterministic_and_complete(canonical_requests) -> None:
    shards = [_selected(canonical_requests, index, 7) for index in range(7)]
    ids = [item.request_id for shard in shards for item in shard]
    assert len(ids) == len(set(ids)) == 24
    assert set(ids) == {item.request_id for item in canonical_requests}
    assert shards == [_selected(canonical_requests, index, 7) for index in range(7)]
    assert sorted(map(len, shards)) == [3, 3, 3, 3, 4, 4, 4]


def test_catalog_refuses_untrusted_records(tmp_path: Path) -> None:
    bundle = tmp_path / "verified"
    bundle.mkdir()
    (bundle / "metadata.json").write_text(
        json.dumps({"canonical_status": "untrusted_pending_gpu_gate"}),
        encoding="utf-8",
    )
    with pytest.raises(CanonicalFactoryError, match="untrusted"):
        merge_promoted_tasks(
            None,
            tmp_path / "catalog.yaml",
            [{"id": "case", "kernel_path": str(bundle)}],
        )


def test_resume_only_accepts_matching_gpu_trust(tmp_path: Path) -> None:
    bundle = tmp_path / "verified"
    bundle.mkdir()
    metadata_path = bundle / "metadata.json"
    contract_hash = "a" * 64
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "case",
                        "contract_hash": contract_hash,
                        "kernel_path": str(bundle),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"contract_hash": contract_hash}), encoding="utf-8"
    )
    assert _trusted_completed_ids(catalog) == set()
    metadata_path.write_text(
        json.dumps(
            {
                "contract_hash": contract_hash,
                "trust": {"trusted": True, "contract_hash": contract_hash},
            }
        ),
        encoding="utf-8",
    )
    assert _trusted_completed_ids(catalog) == {"case"}
