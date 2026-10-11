from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from multi_tune_agent.scaled_quant_gemm_templates import (
    EXPECTED_SHAPES,
    FAMILY_ID,
    CanonicalTemplateError,
    load_family_requests,
    main,
    materialize_request,
    merge_catalog,
    render_template,
    verify_locked_source,
)


REQUESTS = Path(
    "/home/danyzhan/phase1_control/splits/v3/generation-requests.yaml"
)


@pytest.fixture(scope="module")
def family_requests():
    if not REQUESTS.is_file():
        pytest.skip("phase-one generation requests are unavailable")
    return load_family_requests(REQUESTS)


def test_locked_family_has_two_complete_26_shape_lanes(family_requests) -> None:
    assert len(family_requests) == 52
    assert {item.language for item in family_requests} == {"hip", "triton"}
    for language in ("hip", "triton"):
        assert {
            item.shape for item in family_requests if item.language == language
        } == EXPECTED_SHAPES
    assert all(
        item.seed_provenance["contract_family_id"] == FAMILY_ID
        for item in family_requests
    )


def test_locked_aiter_artifact_hashes_are_current() -> None:
    aiter_root = Path("/home/danyzhan/aiter")
    if not aiter_root.is_dir():
        pytest.skip("locked AITER checkout is unavailable")
    verify_locked_source(aiter_root)


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_rendered_kernel_is_real_standalone_and_source_backed(
    family_requests, language: str
) -> None:
    request = next(item for item in family_requests if item.language == language)
    first = render_template(request)
    second = render_template(request)

    assert first == second
    kernel = first["kernel.py"]
    assert "aiter" not in kernel.lower()
    if language == "hip":
        assert "load_inline(" in kernel
        assert "cuda_sources=_HIP_SOURCE" in kernel
        assert "with_cuda=True" in kernel
        assert "#include <hip/hip_runtime.h>" in kernel
        assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in kernel
        assert "scaled_quant_gemm_i8_kernel" in kernel
    else:
        assert "@triton.jit" in kernel
        assert "tl.dot(a, w, out_dtype=tl.int32)" in kernel
    runner = first["scripts/task_runner.py"]
    assert "torch.matmul(a.float(), weight.float().transpose(0, 1))" in runner
    assert ".to(torch.int32)" in runner
    assert "gcnArchName" in runner
    assert 'SUPPORTED_ARCH = "gfx942"' in runner
    metadata = json.loads(first["metadata.json"])
    assert metadata["format"] == "int8"
    assert metadata["layout"] == "TN"
    assert metadata["accum_dtype"] == "int32"
    assert metadata["materialization_status"] == "untrusted_pending_gpu_gate"
    assert "trust" not in metadata
    assert metadata["provenance"]["source_sha"] == (
        "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
    )
    assert len(metadata["provenance"]["source_artifacts"]) == 5


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_materialization_is_static_valid_deterministic_and_untrusted(
    family_requests, tmp_path: Path, language: str
) -> None:
    request = next(item for item in family_requests if item.language == language)
    first = materialize_request(request, tmp_path / "candidates")
    hashes = {
        relative: hashlib.sha256((first.path / relative).read_bytes()).hexdigest()
        for relative in (
            "kernel.py",
            "config.yaml",
            "scripts/task_runner.py",
            "metadata.json",
        )
    }
    second = materialize_request(request, tmp_path / "candidates")

    assert first.path == second.path
    assert first.valid and second.valid
    assert hashes == {
        relative: hashlib.sha256((second.path / relative).read_bytes()).hexdigest()
        for relative in hashes
    }
    metadata = json.loads((first.path / "metadata.json").read_text(encoding="utf-8"))
    assert "trust" not in metadata


def test_contract_drift_is_rejected(tmp_path: Path, family_requests) -> None:
    payload = yaml.safe_load(REQUESTS.read_text(encoding="utf-8"))
    target_id = family_requests[0].request_id
    target = next(item for item in payload["requests"] if item["id"] == target_id)
    target["recognized_contract"]["contract"]["accum_dtype"] = "fp32"
    path = tmp_path / "requests.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(CanonicalTemplateError, match="semantics mismatch"):
        load_family_requests(path)


def test_seven_way_materialize_shard(family_requests, tmp_path: Path, capsys) -> None:
    result = main(
        [
            "materialize",
            "--requests",
            str(REQUESTS),
            "--candidate-root",
            str(tmp_path / "candidates"),
            "--aiter-root",
            "/home/danyzhan/aiter",
            "--shard-index",
            "0",
            "--shard-count",
            "7",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert summary["selected"] == 8
    assert summary["materialized"] == 8
    assert len(list((tmp_path / "candidates").iterdir())) == 8


def test_catalog_refuses_materialized_candidate_without_gpu_trust(
    family_requests, tmp_path: Path
) -> None:
    request = family_requests[0]
    draft = materialize_request(request, tmp_path / "candidates")
    record = {
        "id": request.request_id,
        "kernel_path": str(draft.path),
    }
    with pytest.raises(CanonicalTemplateError, match="untrusted"):
        merge_catalog(tmp_path / "catalog.yaml", [record], None)
    assert not (tmp_path / "catalog.yaml").exists()
