from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from multi_tune_agent.candidate_expansion import (
    LOCKED_AITER_SHA,
    generate_document,
    main,
    merge_documents,
)
from multi_tune_agent.phase1_splits import REQUIRED_SOURCE_FIELDS, promote_requests


AITER_ROOT = Path("/home/danyzhan/aiter")


@pytest.fixture(scope="module")
def generated():
    return generate_document(AITER_ROOT)


def test_generates_large_explicit_unvalidated_pool(generated):
    candidates = generated["candidates"]
    assert len(candidates) >= 100
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    assert len({candidate["source_lineage_id"] for candidate in candidates}) >= 8
    assert all(candidate["status"] == "extracted_candidate" for candidate in candidates)
    assert all(candidate["gpu_validated"] is False for candidate in candidates)
    assert all(candidate["lineage"]["no_cross_product"] is True for candidate in candidates)
    assert all(
        candidate["target_lanes"] == ["hip_gfx942", "triton_gfx942"]
        for candidate in candidates
    )


def test_every_candidate_has_locked_source_and_explicit_contract(generated):
    for candidate in generated["candidates"]:
        source = candidate["source"]
        assert source["git_sha"] == LOCKED_AITER_SHA
        assert all(source[field] for field in REQUIRED_SOURCE_FIELDS)
        assert "path" not in source
        assert source["line"] >= 1
        assert len(source["sha256"]) == 64
        assert candidate["priority"] in {"P0", "P1"}
        assert candidate["contract_family_id"].startswith("CF-GFX942-AUTO-")
        assert candidate["oracle"]["tier"] == "pure_torch"
        assert candidate["oracle"]["method"]
        assert candidate["shape_cluster"]
        assert candidate["contract"]["operator"]
        assert candidate["contract"]["shape"]
        assert candidate["architecture_guard"]["target"] == "gfx942"


def test_tuned_csv_rows_are_filtered_to_gfx942(generated):
    rows = [
        candidate
        for candidate in generated["candidates"]
        if candidate["lineage"]["kind"] == "gfx942_tuned_csv_row"
    ]
    assert len(rows) == 26
    assert all(
        candidate["architecture_guard"]["evidence"]["value"] == "gfx942"
        for candidate in rows
    )


def test_generation_is_deterministic(generated):
    repeated = generate_document(AITER_ROOT)
    assert yaml.safe_dump(repeated, sort_keys=False) == yaml.safe_dump(
        generated, sort_keys=False
    )


def test_merge_preserves_existing_candidates_and_ids(generated):
    original = {
        "schema_version": "existing",
        "candidates": [
            {"id": "existing-7", "status": "registered_seed", "custom": {"keep": True}},
            deepcopy(generated["candidates"][0]),
        ],
    }
    before = deepcopy(original["candidates"])
    merged = merge_documents(original, generated)
    assert merged["candidates"][:2] == before
    assert merged["candidates"][0]["id"] == "existing-7"
    assert len(merged["candidates"]) == len(generated["candidates"]) + 1


def test_generated_candidates_are_consumable_by_split_promotion(generated):
    groups = [
        {
            "source_lineage_id": lineage,
            "split": "train",
        }
        for lineage in sorted(
            {candidate["source_lineage_id"] for candidate in generated["candidates"]}
        )
    ]
    requests, report = promote_requests(generated, groups)
    assert len(requests) == 2 * len(generated["candidates"])
    assert {item["status"] for item in report} == {
        "promoted_to_generation_requests"
    }
    assert all(
        request["seed_provenance"]["gpu_verification_status"] == "not_run"
        for request in requests
    )


def test_cli_writes_yaml(tmp_path):
    output = tmp_path / "expanded.yaml"
    assert main(["--aiter-root", str(AITER_ROOT), "--output", str(output)]) == 0
    document = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert document["policy"]["generated_candidates_are_gpu_validated"] is False
    assert len(document["candidates"]) >= 100
