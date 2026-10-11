from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multi_tune_agent.dev_base_case_templates import (
    SELECTED_IDS,
    TRITON_SCALED_SILU_ID,
    coverage_counts,
    load_base_cases,
    materialize_candidate,
)


REQUESTS = Path("/home/danyzhan/phase1_control/splits/v4/generation-requests.yaml")


@pytest.fixture(scope="module")
def cases():
    if not REQUESTS.is_file():
        pytest.skip("frozen split-v4 requests are unavailable")
    return load_base_cases(REQUESTS)


def test_selection_is_exactly_lane_balanced_for_deficit(cases) -> None:
    assert {item.request_id for item in cases} == SELECTED_IDS
    assert coverage_counts(cases) == {"hip": 12, "triton": 3, "total": 15}
    assert all(
        item.request["seed_provenance"]["split_group"] == "dev"
        and item.request["seed_provenance"]["split_version"] == "v4"
        for item in cases
    )


def test_materialized_identity_includes_contract_and_request(cases, tmp_path: Path) -> None:
    item = next(case for case in cases if case.request_id == TRITON_SCALED_SILU_ID)
    draft = materialize_candidate(item, tmp_path)
    expected_id = hashlib.sha256(
        f"{item.contract.contract_hash}:{item.request_id}".encode()
    ).hexdigest()
    assert draft.path.name == expected_id
    assert draft.valid
    metadata = json.loads((draft.path / "metadata.json").read_text(encoding="utf-8"))
    assert "trust" not in metadata
    assert metadata["provenance"]["request_id"] == item.request_id
    assert metadata["provenance"]["contract_hash"] == item.contract.contract_hash
    assert metadata["provenance"]["case_seed"] == item.request["seed_provenance"]


def test_materialization_is_deterministic(cases, tmp_path: Path) -> None:
    item = cases[0]
    first = materialize_candidate(item, tmp_path)
    second = materialize_candidate(item, tmp_path)
    assert first.path == second.path
    for relative in ("kernel.py", "config.yaml", "scripts/task_runner.py", "metadata.json"):
        assert (first.path / relative).read_bytes() == (second.path / relative).read_bytes()


def test_scaled_silu_uses_per_row_split_offsets(cases, tmp_path: Path) -> None:
    item = next(case for case in cases if case.request_id == TRITON_SCALED_SILU_ID)
    draft = materialize_candidate(item, tmp_path)
    kernel = (draft.path / "kernel.py").read_text(encoding="utf-8")
    assert "input_offset = row * (2 * HALF) + column" in kernel
    assert "input_offset + HALF" in kernel


def test_no_held_out_or_train_request_can_enter_selection(tmp_path: Path) -> None:
    if not REQUESTS.is_file():
        pytest.skip("frozen split-v4 requests are unavailable")
    text = REQUESTS.read_text(encoding="utf-8")
    tampered = text.replace("split_group: dev", "split_group: train", 1)
    path = tmp_path / "requests.yaml"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="non-frozen or non-Dev provenance"):
        load_base_cases(path)
