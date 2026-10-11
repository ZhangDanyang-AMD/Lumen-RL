from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_tune_agent.general_coding_held_out import (
    HeldOutInventoryError,
    build_inventory,
    export_exclusion_component,
    package_private_domain,
    task_id_for,
    update_replay_registry,
    update_replay_registry_from_public_held_out,
    validate_private_candidate,
)


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _candidate(root: Path, index: int = 0, language: str = "python") -> dict:
    private = root / "private"
    repository = f"https://github.com/example/repo-{index}.git"
    commit = f"{index + 1:040x}"
    lineage = f"fresh-mutation:example/repo-{index}:{index:04d}"
    mutation_id = f"mutation-{index:04d}"
    license_path = private / f"license-{index}.txt"
    hidden_path = private / f"hidden-{index}.tar"
    oracle_path = private / f"oracle-{index}.patch"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text(
        json.dumps(
            {
                "repository": repository,
                "base_commit": commit,
                "spdx": "MIT",
                "license_blob_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    hidden_path.write_bytes(f"hidden-{index}".encode())
    oracle_path.write_bytes(f"oracle-{index}".encode())
    receipt = {
        "baseline": {"passed": True, "returncode": 0},
        "defect": {"passed": False, "returncode": 1},
        "oracle": {"passed": True, "returncode": 0},
        "network": "disabled",
        "cpu_only": True,
        "oracle_access": {"authorized": True, "scope": "eval_only"},
    }
    receipt_path = Path(_json(private / f"receipt-{index}.json", receipt))
    return {
        "task_id": task_id_for(repository, commit, lineage, mutation_id),
        "repository": repository,
        "base_commit": commit,
        "source_lineage_id": lineage,
        "primary_language": language,
        "mutation_id": mutation_id,
        "license": {
            "spdx": "MIT",
            "evidence_path": str(license_path),
        },
        "problem_contract": {
            "title": "Repair the documented behavior.",
            "description": "Make the repository satisfy the stated invariant.",
            "allowed_paths": ["src/value.py"],
            "public_test_command": "python3 -m pytest -q",
        },
        "parent_source_refs": [
            {"path": "src/value.py", "sha256": "a" * 64},
        ],
        "protected": {
            "hidden_tests_path": str(hidden_path),
            "oracle_path": str(oracle_path),
            "validation_receipt_path": str(receipt_path),
        },
        "evaluation": {
            "command": ["python3", "-m", "pytest", "-q"],
            "timeout_seconds": 300,
            "network": "disabled",
            "cpu_only": True,
        },
    }


def test_private_candidate_requires_passing_baseline_failing_defect_and_oracle(tmp_path):
    candidate = _candidate(tmp_path)
    frozen = validate_private_candidate(candidate, tmp_path / "private")

    assert frozen["reservation"]["task_id"] == candidate["task_id"]
    assert "hidden_tests_path" not in json.dumps(frozen["reservation"])
    assert len(frozen["reservation"]["protected"]["hidden_tests_sha256"]) == 64
    assert len(frozen["reservation"]["protected"]["oracle_access_receipt_sha256"]) == 64

    receipt = Path(candidate["protected"]["validation_receipt_path"])
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["defect"]["passed"] = True
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(HeldOutInventoryError, match="defect must fail"):
        validate_private_candidate(candidate, tmp_path / "private")


def test_build_inventory_never_reserves_invalid_or_partial_tasks(tmp_path):
    candidate = _candidate(tmp_path)
    Path(candidate["protected"]["oracle_path"]).unlink()
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

    report = build_inventory(source, tmp_path / "private", tmp_path / "control")

    assert report["materialized"] == 0
    assert report["reserved"] == 0
    assert report["deficit"] == 40
    assert not (tmp_path / "control/reservations.jsonl").read_text().strip()


def test_exclusion_export_and_private_package_are_hash_only(tmp_path):
    languages = ("python", "cpp", "go", "javascript_typescript", "rust")
    rows = [_candidate(tmp_path, index, languages[index % 5]) for index in range(40)]
    source = tmp_path / "candidates.jsonl"
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    control = tmp_path / "control"
    report = build_inventory(source, tmp_path / "private", control)
    component = export_exclusion_component(control / "reservations.jsonl", control)
    package = package_private_domain(control, tmp_path / "private", tmp_path / "package")

    assert report["status"] == "ready"
    assert report["language_distribution"] == {language: 8 for language in languages}
    assert component["task_count"] == 40
    assert package["task_count"] == 40
    public_text = (control / "reservations.jsonl").read_text(encoding="utf-8")
    assert "oracle_path" not in public_text
    assert "hidden_tests_path" not in public_text
    assert "oracle-0" not in public_text
    exclusion_text = (control / "exclusion-component.json").read_text(encoding="utf-8")
    assert "problem_contract" not in exclusion_text
    assert "hidden_tests_sha256" in exclusion_text


def test_incomplete_component_cannot_clear_replay_blocker(tmp_path):
    candidate = _candidate(tmp_path)
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    control = tmp_path / "control"
    build_inventory(source, tmp_path / "private", control)
    export_exclusion_component(control / "reservations.jsonl", control)
    registry = tmp_path / "registry.json"
    original = {
        "blockers": [{"id": "general_coding_repository_held_out"}],
        "pending": ["general_coding_repository_held_out"],
    }
    registry.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(HeldOutInventoryError, match="incomplete"):
        update_replay_registry(registry, control / "exclusion-component.json")

    assert json.loads(registry.read_text(encoding="utf-8")) == original


def test_public_component_replaces_only_repository_held_out_blocker(tmp_path):
    entries = []
    for index in range(40):
        entries.append(
            {
                "task_id": f"gc-ho-public-{index:024x}",
                "problem_id": f"problem-{index}",
                "repository": f"https://github.com/example/repo-{index}",
                "base_commit": f"{index + 1:040x}",
                "source_lineage_id": f"public-lineage-{index}",
                "primary_language": "cpp",
                "solution_patch_sha256": f"{index + 1:064x}",
                "normalized_solution_patch_sha256": f"{index + 101:064x}",
                "test_patch_sha256": f"{index + 201:064x}",
                "oracle_results_sha256": f"{index + 301:064x}",
                "container_image_repo_digest": (
                    f"mswebench/example_m_repo-{index}@sha256:{index + 401:064x}"
                ),
            }
        )
    component = {
        "complete": True,
        "task_count": 40,
        "component_sha256": "a" * 64,
        "source": {"sha256": "b" * 64},
        "entries": entries,
    }
    component_path = tmp_path / "component.json"
    component_path.write_text(json.dumps(component), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "blockers": [
                    {"id": "general_coding_repository_held_out"},
                    {"id": "kernel_dev_held_out"},
                ],
                "pending": [
                    "general_coding_repository_held_out",
                    "kernel_dev_held_out",
                ],
                "problem_ids": ["existing"],
                "repositories": [],
                "source_lineages": [],
                "normalized_patch_hashes": [],
            }
        ),
        encoding="utf-8",
    )

    registry = update_replay_registry_from_public_held_out(
        registry_path, component_path
    )

    assert registry["pending"] == ["kernel_dev_held_out"]
    assert registry["general_coding_repository_held_out"]["classification"] == (
        "public_eval_only_repository_held_out"
    )
    assert registry["general_coding_repository_held_out"][
        "base_model_pretraining_exposure"
    ] == "possible"
    assert len(registry["general_coding_repository_held_out"]["task_ids"]) == 40
