import json
from pathlib import Path

import yaml

from multi_tune_agent.general_coding_finalize import finalize


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_finalize_recursively_aggregates_and_fails_closed_on_package_gate(tmp_path):
    control = tmp_path / "control"
    for index, language in enumerate(("python", "cpp")):
        case = control / f"wave-{index}" / "verified" / f"case-{index}"
        artifacts = control / "source-cache" / f"case-{index}"
        artifacts.mkdir(parents=True)
        (artifacts / "problem.md").write_text(
            f"Fix case {index}.\n", encoding="utf-8"
        )
        (artifacts / "target.patch").write_text(
            f"diff --git a/file-{index} b/file-{index}\n", encoding="utf-8"
        )
        receipt = {
            "container_id": f"target-{index}",
            "network": "none",
            "gpu_visibility": {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            },
            "gpu_devices_mounted": False,
            "targeted_tests": {"ok": True},
            "regression_tests": {"ok": True},
        }
        _write_json(case / "target-receipt.json", receipt)
        _write_json(case / "fresh-receipt.json", {**receipt, "container_id": f"fresh-{index}"})
        _write_json(
            case / "verified.json",
            {
                "case_id": f"case-{index}",
                "status": "verified",
                "source_id": f"source-{index}",
                "dataset_id": "test/replay",
                "dataset_revision": "revision",
                "primary_language": language,
                "primary_task_type": f"task-{index}",
                "upstream_repository": f"https://example.invalid/repo-{index}.git",
                "base_commit": str(index) * 40,
                "problem_id": f"problem-{index}",
                "source_lineage_id": f"lineage-{index}",
                "source_hash": f"source-hash-{index}",
                "normalized_patch_hash": f"patch-{index}",
                "ast_fingerprint": f"ast-{index}",
                "diff_hunk_hash": f"diff-{index}",
                "test_set_hash": f"tests-{index}",
                "problem_statement_path": f"source-cache/case-{index}/problem.md",
                "target_patch_path": f"source-cache/case-{index}/target.patch",
                "license_spdx": "MIT",
                "network_policy": "disabled",
                "gpu_required": False,
                "target_test_receipt": "target-receipt.json",
                "fresh_verify_receipt": "fresh-receipt.json",
                "verified_runtime_seconds": index + 1,
            },
        )
    (control / "quotas.yaml").write_text(
        yaml.safe_dump(
            {
                "source_targets": {"source-0": 1, "source-1": 1},
                "source_language_targets": {
                    "source-0": {"python": 1},
                    "source-1": {"cpp": 1},
                },
                "language_targets": {"python": 1, "cpp": 1},
                "task_type_targets": {"task-0": 1, "task-1": 1},
            }
        ),
        encoding="utf-8",
    )
    (control / "quota-overrides.yaml").write_text("{}\n", encoding="utf-8")
    _write_json(
        control / "exclusion-registry.json",
        {"general_coding_repository_held_out": {"problem_ids": []}},
    )
    (control / "general-coding-dev-reservations.jsonl").write_text("", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    report = finalize(control, empty, empty, empty, target=2)

    assert report["status"] == "exact"
    assert report["verified_pool"] == 2
    assert len((control / "accepted.jsonl").read_text().splitlines()) == 2
    package = json.loads((control / "package-gate-report.json").read_text())
    assert package["status"] == "blocked"
    assert "assistant_loss_tokens present on 0 of 2 rows" in package["blockers"]
