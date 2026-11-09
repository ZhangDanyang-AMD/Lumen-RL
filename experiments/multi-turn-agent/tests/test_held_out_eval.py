from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from multi_tune_agent.held_out_eval import (
    HeldOutError,
    build_control_root,
    deterministic_assign,
    inventory_deficit,
    leakage_audit,
    protected_terms,
    validate_eval_task,
)


def _record(index: int) -> dict:
    suites = ("geak_native", "aiter_derived", "adversarial_boundary")
    lane = "hip_gfx942" if index % 2 == 0 else "triton_gfx942"
    return {
        "task_id": f"held-task-{index:03d}",
        "source_lineage_id": f"lineage-{index:03d}",
        "contract_family_id": f"contract-{index:03d}",
        "implementation_family_id": f"implementation-{index:03d}",
        "lane": lane,
        "source_suite": suites[index % 3],
        "contract": {"operator": f"operator_{index:03d}", "shape": [index + 1]},
        "initial_source": {
            "path": f"initial/{index:03d}",
            "sha256": "a" * 64,
        },
        "protected": {
            "harness": {"sha256": "b" * 64},
            "oracle": {"sha256": "c" * 64},
        },
        "environment_hash": "d" * 64,
        "protected_symbols": [f"held_symbol_{index:03d}"],
    }


def test_deterministic_assignment_meets_all_hard_quotas():
    records = [_record(index) for index in range(120)]
    first = deterministic_assign(records)
    second = deterministic_assign(list(reversed(records)))

    assert first == second
    assert Counter(item["lane"] for item in first) == {
        "hip_gfx942": 60,
        "triton_gfx942": 60,
    }
    assert Counter(item["source_suite"] for item in first) == {
        "geak_native": 40,
        "aiter_derived": 40,
        "adversarial_boundary": 40,
    }
    assert Counter(item["task_type"] for item in first) == {
        "cold_start": 18,
        "profile_guided": 18,
        "direction_conditioned": 54,
        "error_recovery": 18,
        "regression_balance": 12,
    }
    assert all(item["sft_enabled"] is False and item["eval_only"] for item in first)


def test_schema_rejects_training_and_answer_fields():
    task = deterministic_assign([_record(index) for index in range(120)])[0]
    task["output"] = {"patch": "secret"}
    with pytest.raises(HeldOutError, match="forbidden"):
        validate_eval_task(task)

    task.pop("output")
    task["trajectory"] = []
    with pytest.raises(HeldOutError, match="forbidden"):
        validate_eval_task(task)


def test_v4_reservations_report_precise_non_materialized_deficit():
    groups = []
    candidate_counts = [2, 1, 1, 1, 1, 26, 1, 1, 1, 1, 1, 1]
    for index, count in enumerate(candidate_counts):
        lanes = (
            ["hip_gfx942"]
            if index == 0
            else ["triton_gfx942"]
            if index >= 10
            else ["hip_gfx942", "triton_gfx942"]
        )
        groups.append(
            {
                "source_lineage_id": (
                    f"aiter:lineage-{index}" if index < 10 else f"lumen:lineage-{index}"
                ),
                "candidate_ids": [f"candidate-{index}-{item}" for item in range(count)],
                "target_lanes": lanes,
            }
        )

    report = inventory_deficit(groups)

    assert report["status"] == "not_materialized"
    assert report["task_deficit"] == 99
    assert report["materializable_task_deficit"] == 120
    assert report["reservation_evidence"] == {
        "lineage_count": 12,
        "candidate_ref_count": 38,
        "candidate_lane_slot_count": 72,
        "lineage_lane_unit_count": 21,
        "lane_available": {"hip_gfx942": 10, "triton_gfx942": 11},
        "source_suite_available": {"aiter_derived": 19, "unclassified": 2},
    }


def test_joint_leakage_audit_and_protected_terms(tmp_path: Path):
    task = {
        **_record(0),
        "schema_version": "geak_kernel_held_out_eval_v1",
        "split": "held_out",
        "eval_only": True,
        "sft_enabled": False,
        "task_type": "cold_start",
    }
    train = tmp_path / "train.jsonl"
    sample = {
        "sample_id": "train-1",
        "input": {"contract": {"provenance": {"case_seed": {
            "source_lineage_id": task["source_lineage_id"],
        }}}},
        "provenance": {},
    }
    train.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    report = leakage_audit([task], train)

    assert report["status"] == "fail"
    assert report["lineage_contract_symbol_hit_count"] == 1
    assert protected_terms([task]) == [
        "contract-000",
        "held-task-000",
        "held_symbol_000",
        "lineage-000",
    ]


def test_incomplete_inventory_never_materializes_partial_tasks(tmp_path: Path):
    groups = tmp_path / "held_out_groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "source_lineage_id": "aiter:reserved",
                "candidate_ids": ["reserved-kernel"],
                "contract_family_ids": ["reserved-contract"],
                "target_lanes": ["hip_gfx942"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_control_root(groups, tmp_path / "control")

    assert report["status"] == "not_materialized"
    assert not (tmp_path / "control/tasks/kernel.jsonl").exists()
    assert (tmp_path / "control/protected_held_out_terms.json").is_file()
    assert (tmp_path / "control/checksums.sha256").is_file()
