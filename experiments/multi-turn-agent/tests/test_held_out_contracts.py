from __future__ import annotations

from collections import Counter
from pathlib import Path

from multi_tune_agent.held_out_contracts import (
    build_candidate_inventory,
    inventory_quota_report,
    split_overlap_report,
)


def test_contract_matrix_builds_complete_balanced_inventory(tmp_path: Path):
    records = build_candidate_inventory(tmp_path)
    report = inventory_quota_report(records)

    assert report["task_count"] == 120
    assert report["lanes"] == {"hip_gfx942": 60, "triton_gfx942": 60}
    assert report["source_suites"] == {
        "adversarial_boundary": 40,
        "aiter_derived": 40,
        "geak_native": 40,
    }
    assert set(report["top10_families"]) == {
        "all_reduce",
        "blockscale_gemm",
        "fused_moe",
        "gemm",
        "mha",
        "mla",
        "paged_attention",
        "rms_norm",
        "rope_kv_cache",
        "sampling",
    }
    assert Counter(report["top10_families"].values()) == {12: 10}
    assert all(
        (tmp_path / item["initial_source"]["path"]).is_file() for item in records
    )
    assert all("task_type" not in item for item in records)


def test_split_overlap_report_is_defect_first(tmp_path: Path):
    records = build_candidate_inventory(tmp_path)
    clean = split_overlap_report(records, [], [])
    assert clean["status"] == "pass"

    collision = {
        "source_lineage_id": records[0]["source_lineage_id"],
        "contract_family_ids": [records[1]["contract_family_id"]],
        "candidate_ids": records[2]["protected_symbols"],
    }
    report = split_overlap_report(records, [collision], [])
    assert report["status"] == "fail"
    assert report["overlap_count"] == 3
    assert {item["kind"] for item in report["hits"]} == {
        "lineage",
        "contract",
        "symbol",
    }
