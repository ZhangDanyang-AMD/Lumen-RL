import copy

import pytest

from multi_tune_agent.general_coding_quota_override import (
    BYTEDANCE_REVISION,
    EFFECTIVE_SOURCE_TARGETS,
    OVERRIDE_SCHEMA,
    SOURCE_LANGUAGE_TARGETS,
    adapt_bytedance_row,
    candidate_preflight,
    overlap_reasons,
    select_fallback_candidates,
    swe_smith_admission_decision,
    validate_override_document,
)
from multi_tune_agent.general_coding_replay import (
    PERMISSIVE_SOURCE_LICENSES,
    PHASE1_LANGUAGE_TARGETS,
    PHASE1_TASK_TYPE_TARGETS,
)


def _override():
    return {
        "schema_version": OVERRIDE_SCHEMA,
        "status": "approved_source_infeasibility_override",
        "effective_quotas": {
            "source_targets": EFFECTIVE_SOURCE_TARGETS,
            "source_language_targets": SOURCE_LANGUAGE_TARGETS,
            "language_targets": PHASE1_LANGUAGE_TARGETS,
            "task_type_targets": PHASE1_TASK_TYPE_TARGETS,
        },
    }


def _raw(instance_id="owner__repo-1"):
    return {
        "instance_id": instance_id,
        "org": "owner",
        "repo": "repo",
        "number": 1,
        "base": {"sha": "a" * 40},
        "title": "Fix parser regression",
        "body": "The parser returns the wrong result.",
        "fix_patch": "diff --git a/src/lib.rs b/src/lib.rs\n-old\n+new\n",
        "test_patch": "diff --git a/tests/lib.rs b/tests/lib.rs\n+test\n",
        "f2p_tests": {"parser::regression": {"fix": "PASS"}},
        "p2p_tests": {"parser::existing": {"fix": "PASS"}},
    }


def test_override_preserves_total_language_and_task_quotas():
    effective = validate_override_document(_override())
    assert sum(effective["source_targets"].values()) == 500
    assert sum(effective["language_targets"].values()) == 500
    assert sum(effective["task_type_targets"].values()) == 500
    assert effective["source_language_targets"]["multi_swe_rl_verified"] == {
        "go": 50,
        "javascript_typescript": 75,
    }
    assert effective["source_language_targets"]["bytedance_multi_swe_rl_fallback"] == {
        "cpp": 43,
        "rust": 50,
    }
    assert effective["source_language_targets"]["swe_rebench_v2_fallback"] == {
        "cpp": 10,
    }
    assert effective["source_language_targets"]["swe_bench_live_multilang_cpp"] == {
        "cpp": 12,
    }
    assert effective["source_language_targets"]["menvdata_swe_cpp"] == {"cpp": 10}


def test_override_rejects_quota_drift():
    document = _override()
    document["effective_quotas"]["language_targets"] = {
        **PHASE1_LANGUAGE_TARGETS,
        "rust": 49,
    }
    with pytest.raises(ValueError, match="language_targets"):
        validate_override_document(document)


def test_bytedance_adapter_retains_executable_evidence_without_verifying():
    row = adapt_bytedance_row(
        _raw(),
        "rust",
        source_file="rust/owner__repo.jsonl",
        source_file_sha256="b" * 64,
    )
    assert row["dataset_revision"] == BYTEDANCE_REVISION
    assert row["base_commit"] == "a" * 40
    assert row["fail_to_pass"] == ["parser::regression"]
    assert row["pass_to_pass"] == ["parser::existing"]
    assert row["verified"] is False
    assert row["local_replay_passes"] == 0
    assert row["task_type_evidence"]["method"] == "title_intent_and_patch_structure_v2"


def test_boost_software_license_is_explicitly_permissive():
    assert "BSL-1.0" in PERMISSIVE_SOURCE_LICENSES


def test_adapter_rejects_nonimmutable_parent_and_missing_tests():
    raw = _raw()
    raw["base"] = {"sha": "main"}
    with pytest.raises(ValueError, match="exact base commit"):
        adapt_bytedance_row(raw, "cpp", source_file="x", source_file_sha256="b" * 64)
    raw = _raw()
    raw["f2p_tests"] = {}
    with pytest.raises(ValueError, match="patch/test evidence"):
        adapt_bytedance_row(raw, "cpp", source_file="x", source_file_sha256="b" * 64)


def test_overlap_checks_repository_problem_lineage_patch_and_tests():
    row = adapt_bytedance_row(
        _raw(),
        "rust",
        source_file="x",
        source_file_sha256="b" * 64,
    )
    dimensions = {
        "repository": {"https://github.com/owner/repo"},
        "problem": {row["problem_id"]},
        "lineage": {row["source_lineage_id"]},
        "patch": {row["normalized_patch_hash"]},
        "test": {row["test_set_hash"]},
    }
    assert overlap_reasons(row, dimensions) == [
        "lineage",
        "patch",
        "problem",
        "repository",
        "test",
    ]


def test_deterministic_selector_enforces_language_and_repository_caps():
    rows = []
    for language, target in (("cpp", 4), ("rust", 3)):
        for index in range(target + 2):
            row = adapt_bytedance_row(
                _raw(f"owner{index // 2}__repo-{index}"),
                language,
                source_file="x",
                source_file_sha256="b" * 64,
            )
            row["upstream_repository"] = (
                f"https://github.com/{language}-owner{index // 2}/repo.git"
            )
            rows.append(row)
    first, report = select_fallback_candidates(
        rows, {"cpp": 4, "rust": 3}, repository_cap=2
    )
    second, _ = select_fallback_candidates(
        reversed(copy.deepcopy(rows)), {"cpp": 4, "rust": 3}, repository_cap=2
    )
    assert report["status"] == "exact"
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert max(report["selected_by_repository"].values()) <= 2


def test_preflight_never_counts_candidate_as_verified():
    row = adapt_bytedance_row(
        _raw(),
        "rust",
        source_file="x",
        source_file_sha256="b" * 64,
    )
    row.update(
        {
            "license_spdx": "MIT",
            "license_evidence_path": "licenses/owner__repo.LICENSE",
            "container_image": {
                "repo_digest": "example/image@sha256:" + "c" * 64,
            },
        }
    )
    report = candidate_preflight([row], _override())
    assert report["status"] == "ready_for_local_double_replay"
    assert report["verified_rows"] == 0
    assert report["required_local_replay_passes"] == 2


def test_swe_smith_rejects_pristine_head_as_buggy_parent_commit():
    row = {
        "image_name": "swebench/swesmith.x86_64.owner_repo.snapshot",
        "patch": "--- a/source.cpp\n+++ b/source.cpp\n-old\n+bug\n",
    }
    probe = {
        "image_head": "a" * 40,
        "repo_digest": "swebench/image@sha256:" + "b" * 64,
        "patch_applies_to_image_head": True,
        "bug_present_only_after_patch": True,
        "inverse_patch_restores_tests": True,
        "license_evidence_sha256": "c" * 64,
    }

    decision = swe_smith_admission_decision(row, probe)

    assert decision["status"] == "rejected_no_exact_buggy_parent_commit"
    assert decision["pristine_head_recovered"] is True
    assert decision["executable_buggy_parent_is_git_commit"] is False


def test_swe_smith_admission_requires_immutable_image_and_full_head():
    decision = swe_smith_admission_decision(
        {"image_name": "image", "patch": "diff"},
        {
            "image_head": "short",
            "repo_digest": "image:latest",
            "patch_applies_to_image_head": True,
            "bug_present_only_after_patch": True,
            "inverse_patch_restores_tests": True,
            "license_evidence_sha256": "c" * 64,
        },
    )

    assert decision["status"] == "rejected_incomplete_preflight"
    assert decision["missing_or_invalid"] == ["image_head", "repo_digest"]
