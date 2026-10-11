import pytest

from multi_tune_agent.general_coding_rebench import (
    DATASET_ID,
    DATASET_REVISION,
    SOURCE_ID,
    adapt_rebench_cpp_row,
)


def _row():
    return {
        "base_commit": "a" * 40,
        "created_at": "2025-01-01",
        "image_name": "docker.io/swerebenchv2/owner-repo:1-aaaaaaa",
        "instance_id": "owner__repo-1",
        "interface": "",
        "language": "cpp",
        "license": "MIT",
        "patch": "diff --git a/src/a.cpp b/src/a.cpp\n-old\n+new\n",
        "pr_description": "",
        "problem_statement": "Fix parser behavior.",
        "repo": "owner/repo",
        "test_patch": "diff --git a/tests/a.cpp b/tests/a.cpp\n-old test\n+new test\n",
        "FAIL_TO_PASS": ["Parser.Fix"],
        "PASS_TO_PASS": ["Parser.Existing"],
        "install_config": {"test_cmd": "ctest --test-dir build"},
        "meta": {},
    }


def test_rebench_adapter_freezes_required_executable_evidence():
    candidate = adapt_rebench_cpp_row(_row(), source_sha256="b" * 64)

    assert candidate["source_id"] == SOURCE_ID
    assert candidate["dataset_id"] == DATASET_ID
    assert candidate["dataset_revision"] == DATASET_REVISION
    assert candidate["base_commit"] == "a" * 40
    assert candidate["fail_to_pass"] == ["Parser.Fix"]
    assert candidate["pass_to_pass"] == ["Parser.Existing"]
    assert candidate["test_command"] == "ctest --test-dir build"
    assert candidate["verified"] is False
    assert candidate["local_replay_passes"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_commit", "main"),
        ("patch", ""),
        ("test_patch", ""),
        ("FAIL_TO_PASS", []),
        ("PASS_TO_PASS", []),
        ("image_name", ""),
    ],
)
def test_rebench_adapter_rejects_missing_gates(field, value):
    row = _row()
    row[field] = value

    with pytest.raises(ValueError):
        adapt_rebench_cpp_row(row, source_sha256="b" * 64)
