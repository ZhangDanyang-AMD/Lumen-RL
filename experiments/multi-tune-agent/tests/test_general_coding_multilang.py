import pytest

from multi_tune_agent.general_coding_multilang import (
    DATASET_ID,
    DATASET_REVISION,
    SOURCE_ID,
    adapt_multilang_row,
)


def _row():
    return {
        "repo": "BYVoid/OpenCC",
        "instance_id": "BYVoid__OpenCC-1123",
        "base_commit": "a" * 40,
        "patch": "diff --git a/src/a.cpp b/src/a.cpp\n-old\n+new\n",
        "test_patch": "diff --git a/test/a.cpp b/test/a.cpp\n-old\n+new\n",
        "problem_statement": "Fix conversion behavior.",
        "FAIL_TO_PASS": ["Conversion.Fix"],
        "PASS_TO_PASS": ["Conversion.Existing"],
        "rebuild_cmds": ["cmake --build build --parallel"],
        "test_cmds": ["ctest --test-dir build 2>&1 | tee test-output.log"],
        "docker_image": "owner/image",
    }


def test_multilang_adapter_freezes_strict_evidence():
    candidate = adapt_multilang_row(_row())

    assert candidate["source_id"] == SOURCE_ID
    assert candidate["dataset_id"] == DATASET_ID
    assert candidate["dataset_revision"] == DATASET_REVISION
    assert candidate["dataset_license_spdx"] == "Apache-2.0"
    assert candidate["fail_to_pass"] == ["Conversion.Fix"]
    assert candidate["pass_to_pass"] == ["Conversion.Existing"]
    assert candidate["network_policy"] == "disabled"
    assert candidate["gpu_required"] is False


@pytest.mark.parametrize(
    "field",
    [
        "base_commit",
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "rebuild_cmds",
        "test_cmds",
        "docker_image",
    ],
)
def test_multilang_adapter_rejects_missing_strict_gate(field):
    row = _row()
    row[field] = [] if isinstance(row[field], list) else ""

    with pytest.raises(ValueError):
        adapt_multilang_row(row)
