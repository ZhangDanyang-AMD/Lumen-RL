import pytest

from multi_tune_agent.general_coding_menvdata import (
    DATASET_ID,
    DATASET_REVISION,
    REVIEWED_EXECUTIONS,
    SOURCE_FILE_SHA256,
    SOURCE_ID,
    adapt_menvdata_cpp_row,
    derive_execution_commands,
)


def _row():
    return {
        "repo": "CLIUtils/CLI11",
        "instance_id": "CLIUtils__CLI11-926",
        "base_commit": "a" * 40,
        "patch": "diff --git a/include/a.hpp b/include/a.hpp\n-old\n+new\n",
        "test_patch": "diff --git a/tests/a.cpp b/tests/a.cpp\n-old\n+new\n",
        "problem_statement": "Fix validated environment options.",
        "language": "C++",
        "image_name": "swe-images-cpp:cliutils-cli11-pr-926",
        "env_setup_script": "#!/bin/bash\ncmake -S . -B build\n",
        "eval_script": "#!/bin/bash\n./build/tests/HelpTest\n",
    }


def test_adapter_pins_source_patches_and_cpu_offline_policy():
    candidate = adapt_menvdata_cpp_row(_row())

    assert candidate["source_id"] == SOURCE_ID
    assert candidate["dataset_id"] == DATASET_ID
    assert candidate["dataset_revision"] == DATASET_REVISION
    assert candidate["source_file_sha256"] == SOURCE_FILE_SHA256
    assert candidate["target_patch_sha256"]
    assert candidate["test_patch_sha256"]
    assert candidate["claimed_upstream_license_spdx"] == "BSD-3-Clause"
    assert candidate["image_name"].startswith("mcatwj/")
    assert candidate["network_policy"] == "disabled"
    assert candidate["gpu_required"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language", "C"),
        ("base_commit", "main"),
        ("patch", ""),
        ("test_patch", ""),
        ("env_setup_script", ""),
        ("eval_script", ""),
        ("image_name", ""),
    ],
)
def test_adapter_rejects_missing_strict_evidence(field, value):
    row = _row()
    row[field] = value

    with pytest.raises(ValueError):
        adapt_menvdata_cpp_row(row)


def test_adapter_rejects_dataset_license_as_repository_evidence():
    row = _row()
    row["repo"] = "unknown/apache-dataset-only"

    with pytest.raises(ValueError, match="upstream repository"):
        adapt_menvdata_cpp_row(row)


def test_reviewed_preflight_commands_are_targeted_and_offline():
    candidate = adapt_menvdata_cpp_row(_row())
    candidate["eval_script_sha256"] = REVIEWED_EXECUTIONS[
        "CLIUtils__CLI11-926"
    ]["eval_script_sha256"]
    commands = derive_execution_commands(candidate)

    assert "HelpTest" in commands["targeted_test_command"]
    assert "SubcommandTest" in commands["targeted_test_command"]
    assert "curl" not in "\n".join(commands.values())
    assert "git apply" not in "\n".join(commands.values())


def test_unreviewed_eval_script_fails_closed():
    candidate = adapt_menvdata_cpp_row(_row())
    candidate["dataset_row_id"] = "CLIUtils__CLI11-999"

    with pytest.raises(ValueError, match="source-specific command review"):
        derive_execution_commands(candidate)


@pytest.mark.parametrize(
    "instance_id",
    [
        "CLIUtils__CLI11-421",
        "CLIUtils__CLI11-1203",
        "CLIUtils__CLI11-370",
        "CLIUtils__CLI11-1199",
        "CLIUtils__CLI11-1058",
        "ArthurSonzogni__FTXUI-121",
        "ArthurSonzogni__FTXUI-298",
        "ArthurSonzogni__FTXUI-755",
        "CrowCpp__Crow-897",
        "CrowCpp__Crow-918",
        "BehaviorTree__BehaviorTree.CPP-424",
        "BehaviorTree__BehaviorTree.CPP-885",
        "NVIDIA__stdexec-744",
        "Tencent__rapidjson-2207",
        "FastLED__FastLED-1842",
        "OpenNMT__CTranslate2-898",
        "KhronosGroup__SPIRV-Tools-5025",
        "KhronosGroup__glslang-4005",
    ],
)
def test_additional_reviewed_commands_are_pinned_targeted_and_offline(instance_id):
    reviewed = REVIEWED_EXECUTIONS[instance_id]
    row = {
        "dataset_row_id": instance_id,
        "eval_script_sha256": reviewed["eval_script_sha256"],
    }

    commands = derive_execution_commands(row)
    combined = "\n".join(commands.values())

    assert commands["full_regression_command"] == "true"
    assert commands["targeted_test_command"] != "true"
    assert not any(
        token in combined
        for token in ("curl", "wget", "apt-get", "pip ", "git ", "docker ")
    )


def test_reviewed_eval_script_hash_drift_fails_closed():
    row = {
        "dataset_row_id": "CrowCpp__Crow-897",
        "eval_script_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="hash drifted"):
        derive_execution_commands(row)


def test_remaining_review_wave_spans_distinct_permissive_repositories():
    selected = {
        "NVIDIA__stdexec-744",
        "Tencent__rapidjson-2207",
        "FastLED__FastLED-1842",
        "OpenNMT__CTranslate2-898",
        "KhronosGroup__SPIRV-Tools-5025",
        "KhronosGroup__glslang-4005",
    }

    assert selected <= REVIEWED_EXECUTIONS.keys()
    assert len(selected) == 6
