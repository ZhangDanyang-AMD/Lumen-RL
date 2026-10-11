import json
from types import SimpleNamespace

import pytest

from multi_tune_agent import general_coding_bytedance as bytedance


def _row(repository="bitcoin/bitcoin", instance_id="bitcoin__bitcoin-19740"):
    return {
        "base_commit": "a" * 40,
        "case_id": f"gc-replay-bytedance_multi_swe_rl_fallback-{instance_id}",
        "container_image": {
            "manifest_digest": "sha256:" + "b" * 64,
            "repo_digest": "mswebench/example@sha256:" + "b" * 64,
            "repo_tag": "mswebench/example:pr-1",
        },
        "dataset_id": bytedance.DATASET_ID,
        "dataset_revision": bytedance.DATASET_REVISION,
        "dataset_row_id": instance_id,
        "gpu_required": False,
        "license_evidence_path": "license.txt",
        "license_spdx": "MIT",
        "network_policy": "disabled_for_verification",
        "primary_language": "cpp",
        "primary_task_type": "repository_bug_fix_or_test_repair",
        "problem_id": instance_id,
        "problem_statement": "Fix the regression.",
        "source_id": bytedance.SOURCE_ID,
        "source_lineage_id": f"lineage:{instance_id}",
        "target_patch": "diff --git a/src/a.cpp b/src/a.cpp\n-old\n+new\n",
        "test_patch": "diff --git a/tests/a.cpp b/tests/a.cpp\n-old\n+new\n",
        "upstream_repository": f"https://github.com/{repository}.git",
    }


@pytest.mark.parametrize(
    ("repository", "instance_id", "expected"),
    [
        ("bitcoin/bitcoin", "bitcoin__bitcoin-19740", "./configure"),
        ("halide/Halide", "halide__Halide-5135", "TARGET_WEBASSEMBLY=OFF"),
        ("yhirose/cpp-httplib", "yhirose__cpp-httplib-755", "cd test"),
        ("catchorg/Catch2", "catchorg__Catch2-1306", "CATCH_DEVELOPMENT_BUILD"),
    ],
)
def test_derive_test_command_uses_patch_free_run_script(
    repository, instance_id, expected
):
    command = bytedance.derive_test_command(_row(repository, instance_id))

    assert expected in command
    assert "git apply" not in command
    assert "/home/test.patch" not in command


def test_prepare_sets_strict_container_policy(tmp_path, monkeypatch):
    source = (
        tmp_path
        / "bytedance-multi-swe-rl-fallback-candidate-source-manifest.jsonl"
    )
    source.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        bytedance, "derive_test_command", lambda row: "cd /home/bitcoin\nmake check"
    )
    monkeypatch.setattr(
        bytedance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(
        bytedance,
        "_docker_image_identity",
        lambda image: {
            "repo_tag": image,
            "repo_digest": "mswebench/example@sha256:" + "b" * 64,
            "manifest_digest": "sha256:" + "b" * 64,
            "image_id": "sha256:" + "c" * 64,
            "working_dir": "/testbed",
        },
    )

    report = bytedance.prepare_bytedance_cpp_images(tmp_path)
    prepared = json.loads(
        (tmp_path / "bytedance-cpp-executable-manifest.jsonl").read_text()
    )

    assert report["prepared"] == 1
    assert prepared["network_policy"] == "disabled"
    assert prepared["gpu_required"] is False
    assert prepared["container_workdir"] == "/home/bitcoin"
    assert prepared["verification_backend"] == "docker"
    assert prepared["full_regression_command"] == "true"
    assert prepared["allowed_paths"] == ["src/a.cpp"]
