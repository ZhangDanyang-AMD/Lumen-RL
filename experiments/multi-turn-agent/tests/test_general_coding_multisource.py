import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_tune_agent import general_coding_multisource as multisource
from multi_tune_agent.general_coding_replay import canonical_json, sha256_bytes


def _raw(instance_id="beego__beego-3455"):
    return {
        "instance_id": instance_id,
        "lang": "go",
        "f2p_tests": {"name": ["TestFixed"], "test": ["FAIL"], "fix": ["PASS"]},
        "p2p_tests": {"name": ["TestExisting"], "test": ["PASS"], "fix": ["PASS"]},
    }


def _row(instance_id="beego__beego-3455", repository="beego/beego"):
    raw = _raw(instance_id)
    target = "diff --git a/a.go b/a.go\n-old\n+new\n"
    tests = "diff --git a/a_test.go b/a_test.go\n-old\n+new\n"
    digest = "sha256:" + "b" * 64
    return {
        "base_commit": "a" * 40,
        "case_id": f"gc-replay-multi_swe_rl_verified-{instance_id}",
        "container_image": {
            "manifest_digest": digest,
            "repo_digest": f"mswebench/beego_m_beego@{digest}",
            "repo_tag": "mswebench/beego_m_beego:pr-3455",
        },
        "dataset_id": multisource.MULTISWE_DATASET_ID,
        "dataset_revision": multisource.MULTISWE_VERIFIED_REVISION,
        "dataset_row_id": instance_id,
        "license_evidence_path": "license.txt",
        "license_spdx": "Apache-2.0",
        "primary_language": "go",
        "primary_task_type": "repository_bug_fix_or_test_repair",
        "problem_id": instance_id,
        "problem_statement": "Fix it.",
        "source_id": multisource.MULTISWE_SOURCE_ID,
        "source_lineage_id": f"lineage:{instance_id}",
        "source_record_sha256": sha256_bytes(canonical_json(raw).encode()),
        "target_patch": target,
        "target_patch_sha256": sha256_bytes(target.encode()),
        "test_patch": tests,
        "test_patch_sha256": sha256_bytes(tests.encode()),
        "upstream_adapter_revision": "24f493f8a103e72312ded4f6b9c89f081d69cb09",
        "upstream_repository": f"https://github.com/{repository}.git",
    }


def _write_source(tmp_path: Path, row):
    (tmp_path / "license.txt").write_text("Apache License", encoding="utf-8")
    (tmp_path / "multi-swe-verified-candidate-source-manifest.jsonl").write_text(
        canonical_json(row) + "\n", encoding="utf-8"
    )
    cache = (
        tmp_path
        / "source-cache"
        / "multi-swe-rl-verified"
        / multisource.MULTISWE_VERIFIED_REVISION
    )
    cache.mkdir(parents=True)
    for language in ("go", "js", "ts"):
        rows = [{"row": _raw()}] if language == "go" else []
        (cache / f"filter-{language}-00000-00100.json").write_text(
            json.dumps({"rows": rows}), encoding="utf-8"
        )


def _identity(row):
    return {
        "repo_tag": row["container_image"]["repo_tag"],
        "repo_digest": row["container_image"]["repo_digest"],
        "image_id": "sha256:" + "c" * 64,
        "working_dir": "/home/beego",
    }


def test_prepare_freezes_strict_executable_evidence(tmp_path, monkeypatch):
    row = _row()
    _write_source(tmp_path, row)
    monkeypatch.setattr(
        multisource,
        "derive_test_command",
        lambda candidate, language: (
            "#!/bin/bash\nset -e\ncd /home/beego\ngo test ./...",
            "/home/beego",
        ),
    )
    monkeypatch.setattr(
        multisource, "_docker_image_identity", lambda image: _identity(row)
    )

    report = multisource.prepare_executable_candidates(
        tmp_path, "multiswe", max_pulls=0
    )
    prepared = json.loads(
        (tmp_path / "multi-swe-verified-executable-manifest.jsonl").read_text()
    )

    assert report["prepared_in_scope"] == 1
    assert report["prepared_scope_case_ids"] == [row["case_id"]]
    assert report["pulled_by_digest"] == 0
    assert prepared["container_image"]["repo_digest"].endswith("b" * 64)
    assert prepared["container_workdir"] == "/home/beego"
    assert prepared["network_policy"] == "disabled"
    assert prepared["gpu_required"] is False
    assert prepared["benchmark_exclusion_gate"] == "ready"
    assert prepared["fail_to_pass"] == ["TestFixed"]
    assert prepared["pass_to_pass"] == ["TestExisting"]
    assert prepared["target_patch_sha256"] == row["target_patch_sha256"]
    assert prepared["test_command_sha256"]
    assert prepared["license_evidence_sha256"]


def test_pull_uses_digest_then_creates_local_verifier_tag(monkeypatch):
    row = _row()
    expected = row["container_image"]
    calls = []
    identities = iter([None, _identity(row)])
    monkeypatch.setattr(
        multisource, "_docker_image_identity", lambda image: next(identities)
    )

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": "sha256:" + "c" * 64,
                            "RepoDigests": [expected["repo_digest"]],
                        }
                    ]
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(multisource.subprocess, "run", run)
    actual, pulled = multisource._localize_exact_image(
        expected, pull_timeout=10, allow_pull=True
    )

    assert pulled is True
    assert calls[0] == ["docker", "pull", expected["repo_digest"]]
    assert expected["repo_tag"] not in calls[0]
    assert calls[-1][0:2] == ["docker", "tag"]
    assert actual["repo_digest"] == expected["repo_digest"]


def test_held_out_repository_fails_closed(tmp_path):
    row = _row(repository="grpc/grpc-go")
    _write_source(tmp_path, row)

    with pytest.raises(ValueError, match="held-out"):
        multisource._validate_source_row(
            tmp_path, row, multisource.SOURCE_CONFIGS["multiswe"]
        )


def test_bounded_pull_budget_defers_without_rejection(tmp_path, monkeypatch):
    row = _row()
    _write_source(tmp_path, row)
    monkeypatch.setattr(
        multisource,
        "derive_test_command",
        lambda candidate, language: ("cd /home/beego\ngo test ./...", "/home/beego"),
    )
    monkeypatch.setattr(multisource, "_docker_image_identity", lambda image: None)

    report = multisource.prepare_executable_candidates(
        tmp_path, "multiswe", max_pulls=0
    )

    assert report["prepared_in_scope"] == 0
    assert report["deferred"] == 1
    assert report["rejected"] == 0


def test_rust_execution_restores_digest_pinned_image_toolchain_path():
    command = multisource._execution_command("cd /home/ripgrep\ncargo test", "rust")

    assert command.startswith("export PATH=/usr/local/cargo/bin:$PATH\n")
    assert command.endswith("cargo test")


def test_go_execution_restores_digest_pinned_image_toolchain_path():
    command = multisource._execution_command("cd /home/beego\ngo test ./...", "go")

    assert command.startswith("export PATH=/go/bin:/usr/local/go/bin:$PATH\n")
    assert command.endswith("go test ./...")
