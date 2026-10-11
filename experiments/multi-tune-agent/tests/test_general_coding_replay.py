import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

import pytest

import multi_tune_agent.general_coding_replay as replay
from multi_tune_agent.general_coding_replay import (
    HUMANEVAL_REVISION,
    LIVECODEBENCH_REVISION,
    LIVECODEBENCH_WINDOW,
    MBPP_REVISION,
    MULTISWE_VALIDATION_LANGUAGE_COUNTS,
    MULTISWE_VERIFIED_REVISION,
    OPENCODE_REVISION,
    PHASE1_LANGUAGE_TARGETS,
    PHASE1_SOURCE_TARGETS,
    REJECTION_REASONS,
    SWE_GYM_PARQUET_SHA256,
    SWE_GYM_REVISION,
    ReplayRejected,
    _audit_prepared_benchmark_overlap,
    _assert_docker_image_identity,
    _benchmark_entry,
    _chunk_pytest_command,
    _docker_create_args,
    _eligible_swe_row,
    _pending_registry_blockers,
    _pytest_command,
    _validate_container_head,
    _validate_container_policy,
    audit_opencode_schema,
    bootstrap_kernel_exclusions,
    build_inventory,
    deduplicate_verified,
    derive_repository_task_type,
    package_replay,
    prepare_swe_gym_candidates,
    select_replay,
    sha256_bytes,
    validate_frozen_file,
    verify_case,
    verify_candidates,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _offline_repository(tmp_path: Path) -> tuple[Path, str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import add\nassert add(2, 3) == 5\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Replay Test",
            "-c",
            "user.email=replay@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "parent",
        ],
        check=True,
    )
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    patch = tmp_path / "target.patch"
    patch.write_text(_git(repo, "diff") + "\n", encoding="utf-8")
    _git(repo, "checkout", "--quiet", "--", "calc.py")
    return repo, commit, patch


def _candidate(tmp_path: Path, repo: Path, commit: str, patch: Path) -> dict:
    statement = tmp_path / "problem.md"
    statement.write_text("Fix add so the repository test passes.\n", encoding="utf-8")
    return {
        "case_id": "gc-replay-other-add-fix",
        "source_id": "other_test_backed_repositories",
        "dataset_id": "synthetic/offline",
        "dataset_revision": commit,
        "upstream_repository": "https://example.invalid/offline.git",
        "local_repository": str(repo),
        "base_commit": commit,
        "problem_id": "add-fix",
        "license_spdx": "MIT",
        "license_evidence_path": str(statement),
        "primary_language": "python",
        "primary_task_type": "repository_bug_fix_or_test_repair",
        "coding_task_type": "repository_bug_fix_or_test_repair",
        "problem_statement_path": str(statement),
        "target_patch_path": str(patch),
        "allowed_paths": ["calc.py"],
        "install_command": "true",
        "compile_or_typecheck_command": "python3 -m py_compile calc.py",
        "targeted_test_command": "python3 test_calc.py",
        "full_regression_command": "python3 test_calc.py",
        "network_policy": "disabled",
        "gpu_required": False,
        "source_lineage_id": "synthetic:add",
        "status": "inventoried",
        "split": "train",
        "sample_domain": "general_coding",
        "task_type": "general_coding_replay",
        "_manifest_dir": str(tmp_path),
        "timeouts": {
            "checkout_dependency": 30,
            "parent": 30,
            "target": 30,
            "regression": 30,
            "fresh_total": 120,
        },
    }


def test_fresh_offline_verification_is_resume_safe(tmp_path):
    repo, commit, patch = _offline_repository(tmp_path)
    candidate = _candidate(tmp_path, repo, commit, patch)
    case_dir = tmp_path / "verified" / candidate["case_id"]

    first = verify_case(candidate, case_dir)
    second = verify_case(candidate, case_dir)

    assert first["status"] == "verified"
    assert second == first
    record = json.loads((case_dir / "verified.json").read_text(encoding="utf-8"))
    assert record["fresh_verify_receipt"] == "fresh-receipt.json"
    receipt = json.loads((case_dir / "fresh-receipt.json").read_text(encoding="utf-8"))
    assert receipt["targeted_tests"]["network"] == "disabled_linux_namespace"
    assert receipt["targeted_tests"]["gpu_visibility"] == "disabled"


def test_forbidden_path_is_standardized_rejection(tmp_path):
    repo, commit, patch = _offline_repository(tmp_path)
    candidate = _candidate(tmp_path, repo, commit, patch)
    candidate["allowed_paths"] = ["README.md"]

    status = verify_case(candidate, tmp_path / "rejected")

    assert status["status"] == "rejected"
    assert status["rejection_reason"] == "forbidden_path_modified"
    assert status["rejection_reason"] in REJECTION_REASONS


def _verified(case_id: str, source: str, task: str, language: str, index: int) -> dict:
    return {
        "case_id": case_id,
        "status": "verified",
        "source_id": source,
        "coding_task_type": task,
        "primary_language": language,
        "upstream_repository": f"https://example.invalid/repo-{index}.git",
        "problem_id": f"problem-{index}",
        "source_lineage_id": f"lineage-{index}",
        "source_hash": f"source-{index}",
        "normalized_patch_hash": f"patch-{index}",
        "ast_fingerprint": f"ast-{index}",
        "diff_hunk_hash": f"diff-{index}",
        "test_set_hash": f"tests-{index}",
        "verified_runtime_seconds": index,
    }


def test_dedup_precedes_exact_three_axis_selection():
    rows = [
        _verified("gc-replay-a-1", "swe_gym", "bug", "python", 1),
        _verified("gc-replay-a-2", "swe_gym", "implementation", "rust", 2),
        _verified("gc-replay-b-3", "other", "bug", "rust", 3),
        _verified("gc-replay-b-4", "other", "implementation", "python", 4),
    ]
    duplicate = dict(rows[0], case_id="gc-replay-z-duplicate", problem_id="different")
    quotas = {
        "source_targets": {"swe_gym": 2, "other": 2},
        "task_type_targets": {"bug": 2, "implementation": 2},
        "language_targets": {"python": 2, "rust": 2},
    }

    pool, rejected = deduplicate_verified(rows + [duplicate])
    selected, report = select_replay(rows + [duplicate], quotas, 4)

    assert len(pool) == 4
    assert rejected[0]["rejection_reason"] == "duplicate_source"
    assert len(selected) == 4
    assert report["status"] == "exact"
    assert report["duplicate_rejections"] == 1


def test_selector_writes_no_rows_when_exact_quota_is_impossible():
    rows = [_verified("gc-replay-a-1", "swe_gym", "bug", "python", 1)]
    quotas = {
        "source_targets": {"swe_gym": 2},
        "task_type_targets": {"bug": 2},
        "language_targets": {"python": 2},
    }

    selected, report = select_replay(rows, quotas, 2)

    assert selected == []
    assert report["status"] == "quota_deficits"
    assert report["deficits"]["language"] == {"python": 1}


def test_exact_selector_enforces_source_language_cells_and_caps():
    rows = [
        _verified("gc-replay-a-python", "source-a", "bug", "python", 1),
        _verified("gc-replay-a-rust", "source-a", "bug", "rust", 2),
        _verified("gc-replay-b-python", "source-b", "implementation", "python", 3),
        _verified("gc-replay-b-rust", "source-b", "implementation", "rust", 4),
    ]
    rows[0]["upstream_repository"] = rows[1]["upstream_repository"] = "repo-a"
    quotas = {
        "source_targets": {"source-a": 1, "source-b": 1},
        "source_language_targets": {
            "source-a": {"python": 1},
            "source-b": {"rust": 1},
        },
        "task_type_targets": {"bug": 1, "implementation": 1},
        "language_targets": {"python": 1, "rust": 1},
        "caps": {"repository": 1, "lineage": 2, "source_fraction": 0.6},
    }

    selected, report = select_replay(rows, quotas, 2)

    assert {row["case_id"] for row in selected} == {
        "gc-replay-a-python",
        "gc-replay-b-rust",
    }
    assert report["selection_method"] == "scipy_milp_highs"


def test_bounded_verification_limit_is_deterministic(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("gc-replay-z-3", "gc-replay-a-1", "gc-replay-m-2")
        )
        + "\n",
        encoding="utf-8",
    )
    attempted = []

    def fake_verify(row, _case_dir):
        attempted.append(row["case_id"])
        return {"status": "verified"}

    monkeypatch.setattr(replay, "verify_case", fake_verify)
    report = verify_candidates(candidates, tmp_path / "verified", limit=2)

    assert attempted == ["gc-replay-a-1", "gc-replay-m-2"]
    assert report["attempted"] == 2


def test_resume_skips_completed_before_applying_limit(tmp_path, monkeypatch):
    rows = [{"case_id": case_id} for case_id in ("gc-replay-a", "gc-replay-b", "gc-replay-c")]
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed_dir = tmp_path / "verified" / "gc-replay-a"
    completed_dir.mkdir(parents=True)
    (completed_dir / "status.json").write_text(
        json.dumps(
            {
                "case_id": "gc-replay-a",
                "candidate_sha256": replay._candidate_verification_digest(rows[0]),
                "status": "verified",
            }
        ),
        encoding="utf-8",
    )
    attempted = []

    def fake_verify(row, _case_dir):
        attempted.append(row["case_id"])
        return {"status": "verified"}

    monkeypatch.setattr(replay, "verify_case", fake_verify)
    report = verify_candidates(
        candidates,
        tmp_path / "verified",
        resume=True,
        limit=2,
        batch_id="resume-test",
    )

    assert attempted == ["gc-replay-b", "gc-replay-c"]
    assert report["attempted"] == 2
    assert report["skipped_completed"] == 1


def test_parallel_verification_writes_namespaced_and_aggregate_reports(tmp_path, monkeypatch):
    rows = [{"case_id": case_id} for case_id in ("gc-replay-a", "gc-replay-b")]
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def fake_verify(row, case_dir):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        status = {
            "case_id": row["case_id"],
            "candidate_sha256": replay._candidate_verification_digest(row),
            "status": "verified",
        }
        replay.write_json(case_dir / "status.json", status)
        with guard:
            active -= 1
        return status

    monkeypatch.setattr(replay, "verify_case", fake_verify)
    report = verify_candidates(
        candidates,
        tmp_path / "verified",
        workers=2,
        batch_id="parallel-test",
    )

    assert maximum_active == 2
    assert report["snapshot"]["verified"] == 2
    assert (tmp_path / "verified" / "batch-reports" / "parallel-test.json").is_file()
    aggregate = json.loads(
        (tmp_path / "verified" / "verification-report.json").read_text(encoding="utf-8")
    )
    assert aggregate["verified"] == 2


def test_inventory_keeps_pending_empty_manifest_empty(tmp_path):
    manifest = tmp_path / "source-manifest.yaml"
    manifest.write_text("status: pending_sources\nsources: []\n", encoding="utf-8")
    registry = tmp_path / "exclusion-registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "candidates.jsonl"

    report = build_inventory(manifest, registry, output)

    assert report["schema_version"] == "general_coding_replay_v1"
    assert report["inventoried"] == 0
    assert report["rejected"] == 0
    assert set(report) >= {
        "source_manifest_sha256",
        "exclusion_registry_sha256",
        "candidates_sha256",
    }
    assert output.read_text(encoding="utf-8") == ""


def test_authoritative_source_revisions_and_checksum_are_frozen():
    assert SWE_GYM_REVISION == "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
    assert OPENCODE_REVISION == "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
    assert SWE_GYM_PARQUET_SHA256 == "60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97"
    assert MULTISWE_VERIFIED_REVISION == "80de95c62ac792c99dcfa8e26569bcd7d036bdc3"
    assert PHASE1_SOURCE_TARGETS == {"swe_gym": 250, "multi_swe_rl_verified": 250}
    assert sum(PHASE1_LANGUAGE_TARGETS.values()) == 500
    assert MULTISWE_VALIDATION_LANGUAGE_COUNTS["cpp"] == 0
    assert MULTISWE_VALIDATION_LANGUAGE_COUNTS["rust"] == 0


def test_repository_task_type_is_evidence_derived_and_deterministic():
    row = {
        "title": "Improve parser performance",
        "body": "Reduce allocations in the hot path.",
        "fix_patch": "diff --git a/src/parser.rs b/src/parser.rs\n+fast();\n",
        "test_patch": "diff --git a/tests/parser.rs b/tests/parser.rs\n+test();\n",
    }

    first = derive_repository_task_type(row)
    second = derive_repository_task_type(dict(row))

    assert first == second
    assert first[0] == "non_kernel_performance_repair"
    assert first[1]["method"] == "title_intent_and_patch_structure_v2"
    assert first[1]["matched_text"]


def test_repository_task_type_does_not_match_incidental_body_words():
    task_type, evidence = derive_repository_task_type(
        {
            "title": "Crash with recursive aliases",
            "body": "The config used to build this example is included in the docs.",
            "fix_patch": "diff --git a/src/types.py b/src/types.py\n+guard_cycle()\n",
            "test_patch": "diff --git a/tests/test_types.py b/tests/test_types.py\n+test_cycle()\n",
        }
    )

    assert task_type == "repository_bug_fix_or_test_repair"
    assert evidence["matched_rule"] == "test_backed_default"


def test_repository_task_type_uses_exclusive_patch_path_evidence():
    task_type, evidence = derive_repository_task_type(
        {
            "title": "Clarify parser behavior",
            "fix_patch": "diff --git a/docs/parser.md b/docs/parser.md\n+Clarification.\n",
            "test_patch": "diff --git a/tests/test_docs.py b/tests/test_docs.py\n+test_example()\n",
        }
    )

    assert task_type == "typing_validation_or_docs_with_test"
    assert evidence["matched_rule"] == "exclusive_documentation_paths"


@pytest.mark.parametrize("title", ["Adapt for pandas 2.1", "Deprecate the old parser", "Rework the allocator"])
def test_repository_task_type_recognizes_api_adaptation_inflections(title):
    task_type, _evidence = derive_repository_task_type(
        {
            "title": title,
            "fix_patch": "diff --git a/src/api.py b/src/api.py\n+updated()\n",
            "test_patch": "diff --git a/tests/test_api.py b/tests/test_api.py\n+test_updated()\n",
        }
    )

    assert task_type == "refactor_or_api_adaptation"


def test_frozen_file_rejects_checksum_drift(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"authoritative bytes")

    try:
        validate_frozen_file(source, "0" * 64)
    except RuntimeError as exc:
        assert "expected sha256" in str(exc)
    else:
        raise AssertionError("checksum drift was accepted")


def test_swe_candidate_rejects_missing_tests_and_nonimmutable_parent():
    row = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "patch": "diff --git a/a b/a\n",
        "problem_statement": "Fix it.",
        "test_patch": "",
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": [],
    }
    assert _eligible_swe_row(row) == (False, "test_metadata_missing")
    row["FAIL_TO_PASS"] = ["test_fix"]
    row["base_commit"] = "main"
    assert _eligible_swe_row(row) == (False, "base_commit_not_immutable")


def test_opencode_schema_emits_zero_admissible_blocker():
    fields = [
        "id",
        "input",
        "output",
        "domain",
        "generation_algorithm",
        "llm_judgement",
        "unit_tests",
        "tests_execution_status",
        "average_test_score",
    ]

    report = audit_opencode_schema(fields)

    assert report["admissible_rows"] == 0
    assert report["rejection_scope_rows"] == 5_000_000
    assert report["requirements"]["executable_tests"]["satisfied"] is True
    assert report["requirements"]["recoverable_parent_revision"]["satisfied"] is False
    assert report["requirements"]["target_diff"]["satisfied"] is False


def test_preflight_reuses_bare_mirror_and_preserves_source_bytes(tmp_path):
    repo, commit, patch = _offline_repository(tmp_path)
    control = tmp_path / "control"
    mirror = control / "mirrors" / "owner__repo.git"
    mirror.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "--mirror", str(repo), str(mirror)],
        check=True,
        capture_output=True,
        text=True,
    )
    (control / "swe-gym-upstream-license-report.json").write_text(
        json.dumps(
            {
                "repositories": [
                    {
                        "repository": "owner/repo",
                        "status": "accepted",
                        "spdx": "MIT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (control / "exclusion-registry.json").write_text(
        json.dumps({"pending": ["external benchmarks"]}), encoding="utf-8"
    )
    (control / "source-manifest.yaml").write_text(
        "schema_version: general_coding_replay_source_manifest_v1\nsources: []\n",
        encoding="utf-8",
    )
    patch_text = patch.read_text(encoding="utf-8")
    test_patch = "diff --git a/test_calc.py b/test_calc.py\n"
    row = {
        "case_id": "gc-replay-swe_gym-owner__repo-1",
        "source_id": "swe_gym",
        "dataset_id": "SWE-Gym/SWE-Gym",
        "dataset_revision": SWE_GYM_REVISION,
        "upstream_repository": "https://github.com/owner/repo.git",
        "base_commit": commit,
        "problem_id": "owner__repo-1",
        "problem_statement": "Fix add without changing this text.\n",
        "target_patch": patch_text,
        "target_patch_sha256": sha256_bytes(patch_text.encode()),
        "test_patch": test_patch,
        "test_patch_sha256": sha256_bytes(test_patch.encode()),
        "pass_to_pass": ["test_calc.py"],
        "fail_to_pass": ["test_calc.py"],
    }
    seed = control / "seed.jsonl"
    seed.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = prepare_swe_gym_candidates(seed, control)

    assert report["prepared"] == 1
    assert report["untrusted_tests_executed"] == 0
    assert report["benchmark_exclusion_gate"] == "blocked_pending_registry"
    prepared = json.loads(
        (control / "prepared-source-manifest.jsonl").read_text(encoding="utf-8")
    )
    assert Path(prepared["target_patch_path"]).read_text(encoding="utf-8") == patch_text
    assert prepared["network_policy"] == "disabled"
    assert prepared["gpu_required"] is False
    assert prepared["parent_reproduced"] is True
    assert prepared["full_regression_command"] == "python3 -m pytest -q test_calc.py"
    assert Path(prepared["test_patch_path"]).read_text(encoding="utf-8") == test_patch


def test_container_commit_mismatch_is_rejected():
    with pytest.raises(ReplayRejected, match="/testbed HEAD mismatch") as caught:
        _validate_container_head({"ok": True, "stdout": "b" * 40 + "\n"}, "a" * 40)
    assert caught.value.reason == "parent_revision_missing"


def test_container_image_identity_is_immutable(monkeypatch):
    expected = {
        "repo_tag": "example/image:latest",
        "repo_digest": "example/image@sha256:" + "a" * 64,
        "image_id": "sha256:" + "b" * 64,
    }
    monkeypatch.setattr(replay, "_docker_image_identity", lambda _tag: dict(expected))
    assert _assert_docker_image_identity(expected) == expected
    changed = dict(expected, image_id="sha256:" + "c" * 64)
    monkeypatch.setattr(replay, "_docker_image_identity", lambda _tag: changed)
    with pytest.raises(ReplayRejected, match="image_id mismatch"):
        _assert_docker_image_identity(expected)


def test_container_policy_forbids_network_and_gpu_devices(tmp_path):
    identity = {"image_id": "sha256:" + "a" * 64}
    patch = tmp_path / "target.patch"
    patch.write_text("", encoding="utf-8")
    args = _docker_create_args(identity, patch, None)
    assert args[args.index("--network") + 1] == "none"
    assert "--gpus" not in args and "--device" not in args
    assert {"CUDA_VISIBLE_DEVICES=", "HIP_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES="} <= set(args)
    valid = {
        "HostConfig": {"NetworkMode": "none", "Devices": [], "DeviceRequests": []},
        "Config": {
            "Env": [
                "CUDA_VISIBLE_DEVICES=",
                "HIP_VISIBLE_DEVICES=",
                "ROCR_VISIBLE_DEVICES=",
            ]
        },
    }
    _validate_container_policy(valid)
    invalid_network = json.loads(json.dumps(valid))
    invalid_network["HostConfig"]["NetworkMode"] = "bridge"
    with pytest.raises(ReplayRejected) as caught:
        _validate_container_policy(invalid_network)
    assert caught.value.reason == "network_isolation_unavailable"
    invalid = json.loads(json.dumps(valid))
    invalid["HostConfig"]["DeviceRequests"] = [{"Driver": "nvidia"}]
    with pytest.raises(ReplayRejected) as caught:
        _validate_container_policy(invalid)
    assert caught.value.reason == "gpu_required"


def test_hydra_pytest_commands_disable_snail_plugin():
    command = _pytest_command(["tests/test_config_loader.py::test_case"], disable_snail=True)
    assert command.startswith("python3 -m pytest -p no:snail -q ")


def test_pytest_command_restores_escaped_parameterized_node_ids():
    command = _pytest_command(["tests/test_app.py::test_case[value:\nnext]"])
    assert "'tests/test_app.py::test_case[value:\\nnext]'" in command
    assert "\n" not in command


def test_oversized_pytest_command_is_split_without_losing_node_ids():
    tests = [f"tests/test_app.py::test_case[{index}]" for index in range(20)]
    command = _pytest_command(tests)
    chunks = _chunk_pytest_command(command, max_chars=180)

    assert len(chunks) > 1
    assert all(len(chunk) <= 180 for chunk in chunks)
    assert [
        token
        for chunk in chunks
        for token in shlex.split(chunk)
        if "::" in token
    ] == tests


def test_relative_path_packaging_and_checksums(tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    rows = []
    for index in range(2):
        artifact = tmp_path / f"artifact-{index}"
        artifact.mkdir()
        (artifact / "code.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
        row = _verified(
            f"gc-replay-other-{index}",
            "other",
            "implementation",
            "python",
            index,
        )
        row.update(
            {
                "artifact_paths": {"candidate_workspace": str(artifact)},
                "assistant_loss_tokens": 20,
                "kernel_assistant_loss_tokens": 160,
            }
        )
        rows.append(row)
    accepted.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    root = tmp_path / "hf"
    manifest = package_replay(accepted, root, expected_count=2)

    packaged = [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["relative_paths_only"] is True
    assert all(not Path(row["artifact_paths"]["candidate_workspace"]).is_absolute() for row in packaged)
    assert (root / "checksums.sha256").is_file()


def test_kernel_exclusion_bootstrap_requires_exact_frozen_row_count(tmp_path):
    train = tmp_path / "train.jsonl"
    row = {
        "provenance": {
            "source_lineage_id": "kernel-lineage",
            "contract_hash": "contract-hash",
            "operator": "gemm",
        },
        "input": {"parent_source": {"kernel.py": "def kernel_symbol():\n    pass\n"}},
    }
    train.write_text((json.dumps(row) + "\n") * 2000, encoding="utf-8")

    registry = bootstrap_kernel_exclusions(train, tmp_path / "registry.json")

    assert registry["kernel_train"]["rows"] == 2000
    assert registry["source_lineages"] == ["kernel-lineage"]
    assert registry["contract_hashes"] == ["contract-hash"]
    assert {"gemm", "kernel_symbol"} <= set(registry["symbols"])


def test_public_benchmark_revisions_and_window_are_immutable():
    assert HUMANEVAL_REVISION == "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
    assert MBPP_REVISION == "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
    assert LIVECODEBENCH_REVISION == "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
    assert LIVECODEBENCH_WINDOW == ("2025-01-01T00:00:00", "2025-04-30T23:59:59")


def test_benchmark_entry_normalizes_text_and_records_signatures():
    kwargs = {
        "dataset_key": "fixture",
        "dataset_id": "fixture/data",
        "revision": "a" * 40,
        "config": "default",
        "split": "test",
        "problem_id": "one",
        "signature_source": "def add(a: int, b: int = 1):\n    pass\n",
    }
    first = _benchmark_entry(prompt="Write add.  \r\n", text="Write add.  \r\n", **kwargs)
    second = _benchmark_entry(prompt="Write add.\n", text="Write add.\n", **kwargs)

    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert first["text_sha256"] == second["text_sha256"]
    assert first["signatures"] == ["def add(a: int, b: int=1)"]
    assert len(first["signature_fingerprints"]) == 1


def test_pending_domains_are_machine_readable_and_keep_gate_blocked(tmp_path):
    statement = tmp_path / "problem.md"
    statement.write_text("A distinct prepared problem.\n", encoding="utf-8")
    manifest = tmp_path / "prepared-source-manifest.jsonl"
    row = {
        "case_id": "gc-replay-swe_gym-owner__repo-1",
        "problem_id": "owner__repo-1",
        "upstream_repository": "https://github.com/owner/repo.git",
        "problem_statement_path": str(statement),
        "benchmark_exclusion_gate": "blocked_pending_registry",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    blockers = _pending_registry_blockers()

    report = _audit_prepared_benchmark_overlap(
        tmp_path,
        {"problem_ids": ["HumanEval/0"], "blockers": blockers},
    )

    assert report["overlap_count"] == 0
    assert report["prepared_manifest_updated"] is False
    assert report["benchmark_exclusion_gate"] == "blocked_pending_registry"
    assert all(item["status"] == "pending_unavailable" for item in blockers)
    assert json.loads(manifest.read_text(encoding="utf-8"))["benchmark_exclusion_gate"] == (
        "blocked_pending_registry"
    )


def test_overlap_audit_reports_excluded_candidate(tmp_path):
    statement = tmp_path / "problem.md"
    statement.write_text("Exact public prompt.\n", encoding="utf-8")
    row = {
        "case_id": "gc-replay-swe_gym-overlap-1",
        "problem_id": "HumanEval/0",
        "upstream_repository": "https://github.com/owner/repo.git",
        "problem_statement_path": str(statement),
        "benchmark_exclusion_gate": "blocked_pending_registry",
    }
    (tmp_path / "prepared-source-manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    report = _audit_prepared_benchmark_overlap(
        tmp_path,
        {"problem_ids": ["HumanEval/0"], "blockers": []},
    )

    assert report["overlap_count"] == 1
    assert report["excluded_candidates"] == [row["case_id"]]
    excluded = json.loads(
        (tmp_path / "benchmark-excluded-candidates.jsonl").read_text(encoding="utf-8")
    )
    assert excluded["matches"] == ["HumanEval/0"]


def test_benchmark_aliases_are_normalized_and_evidenced():
    entry = _benchmark_entry(
        dataset_key="humaneval",
        dataset_id="fixture/humaneval",
        revision="a" * 40,
        config="default",
        split="test",
        problem_id="HumanEval/0",
        prompt="Write foo.\n",
        text="Write foo.\n",
        signature_source="def foo(value: int):\n    pass\n",
    )

    records = replay._benchmark_alias_records([entry])

    assert any(item["kind"] == "normalized_problem_alias" for item in records)
    assert any(item["kind"] == "normalized_signature_alias" for item in records)
    assert any(item["kind"] == "normalized_prompt_sha256" for item in records)
    assert all(item["evidence"][0]["problem_id"] == "HumanEval/0" for item in records)


def test_dev_reservation_requires_evidence_and_exact_mix(tmp_path):
    seed = tmp_path / "seed.jsonl"
    rows = []
    titles = (
        [f"BUG: repair behavior {index}" for index in range(40)]
        + [f"Implement feature {index}" for index in range(20)]
        + [f"DEPR: retire API {index}" for index in range(20)]
    )
    for index, title in enumerate(titles):
        patch = f"diff --git a/file{index}.py b/file{index}.py\n+VALUE = {index}\n"
        rows.append(
            {
                "case_id": f"gc-replay-swe_gym-fixture-{index}",
                "problem_id": f"fixture-{index}",
                "dataset_id": "fixture/swe-gym",
                "dataset_revision": "a" * 40,
                "upstream_repository": f"https://github.com/fixture/repo-{index}.git",
                "primary_task_type": "repository_bug_fix_or_test_repair",
                "problem_statement": title,
                "target_patch": patch,
                "target_patch_sha256": sha256_bytes(patch.encode()),
                "fail_to_pass": ["test"],
            }
        )
    seed.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "prepared-source-manifest.jsonl").write_text("", encoding="utf-8")

    selected, report = replay._reserve_general_coding_dev(seed, tmp_path)

    assert report["status"] == "complete"
    assert report["deficits"] == {}
    assert report["reserved_by_task_type"] == {
        "function_implementation": 20,
        "refactor_or_api_adaptation": 20,
        "repository_bug_fix_or_test_repair": 40,
    }
    assert len(selected) == 80
    assert all(item["classification_evidence"]["field"] != "none" for item in selected)


def test_append_safe_source_check_rejects_checksum_drift():
    with pytest.raises(RuntimeError, match="source drift"):
        replay._assert_append_source(
            {"source": {"path": "split.jsonl", "sha256": "a" * 64}},
            {"source": {"path": "split.jsonl", "sha256": "b" * 64}},
        )
