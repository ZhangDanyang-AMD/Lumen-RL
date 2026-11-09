import difflib
import json
from pathlib import Path

from multi_tune_agent.sft_dataset import (
    _contract_fields,
    _environment_fields,
    _environment_with_override,
    _initial_reasons,
    audit_leakage,
    build_dataset,
    canonical_bytes,
    coverage_report,
    sha256_bytes,
    validate_dataset,
    apply_unified_patch,
)


def test_legacy_environment_and_contract_fields_are_derived():
    environment = {
        "gpu_architecture": {"ok": True, "stdout": "gfx942\n"},
        "gpu_inventory": {"ok": True, "stdout": "Card SKU:\t\tM3000108\n"},
    }
    frozen = {
        "contract": {
            "architecture": "gfx942",
            "shapes": [[128, 32, 8192]],
            "provenance": {"case_seed": {"target_lane": "hip_gfx942"}},
        }
    }

    assert _environment_fields(environment)["gpu_sku"] == "M3000108"
    assert _contract_fields(frozen)["shape_regime"] == "large"


def test_pinned_container_environment_override_requires_matching_evidence():
    digest = "sha256:image"
    environment = {
        "container": {"identity": {"stdout": digest}},
        "lumen_git": {"head": "a", "working_diff_sha256": "b"},
        "geak_git": {"head": "c", "working_diff_sha256": "d"},
    }
    manifest = {
        "environment_overrides": {
            digest: {
                "software": {
                    "rocm_version": "7.0.0",
                    "compiler_version": "clang-20",
                },
                "evidence": {
                    "rocm_version": {
                        "ok": True,
                        "returncode": 0,
                        "command": "probe rocm",
                        "stdout": "7.0.0\n",
                    },
                    "compiler_version": {
                        "ok": True,
                        "returncode": 0,
                        "command": "probe compiler",
                        "stdout": "clang-20\n",
                    },
                },
            }
        }
    }

    merged = _environment_with_override(environment, manifest)

    assert merged["software"]["rocm_version"] == "7.0.0"


def test_error_recovery_does_not_require_benchmark_receipt():
    candidate = {
        "role": "engineer",
        "task_type": "error_recovery",
        "patch_applies": True,
        "independent_verify": True,
        "compile_pass": True,
        "correctness_pass": True,
        "benchmark_valid": False,
        "sft_positive_eligible": True,
    }

    assert not any(
        reason["code"] == "benchmark_invalid"
        for reason in _initial_reasons(candidate)
    )


def test_replay_handles_difflib_changed_final_line_without_newline():
    before = "value = 1"
    after = "value = 2"
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/kernel.py",
            tofile="b/kernel.py",
        )
    ).encode()

    applied, changed = apply_unified_patch({"kernel.py": before.encode()}, patch)

    assert changed == {"kernel.py"}
    assert applied == {"kernel.py": after.encode()}


def test_replay_handles_difflib_adding_final_newline():
    before = "value = 1"
    after = "value = 2\n"
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/kernel.py",
            tofile="b/kernel.py",
        )
    ).encode()

    applied, _ = apply_unified_patch({"kernel.py": before.encode()}, patch)

    assert applied == {"kernel.py": after.encode()}


def _write_blob(root: Path, data: bytes) -> str:
    digest = sha256_bytes(data)
    path = root / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def _receipt(mode: str):
    return {
        "mode": mode,
        "ok": True,
        "returncode": 0,
        "command": f"runner {mode}",
        "stdout": "ok",
        "stderr": "",
    }


def _make_run(
    root: Path,
    blobs: Path,
    name: str,
    *,
    split: str = "train",
    lineage: str = "lineage-a",
    before: str = "value = 1\n",
    after: str = "value = 2\n",
    positive: bool = True,
):
    run = root / "raw" / "runs" / name
    round_dir = run / "round_1"
    round_dir.mkdir(parents=True)
    parent_data = before.encode()
    child_data = after.encode()
    parent_hash = _write_blob(blobs, parent_data)
    child_hash = _write_blob(blobs, child_data)
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/kernel.py",
            tofile="b/kernel.py",
        )
    ).encode()
    patch_hash = _write_blob(blobs, patch)
    parent_source = {"kernel.py": before}
    source_hash = sha256_bytes(
        json.dumps(parent_source, sort_keys=True, default=str).encode()
    )
    frozen = {
        "schema_version": "geak_sft_frozen_input_v1",
        "run_id": name,
        "round": 1,
        "task_type": "cold_start",
        "created_at": 1,
        "input": {
            "contract": {
                "architecture": "gfx942",
                "case_id": f"task-{name}",
                "operator": "gemm",
                "language": "triton",
                "shape_regime": "small",
                "contract_hash": f"contract-{name}",
                "contract": {
                    "operator": "gemm",
                    "dtype": "fp16",
                    "format": "row_major",
                },
                "provenance": {
                    "case_seed": {
                        "target_lane": "triton_gfx942",
                        "source_lineage_id": lineage,
                        "implementation_family_id": f"family-{lineage}",
                        "split_group": split,
                        "split_version": "split-v1",
                    }
                },
            },
            "parent_source": parent_source,
            "parent_source_hash": source_hash,
            "baseline": {"per_case_ms": {"case": 2.0}},
            "source_lineage_id": lineage,
            "split_group": split,
            "split_version": "split-v1",
        },
    }
    frozen_hash = _write_blob(blobs, canonical_bytes(frozen))
    (round_dir / "frozen_input.json").write_bytes(canonical_bytes(frozen))
    evaluation = {
        "compiled": True,
        "correct": True,
        "speedup_geomean": 2.0,
        "compile": _receipt("compile"),
        "correctness": _receipt("correctness"),
        "performance": _receipt("performance"),
    }
    candidate = {
        "schema_version": "geak_sft_candidate_v1",
        "run_id": name,
        "round": 1,
        "role": "engineer",
        "task_type": "cold_start",
        "frozen_input_hash": frozen_hash,
        "candidate": {
            "candidate_id": f"candidate-{name}",
            "accepted": positive,
            "valid_assistant_tokens": 17,
        },
        "parent_sources": {
            "kernel.py": {"sha256": parent_hash, "size": len(parent_data)}
        },
        "candidate_sources": {
            "kernel.py": {"sha256": child_hash, "size": len(child_data)}
        },
        "patch_hash": patch_hash,
        "patch_bytes": len(patch),
        "patch_applies": positive,
        "verify_result": {
            "verify_source": "multitune_independent",
            "verify_session_id": f"verify-{name}",
            "verify_workspace": f"/verify/{name}",
            "evaluation": evaluation,
        },
        "independent_verify": positive,
        "compile_pass": positive,
        "correctness_pass": positive,
        "benchmark_valid": positive,
        "sft_positive_eligible": positive,
    }
    (round_dir / "candidates.jsonl").write_text(
        json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        "schema_version": "geak_sft_environment_v1",
        "gpu": {"architecture": "gfx942", "sku": "MI308X"},
        "software": {"rocm_version": "7.0", "compiler_version": "clang-20"},
        "container": {"digest": "sha256:image"},
        "lumen_git": {
            "head": "a" * 40,
            "working_state_sha256": "c" * 64,
        },
        "geak_git": {
            "head": "b" * 40,
            "working_state_sha256": "d" * 64,
        },
    }
    (run / "environment.json").write_bytes(canonical_bytes(environment))
    run_manifest = {
        "schema_version": "geak_sft_manifest_v1",
        "run_id": name,
        "collector_complete": True,
        "dataset_eligible": True,
    }
    (run / "sft_manifest.json").write_bytes(canonical_bytes(run_manifest))
    return run


def _input_manifest(root: Path, blobs: Path, runs: list[Path]):
    path = root / "input_manifest.json"
    value = {
        "schema_version": "geak_sft_input_manifest_v1",
        "dataset_version": "test-v1",
        "output_root": str(root),
        "blob_root": str(blobs),
        "runs": [
            {
                "path": str(run),
                "candidate_files": ["round_1/candidates.jsonl"],
                "artifacts": {
                    "sft_manifest.json": sha256_bytes(
                        (run / "sft_manifest.json").read_bytes()
                    ),
                    "environment.json": sha256_bytes(
                        (run / "environment.json").read_bytes()
                    ),
                    "round_1/frozen_input.json": sha256_bytes(
                        (run / "round_1/frozen_input.json").read_bytes()
                    ),
                    "round_1/candidates.jsonl": sha256_bytes(
                        (run / "round_1/candidates.jsonl").read_bytes()
                    ),
                },
            }
            for run in runs
        ],
    }
    path.write_bytes(canonical_bytes(value))
    return path


def test_build_validate_audit_coverage_and_idempotence(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    good = _make_run(tmp_path, blobs, "good")
    failed = _make_run(tmp_path, blobs, "failed", positive=False)
    input_manifest = _input_manifest(tmp_path, blobs, [good, failed])
    raw_before = {
        path: path.read_bytes() for path in (tmp_path / "raw").rglob("*") if path.is_file()
    }

    first = build_dataset(input_manifest)
    first_outputs = {
        path: path.read_bytes()
        for path in (tmp_path / "processed").glob("*.jsonl")
    }
    second = build_dataset(input_manifest)

    assert first["accepted"] == second["accepted"] == 1
    assert first_outputs == {
        path: path.read_bytes()
        for path in (tmp_path / "processed").glob("*.jsonl")
    }
    assert raw_before == {path: path.read_bytes() for path in raw_before}
    quality = validate_dataset(Path(first["manifest"]))
    leakage = audit_leakage(Path(first["manifest"]))
    coverage = coverage_report(Path(first["manifest"]))
    assert quality["status"] == "pass"
    assert leakage["status"] == "pass"
    assert coverage["status"] == "pass"
    assert coverage["coverage"]["language"] == {"triton": 1}
    assert coverage["valid_assistant_tokens"]["total"] == 17

    rejected = [
        json.loads(line)
        for line in (tmp_path / "processed" / "rejected.jsonl").read_text().splitlines()
    ]
    assert rejected
    assert all(item["reasons"] for item in rejected)
    assert any(
        reason["code"] == "not_positive_eligible"
        for item in rejected
        for reason in item["reasons"]
    )


def test_exact_and_normalized_patch_duplicates_are_rejected(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    first = _make_run(tmp_path, blobs, "one")
    second = _make_run(tmp_path, blobs, "two")
    result = build_dataset(_input_manifest(tmp_path, blobs, [first, second]))

    assert result["accepted"] == 1
    rejected = [
        json.loads(line)
        for line in (tmp_path / "processed" / "rejected.jsonl").read_text().splitlines()
    ]
    duplicate = next(
        item
        for item in rejected
        if any(reason["code"] == "duplicate_exact_patch" for reason in item["reasons"])
    )
    assert duplicate["duplicate_of"]


def test_deduplication_preserves_explicitly_pinned_sample(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    first = _make_run(tmp_path, blobs, "one")
    second = _make_run(tmp_path, blobs, "two")
    manifest = _input_manifest(tmp_path, blobs, [first, second])
    value = json.loads(manifest.read_text())
    value["pinned_sample_keys"] = [["one", "candidate-one"]]
    manifest.write_bytes(canonical_bytes(value))

    result = build_dataset(manifest)

    assert result["accepted"] == 1
    sample = json.loads((tmp_path / "processed" / "train.jsonl").read_text())
    assert sample["provenance"]["run_id"] == "one"
    assert sample["provenance"]["candidate_id"] == "candidate-one"


def test_input_manifest_can_pin_exact_candidate_ids(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    first = _make_run(tmp_path, blobs, "one")
    second = _make_run(tmp_path, blobs, "two", after="value = 3\n")
    manifest = _input_manifest(tmp_path, blobs, [first, second])
    value = json.loads(manifest.read_text())
    value["runs"][0]["candidate_ids"] = ["candidate-one"]
    value["runs"][1]["candidate_ids"] = ["not-selected"]
    manifest.write_bytes(canonical_bytes(value))

    result = build_dataset(manifest)

    assert result["accepted"] == 1
    sample = json.loads((tmp_path / "processed" / "train.jsonl").read_text())
    assert sample["provenance"]["candidate_id"] == "candidate-one"


def test_coverage_targets_fail_when_quota_is_not_met(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    run = _make_run(tmp_path, blobs, "one")
    input_manifest = _input_manifest(tmp_path, blobs, [run])
    value = json.loads(input_manifest.read_text())
    value["coverage_targets"] = {
        "sample_count": 2,
        "task_type": {"cold_start": 2},
        "lane": {"triton_gfx942": 2},
    }
    input_manifest.write_bytes(canonical_bytes(value))

    result = build_dataset(input_manifest)
    report = coverage_report(Path(result["manifest"]))

    assert report["status"] == "fail"
    assert report["gap_count"] == 3


def test_coverage_resolves_task_family_and_excludes_pinned_checkpoint(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    seed = _make_run(tmp_path, blobs, "seed", lineage="lineage-seed")
    fresh = _make_run(
        tmp_path,
        blobs,
        "fresh",
        lineage="lineage-fresh",
        after="value = 3\n",
    )
    input_manifest = _input_manifest(tmp_path, blobs, [seed, fresh])
    value = json.loads(input_manifest.read_text())
    value["task_families"] = {"task-seed": "gemm", "task-fresh": "gemm"}
    value["pinned_sample_keys"] = [["seed", "candidate-seed"]]
    value["coverage_targets"] = {
        "sample_count": 2,
        "post_checkpoint_top10_family": {"gemm": 1},
    }
    input_manifest.write_bytes(canonical_bytes(value))

    result = build_dataset(input_manifest)
    report = coverage_report(Path(result["manifest"]))

    assert report["status"] == "pass"
    assert report["coverage"]["top10_family"]["gemm"] == 2
    assert report["coverage"]["post_checkpoint_top10_family"]["gemm"] == 1


def test_leakage_audit_fails_on_cross_split_lineage(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    train = _make_run(
        tmp_path, blobs, "train-run", split="train", lineage="shared", after="value = 2\n"
    )
    dev = _make_run(
        tmp_path, blobs, "dev-run", split="dev", lineage="shared", after="value = 3\n"
    )
    result = build_dataset(_input_manifest(tmp_path, blobs, [train, dev]))
    report = audit_leakage(Path(result["manifest"]))

    assert report["status"] == "fail"
    assert report["source_lineage_split_overlap_count"] == 1


def test_tampered_patch_and_protected_path_are_rejected(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    run = _make_run(tmp_path, blobs, "tampered")
    candidates = run / "round_1" / "candidates.jsonl"
    candidate = json.loads(candidates.read_text())
    patch = (
        "--- a/task_runner.py\n"
        "+++ b/task_runner.py\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
    ).encode()
    candidate["patch_hash"] = _write_blob(blobs, patch)
    candidate["patch_bytes"] = len(patch)
    candidate["candidate_sources"]["task_runner.py"] = {
        "sha256": _write_blob(blobs, b"pass\n"),
        "size": 5,
    }
    candidates.write_text(json.dumps(candidate) + "\n")
    manifest = _input_manifest(tmp_path, blobs, [run])

    result = build_dataset(manifest)

    assert result["accepted"] == 0
    text = (tmp_path / "processed" / "rejected.jsonl").read_text()
    assert "harness_modified" in text or "patch_does_not_apply" in text


def test_direction_conditioned_requires_persisted_plan(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    run = _make_run(tmp_path, blobs, "direction")
    frozen_path = run / "round_1" / "frozen_input.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["task_type"] = "direction_conditioned"
    frozen_path.write_bytes(canonical_bytes(frozen))
    candidate_path = run / "round_1" / "candidates.jsonl"
    candidate = json.loads(candidate_path.read_text())
    candidate["task_type"] = "direction_conditioned"
    candidate["candidate"]["direction"] = {"direction_id": "tile"}
    candidate["frozen_input_hash"] = _write_blob(blobs, canonical_bytes(frozen))
    candidate_path.write_text(json.dumps(candidate) + "\n")

    result = build_dataset(_input_manifest(tmp_path, blobs, [run]))

    assert result["accepted"] == 0
    assert "missing_direction_plan" in (
        tmp_path / "processed" / "rejected.jsonl"
    ).read_text()


def test_mode_context_is_read_from_frozen_input_payload(tmp_path):
    blobs = tmp_path / "blobs" / "sha256"
    run = _make_run(tmp_path, blobs, "profile")
    frozen_path = run / "round_1" / "frozen_input.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["task_type"] = "profile_guided"
    frozen["input"]["profile"] = {
        "source_hash": frozen["input"]["parent_source_hash"],
        "hotspots": ["kernel"],
    }
    frozen_path.write_bytes(canonical_bytes(frozen))
    candidate_path = run / "round_1" / "candidates.jsonl"
    candidate = json.loads(candidate_path.read_text())
    candidate["task_type"] = "profile_guided"
    candidate["frozen_input_hash"] = _write_blob(blobs, canonical_bytes(frozen))
    candidate_path.write_text(json.dumps(candidate) + "\n")

    result = build_dataset(_input_manifest(tmp_path, blobs, [run]))

    assert result["accepted"] == 1
