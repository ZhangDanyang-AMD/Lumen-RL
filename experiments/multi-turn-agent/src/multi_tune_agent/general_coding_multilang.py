"""Freeze and verify SWE-bench-Live MultiLang C++ replay candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .general_coding_quota_override import (
    _docker_hub_digest,
    _exclusion_dimensions,
    overlap_reasons,
)
from .general_coding_rebench import (
    _freeze_license,
    _normalized_patch,
    _supplemental_exclusions,
    _write_frozen_jsonl,
)
from .general_coding_replay import (
    _docker_image_identity,
    _patch_paths,
    atomic_write,
    canonical_json,
    derive_repository_task_type,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_candidates,
    write_json,
    write_jsonl,
)


DATASET_ID = "SWE-bench-Live/MultiLang"
DATASET_REVISION = "22091f6ce331c5c60c241d76512d4be7ee1a555b"
PARQUET_SHA256 = "8448db887817b63e4c0c284ca99de1ccda15023f48e5b2234a4084466e0768ae"
SOURCE_ID = "swe_bench_live_multilang_cpp"
TARGET_REPOSITORIES = {
    "BYVoid/OpenCC": "Apache-2.0",
    "duckdb/duckdb": "MIT",
}
REPOSITORY_CAP = 15
TARGET_ROWS = 26
NEEDED_ROWS = 21


def adapt_multilang_row(row: Mapping[str, Any]) -> dict[str, Any]:
    repository = str(row.get("repo", ""))
    if repository not in TARGET_REPOSITORIES:
        raise ValueError("repository not selected")
    instance_id = str(row.get("instance_id", ""))
    commit = str(row.get("base_commit", ""))
    patch = str(row.get("patch", ""))
    test_patch = str(row.get("test_patch", ""))
    f2p = [str(value) for value in row.get("FAIL_TO_PASS") or []]
    p2p = [str(value) for value in row.get("PASS_TO_PASS") or []]
    rebuild = [str(value) for value in row.get("rebuild_cmds") or []]
    tests = [str(value) for value in row.get("test_cmds") or []]
    image_name = str(row.get("docker_image", ""))
    if (
        len(commit) != 40
        or not instance_id
        or not patch.strip()
        or not test_patch.strip()
        or not f2p
        or not p2p
        or not rebuild
        or not tests
        or not image_name
    ):
        raise ValueError("strict executable evidence missing")
    task_type, task_evidence = derive_repository_task_type(
        {
            "title": str(row.get("problem_statement", "")).splitlines()[0],
            "body": str(row.get("problem_statement", "")),
            "fix_patch": patch,
            "test_patch": test_patch,
        }
    )
    return {
        "schema_version": "general_coding_replay_candidate_source_v3",
        "case_id": f"gc-replay-{SOURCE_ID}-{instance_id}",
        "source_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_row_id": instance_id,
        "source_record_sha256": sha256_bytes(canonical_json(row).encode()),
        "source_lineage_id": f"{DATASET_ID}@{DATASET_REVISION}:{instance_id}",
        "upstream_repository": f"https://github.com/{repository}.git",
        "repository": repository,
        "base_commit": commit,
        "problem_id": instance_id,
        "problem_statement": str(row.get("problem_statement", "")),
        "target_patch": patch,
        "target_patch_sha256": sha256_bytes(patch.encode()),
        "normalized_patch_hash": sha256_bytes(
            _normalized_patch(patch).encode()
        ),
        "test_patch": test_patch,
        "test_patch_sha256": sha256_bytes(test_patch.encode()),
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "test_set_hash": sha256_bytes(
            canonical_json({"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}).encode()
        ),
        "rebuild_commands": rebuild,
        "test_commands": tests,
        "image_name": image_name,
        "primary_language": "cpp",
        "primary_task_type": task_type,
        "task_type_evidence": task_evidence,
        "dataset_license_spdx": TARGET_REPOSITORIES[repository],
        "network_policy": "disabled",
        "gpu_required": False,
        "verified": False,
        "local_replay_passes": 0,
    }


def _materialize_artifacts(control_root: Path, row: dict[str, Any]) -> None:
    root = (
        control_root
        / "source-cache"
        / "swe-bench-live-multilang"
        / DATASET_REVISION
        / "candidates"
        / str(row["problem_id"])
    )
    artifacts = {
        "target.patch": str(row.pop("target_patch")).encode(),
        "test.patch": str(row.pop("test_patch")).encode(),
        "problem.md": str(row.pop("problem_statement")).encode(),
    }
    for name, content in artifacts.items():
        path = root / name
        if path.is_file() and path.read_bytes() != content:
            raise RuntimeError(f"{row['case_id']}: frozen artifact drift")
        if not path.is_file():
            atomic_write(path, content)
    row.update(
        {
            "target_patch_path": (root / "target.patch")
            .relative_to(control_root)
            .as_posix(),
            "test_patch_path": (root / "test.patch")
            .relative_to(control_root)
            .as_posix(),
            "problem_statement_path": (root / "problem.md")
            .relative_to(control_root)
            .as_posix(),
        }
    )


def freeze_multilang_cpp(control_root: Path) -> dict[str, Any]:
    cache = control_root / "source-cache" / "swe-bench-live-multilang" / DATASET_REVISION
    parquet = cache / "cpp-00000-of-00001.parquet"
    if not parquet.is_file() or sha256_file(parquet) != PARQUET_SHA256:
        raise RuntimeError("pinned MultiLang C++ parquet missing or drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for MultiLang freeze") from exc
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    dimensions = _exclusion_dimensions(
        registry, _supplemental_exclusions(control_root)
    )
    eligible = []
    rejections = []
    for raw in pq.read_table(parquet).to_pylist():
        try:
            candidate = adapt_multilang_row(raw)
        except ValueError as error:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "schema_or_repository",
                    "detail": str(error),
                }
            )
            continue
        reasons = overlap_reasons(candidate, dimensions)
        if reasons:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "overlap",
                    "dimensions": reasons,
                }
            )
            continue
        eligible.append(candidate)
    selected = []
    counts: Counter[str] = Counter()
    for candidate in sorted(eligible, key=lambda row: str(row["case_id"])):
        repository = str(candidate["repository"])
        if counts[repository] >= REPOSITORY_CAP:
            continue
        selected.append(candidate)
        counts[repository] += 1
    if len(selected) < TARGET_ROWS:
        raise RuntimeError(
            f"MultiLang strict capacity is {len(selected)}, expected {TARGET_ROWS}"
        )
    selected = selected[:TARGET_ROWS]
    license_cache: dict[tuple[str, str], dict[str, Any]] = {}
    image_cache: dict[str, dict[str, str]] = {}
    frozen = []
    for candidate in selected:
        key = (str(candidate["repository"]), str(candidate["base_commit"]))
        if key not in license_cache:
            license_cache[key] = _freeze_license(control_root, candidate)
        license_evidence = license_cache[key]
        image_name = str(candidate.pop("image_name"))
        if image_name not in image_cache:
            image_cache[image_name] = _docker_hub_digest(image_name)
        candidate.update(
            {
                "container_image": image_cache[image_name],
                "license_spdx": license_evidence["spdx"],
                "license_evidence_path": license_evidence[
                    "license_evidence_path"
                ],
                "license_evidence_sha256": license_evidence[
                    "license_evidence_sha256"
                ],
            }
        )
        _materialize_artifacts(control_root, candidate)
        candidate["_manifest_dir"] = str(control_root.resolve())
        frozen.append(candidate)
    manifest = control_root / "multilang-cpp-candidate-source-manifest.jsonl"
    _write_frozen_jsonl(manifest, frozen)
    write_jsonl(control_root / "multilang-cpp-freeze-rejections.jsonl", rejections)
    report = {
        "schema_version": "general_coding_replay_multilang_cpp_freeze_v1",
        "status": "strict_candidate_pool_ready",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "parquet_sha256": PARQUET_SHA256,
        "source_rows": 142,
        "eligible_rows": len(eligible),
        "selected_rows": len(frozen),
        "selected_by_repository": dict(sorted(counts.items())),
        "overlap_rejections": sum(
            row["reason"] == "overlap" for row in rejections
        ),
        "manifest": {
            "path": manifest.name,
            "rows": len(frozen),
            "sha256": sha256_file(manifest),
        },
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "multilang-cpp-freeze-report.json", report)
    return report


def prepare_multilang_cpp(
    control_root: Path,
    *,
    limit: int | None = None,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    rows = read_jsonl(control_root / "multilang-cpp-candidate-source-manifest.jsonl")
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if str(row["case_id"]) in requested]
        missing = sorted(requested - {str(row["case_id"]) for row in rows})
        if missing:
            raise ValueError("unknown MultiLang case IDs: " + ", ".join(missing))
    if limit is not None:
        rows = rows[:limit]
    prepared = []
    rejected = []
    for raw in rows:
        row = dict(raw)
        expected = dict(row["container_image"])
        pull = subprocess.run(
            ["docker", "pull", str(expected["repo_tag"])],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        if pull.returncode:
            rejected.append(
                {"case_id": row["case_id"], "detail": pull.stderr[-2000:]}
            )
            continue
        actual = _docker_image_identity(str(expected["repo_tag"]))
        if actual is None or actual["repo_digest"] != expected["repo_digest"]:
            rejected.append(
                {"case_id": row["case_id"], "detail": "image digest mismatch"}
            )
            continue
        patch_path = control_root / str(row["target_patch_path"])
        rebuild = " && ".join(str(cmd) for cmd in row.pop("rebuild_commands"))
        tests = " && ".join(str(cmd) for cmd in row.pop("test_commands"))
        row.update(
            {
                "container_image": actual,
                "container_workdir": str(actual["working_dir"]),
                "verification_backend": "docker",
                "allowed_paths": _patch_paths(
                    patch_path.read_text(encoding="utf-8")
                ),
                "install_command": "true",
                "compile_or_typecheck_command": rebuild,
                "targeted_test_command": f"set -o pipefail; {tests}",
                "full_regression_command": "true",
                "benchmark_exclusion_gate": "ready",
                "timeouts": {
                    "checkout_dependency": 120,
                    "parent": 3600,
                    "target": 3600,
                    "regression": 600,
                    "fresh_total": 10800,
                },
            }
        )
        prepared.append(row)
    manifest = control_root / "multilang-cpp-executable-manifest.jsonl"
    write_jsonl(manifest, prepared)
    write_jsonl(control_root / "multilang-cpp-image-rejections.jsonl", rejected)
    report = {
        "schema_version": "general_coding_replay_multilang_cpp_images_v1",
        "requested": len(rows),
        "prepared": len(prepared),
        "rejected": len(rejected),
        "manifest": {
            "path": manifest.name,
            "rows": len(prepared),
            "sha256": sha256_file(manifest),
        },
    }
    write_json(control_root / "multilang-cpp-image-report.json", report)
    return report


def verify_multilang_cpp(
    control_root: Path,
    *,
    limit: int | None,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    image_report = prepare_multilang_cpp(
        control_root, limit=limit, case_ids=case_ids
    )
    if image_report["prepared"] != image_report["requested"]:
        return {"status": "blocked_image_gate", "image_report": image_report}
    verification = verify_candidates(
        control_root / "multilang-cpp-executable-manifest.jsonl",
        control_root / "multilang-cpp-verification" / "verified",
        resume=True,
    )
    report = {
        "schema_version": "general_coding_replay_multilang_cpp_preflight_v1",
        "status": (
            "passed"
            if verification["verified"] == image_report["prepared"]
            and verification["rejected"] == 0
            else "failed"
        ),
        "scope": "bounded" if limit is not None or case_ids else "full_26",
        "image_report": image_report,
        "verification": verification,
        "verified_rows": verification["verified"],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "multilang-cpp-preflight-report.json", report)
    return report


def build_multilang_capacity_report(control_root: Path) -> dict[str, Any]:
    root = control_root / "multilang-cpp-verification" / "verified"
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/verified.json"))
    ]
    rows = [
        row
        for row in rows
        if row.get("status") == "verified"
        and row.get("source_id") == SOURCE_ID
        and row.get("dataset_revision") == DATASET_REVISION
    ]
    admitted = rows[:NEEDED_ROWS]
    manifest = control_root / "multilang-cpp-admitted.jsonl"
    write_jsonl(manifest, admitted)
    report = {
        "schema_version": "general_coding_replay_multilang_cpp_capacity_v1",
        "status": (
            "exact_21_ready_for_formal_integration"
            if len(admitted) == NEEDED_ROWS
            else "verified_capacity_deficit"
        ),
        "verified_rows_available": len(rows),
        "admitted_capacity": len(admitted),
        "remaining_cpp_deficit": NEEDED_ROWS - len(admitted),
        "manifest": {
            "path": manifest.name,
            "rows": len(admitted),
            "sha256": sha256_file(manifest),
        },
        "formal_quota_files_mutated": False,
    }
    write_json(control_root / "multilang-cpp-capacity-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    for name in ("prepare", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--limit", type=int)
        command.add_argument("--case-id", action="append", default=[])
    subparsers.add_parser("capacity")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = freeze_multilang_cpp(args.control_root)
    elif args.command == "prepare":
        report = prepare_multilang_cpp(
            args.control_root, limit=args.limit, case_ids=args.case_id
        )
    elif args.command == "verify":
        report = verify_multilang_cpp(
            args.control_root, limit=args.limit, case_ids=args.case_id
        )
    else:
        report = build_multilang_capacity_report(args.control_root)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
