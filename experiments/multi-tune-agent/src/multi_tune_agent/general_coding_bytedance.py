"""Prepare and verify frozen ByteDance Multi-SWE-RL C++ replay rows."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from multi_swe_bench.harness.instance import Config, Instance, PullRequest
from multi_swe_bench.harness.pull_request import Base

from .general_coding_replay import (
    _docker_image_identity,
    _patch_paths,
    atomic_write,
    canonical_json,
    read_jsonl,
    sha256_file,
    verify_candidates,
    write_json,
    write_jsonl,
)


DATASET_ID = "ByteDance-Seed/Multi-SWE-RL"
DATASET_REVISION = "9777648932daa214ba18c70c81e85821b5836f32"
SOURCE_ID = "bytedance_multi_swe_rl_fallback"
CPP_DEFICIT = 26
REPOSITORY_CAP = 15
LINEAGE_CAP = 2
WORKDIRS = {
    "bitcoin/bitcoin": "/home/bitcoin",
    "catchorg/Catch2": "/home/Catch2",
    "halide/Halide": "/home/Halide",
    "yhirose/cpp-httplib": "/home/cpp-httplib",
}


def _repository_name(row: Mapping[str, Any]) -> str:
    repository = str(row["upstream_repository"])
    name = repository.removeprefix("https://github.com/").removesuffix(".git")
    if name not in WORKDIRS:
        raise ValueError(f"unsupported ByteDance C++ repository: {name}")
    return name


def _pull_request(row: Mapping[str, Any]) -> PullRequest:
    repository = _repository_name(row)
    org, repo = repository.split("/", 1)
    instance_id = str(row["dataset_row_id"])
    number = int(instance_id.rsplit("-", 1)[1])
    statement = str(row["problem_statement"])
    return PullRequest(
        org=org,
        repo=repo,
        number=number,
        state="closed",
        title=statement.splitlines()[0],
        body=statement,
        base=Base(label="", ref="", sha=str(row["base_commit"])),
        resolved_issues=[],
        fix_patch=str(row["target_patch"]),
        test_patch=str(row["test_patch"]),
        lang="cpp",
    )


def derive_test_command(row: Mapping[str, Any]) -> str:
    """Freeze the pinned multi-swe-bench run script without patch application."""
    instance = Instance.create(
        _pull_request(row),
        Config(need_clone=False, global_env=None, clear_env=False),
    )
    scripts = [
        file.content
        for file in instance.dependency().files()
        if file.name == "run.sh"
    ]
    if len(scripts) != 1:
        raise ValueError(f"{row['case_id']}: expected exactly one run.sh")
    command = scripts[0].strip()
    if "git apply" in command or "/home/test.patch" in command:
        raise ValueError(f"{row['case_id']}: generated command applies patches")
    return command


def _materialize_artifacts(control_root: Path, row: dict[str, Any]) -> None:
    artifact_root = (
        control_root
        / "source-cache"
        / "bytedance-multi-swe-rl"
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
        path = artifact_root / name
        if path.is_file() and path.read_bytes() != content:
            raise RuntimeError(f"{row['case_id']}: frozen artifact drift at {name}")
        if not path.is_file():
            atomic_write(path, content)
    row.update(
        {
            "target_patch_path": (artifact_root / "target.patch")
            .relative_to(control_root)
            .as_posix(),
            "test_patch_path": (artifact_root / "test.patch")
            .relative_to(control_root)
            .as_posix(),
            "problem_statement_path": (artifact_root / "problem.md")
            .relative_to(control_root)
            .as_posix(),
        }
    )


def prepare_bytedance_cpp_images(
    control_root: Path,
    *,
    limit: int | None = None,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    source = (
        control_root
        / "bytedance-multi-swe-rl-fallback-candidate-source-manifest.jsonl"
    )
    rows = [
        row for row in read_jsonl(source) if row.get("primary_language") == "cpp"
    ]
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if str(row["case_id"]) in requested]
        missing = sorted(requested - {str(row["case_id"]) for row in rows})
        if missing:
            raise ValueError("unknown ByteDance C++ case IDs: " + ", ".join(missing))
    if limit is not None:
        rows = sorted(rows, key=lambda row: str(row["case_id"]))[:limit]
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        try:
            command = derive_test_command(row)
            _materialize_artifacts(control_root, row)
            expected = dict(row["container_image"])
            pull = subprocess.run(
                ["docker", "pull", str(expected["repo_tag"])],
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            if pull.returncode:
                raise RuntimeError(pull.stderr[-2000:])
            actual = _docker_image_identity(str(expected["repo_tag"]))
            if actual is None or actual["repo_digest"] != expected["repo_digest"]:
                raise RuntimeError("image digest mismatch")
            repository = _repository_name(row)
            patch_path = control_root / str(row["target_patch_path"])
            row.update(
                {
                    "_manifest_dir": str(control_root.resolve()),
                    "container_image": actual,
                    "container_workdir": WORKDIRS[repository],
                    "verification_backend": "docker",
                    "allowed_paths": _patch_paths(
                        patch_path.read_text(encoding="utf-8")
                    ),
                    "install_command": "true",
                    "compile_or_typecheck_command": "true",
                    "targeted_test_command": command,
                    "full_regression_command": "true",
                    "network_policy": "disabled",
                    "benchmark_exclusion_gate": "ready",
                    "timeouts": {
                        "checkout_dependency": 120,
                        "parent": 1800,
                        "target": 1800,
                        "regression": 600,
                        "fresh_total": 7200,
                    },
                }
            )
            prepared.append(row)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            rejected.append({"case_id": raw["case_id"], "detail": str(error)[-2000:]})
    manifest = control_root / "bytedance-cpp-executable-manifest.jsonl"
    write_jsonl(manifest, prepared)
    write_jsonl(control_root / "bytedance-cpp-image-rejections.jsonl", rejected)
    report = {
        "schema_version": "general_coding_replay_bytedance_cpp_images_v1",
        "requested": len(rows),
        "prepared": len(prepared),
        "rejected": len(rejected),
        "manifest": {
            "path": manifest.name,
            "rows": len(prepared),
            "sha256": sha256_file(manifest),
        },
    }
    write_json(control_root / "bytedance-cpp-image-report.json", report)
    return report


def verify_bytedance_cpp(
    control_root: Path,
    *,
    limit: int | None,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    image_report = prepare_bytedance_cpp_images(
        control_root, limit=limit, case_ids=case_ids
    )
    if image_report["prepared"] != image_report["requested"]:
        return {
            "status": "blocked_image_gate",
            "image_report": image_report,
            "verified_rows": 0,
        }
    verification = verify_candidates(
        control_root / "bytedance-cpp-executable-manifest.jsonl",
        control_root / "bytedance-cpp-verification" / "verified",
        resume=True,
    )
    report = {
        "schema_version": "general_coding_replay_bytedance_cpp_preflight_v1",
        "status": (
            "passed"
            if verification["verified"] == image_report["prepared"]
            and verification["rejected"] == 0
            else "failed"
        ),
        "scope": "bounded" if limit is not None or case_ids else "full_43",
        "image_report": image_report,
        "verification": verification,
        "verified_rows": verification["verified"],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "bytedance-cpp-preflight-report.json", report)
    return report


def build_bytedance_cpp_capacity_report(control_root: Path) -> dict[str, Any]:
    verified_root = control_root / "bytedance-cpp-verification" / "verified"
    rows = []
    for path in sorted(verified_root.glob("*/verified.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("status") == "verified"
            and row.get("source_id") == SOURCE_ID
            and row.get("dataset_revision") == DATASET_REVISION
            and row.get("primary_language") == "cpp"
        ):
            rows.append(row)
    admitted = []
    repositories: Counter[str] = Counter()
    lineages: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: str(item["case_id"])):
        repository = str(row["upstream_repository"])
        lineage = str(row["source_lineage_id"])
        if (
            repositories[repository] >= REPOSITORY_CAP
            or lineages[lineage] >= LINEAGE_CAP
        ):
            continue
        admitted.append(row)
        repositories[repository] += 1
        lineages[lineage] += 1
        if len(admitted) == CPP_DEFICIT:
            break
    manifest = control_root / "bytedance-cpp-admitted.jsonl"
    write_jsonl(manifest, admitted)
    report = {
        "schema_version": "general_coding_replay_bytedance_cpp_capacity_v1",
        "status": (
            "exact_26_ready_for_formal_integration"
            if len(admitted) == CPP_DEFICIT
            else "verified_capacity_deficit"
        ),
        "verified_rows_available": len(rows),
        "admitted_capacity": len(admitted),
        "remaining_cpp_deficit": CPP_DEFICIT - len(admitted),
        "selected_by_repository": dict(sorted(repositories.items())),
        "repository_cap": REPOSITORY_CAP,
        "lineage_cap": LINEAGE_CAP,
        "manifest": {
            "path": manifest.name,
            "rows": len(admitted),
            "sha256": sha256_file(manifest),
        },
        "formal_quota_files_mutated": False,
    }
    write_json(control_root / "bytedance-cpp-capacity-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--case-id", action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--limit", type=int)
    verify.add_argument("--case-id", action="append", default=[])
    subparsers.add_parser("capacity")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare_bytedance_cpp_images(
            args.control_root, limit=args.limit, case_ids=args.case_id
        )
    elif args.command == "verify":
        report = verify_bytedance_cpp(
            args.control_root, limit=args.limit, case_ids=args.case_id
        )
    else:
        report = build_bytedance_cpp_capacity_report(args.control_root)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
