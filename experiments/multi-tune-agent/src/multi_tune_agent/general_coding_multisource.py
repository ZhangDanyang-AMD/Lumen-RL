"""Prepare Multi-SWE Go/JS/TS and ByteDance Rust replay candidates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from multi_swe_bench.harness.instance import Config, Instance
from multi_swe_bench.harness.pull_request import Base, PullRequest

from .general_coding_replay import (
    MULTISWE_HELD_OUT_REPOSITORIES,
    MULTISWE_VERIFIED_REVISION,
    PERMISSIVE_SOURCE_LICENSES,
    _docker_image_identity,
    _patch_paths,
    atomic_write,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_candidates,
    write_json,
    write_jsonl,
)


MULTISWE_SOURCE_ID = "multi_swe_rl_verified"
MULTISWE_DATASET_ID = "PrimeIntellect/Multi-SWE-RL-Verified"
BYTEDANCE_SOURCE_ID = "bytedance_multi_swe_rl_fallback"
BYTEDANCE_DATASET_ID = "ByteDance-Seed/Multi-SWE-RL"
BYTEDANCE_REVISION = "9777648932daa214ba18c70c81e85821b5836f32"
HARNESS_VERSION = "1.1.2"
IMAGE_DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SOURCE_CONFIGS = {
    "multiswe": {
        "source_manifest": "multi-swe-verified-candidate-source-manifest.jsonl",
        "executable_manifest": "multi-swe-verified-executable-manifest.jsonl",
        "image_report": "multi-swe-verified-image-report.json",
        "rejections": "multi-swe-verified-image-rejections.jsonl",
        "verification_dir": "multi-swe-verified-verification",
        "preflight_report": "multi-swe-verified-executable-preflight-report.json",
        "source_id": MULTISWE_SOURCE_ID,
        "dataset_id": MULTISWE_DATASET_ID,
        "dataset_revision": MULTISWE_VERIFIED_REVISION,
        "languages": {"go", "javascript_typescript"},
        "quota_capacity": 125,
    },
    "rust": {
        "source_manifest": "bytedance-multi-swe-rl-fallback-candidate-source-manifest.jsonl",
        "executable_manifest": "bytedance-rust-executable-manifest.jsonl",
        "image_report": "bytedance-rust-image-report.json",
        "rejections": "bytedance-rust-image-rejections.jsonl",
        "verification_dir": "bytedance-rust-verification",
        "preflight_report": "bytedance-rust-preflight-report.json",
        "source_id": BYTEDANCE_SOURCE_ID,
        "dataset_id": BYTEDANCE_DATASET_ID,
        "dataset_revision": BYTEDANCE_REVISION,
        "languages": {"rust"},
        "quota_capacity": 50,
    },
}


def _repository(row: Mapping[str, Any]) -> str:
    repository = str(row.get("upstream_repository", ""))
    prefix = "https://github.com/"
    if not repository.startswith(prefix) or not repository.endswith(".git"):
        raise ValueError("canonical GitHub repository URL required")
    slug = repository[len(prefix) : -4]
    if slug.count("/") != 1:
        raise ValueError("invalid upstream repository")
    return slug


def _raw_multiswe_rows(control_root: Path) -> dict[str, dict[str, Any]]:
    cache = (
        control_root
        / "source-cache"
        / "multi-swe-rl-verified"
        / MULTISWE_VERIFIED_REVISION
    )
    rows: dict[str, dict[str, Any]] = {}
    for language in ("go", "js", "ts"):
        path = cache / f"filter-{language}-00000-00100.json"
        if not path.is_file():
            raise RuntimeError(f"pinned source receipt missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("rows", []):
            row = item.get("row", {})
            if row.get("lang") != language:
                raise RuntimeError(f"{path}: language receipt drift")
            instance_id = str(row.get("instance_id", ""))
            if not instance_id or instance_id in rows:
                raise RuntimeError(f"{path}: duplicate or missing instance_id")
            rows[instance_id] = row
    return rows


def _source_language(
    row: Mapping[str, Any], raw_multiswe: Mapping[str, Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any] | None]:
    if row.get("source_id") != MULTISWE_SOURCE_ID:
        return str(row["primary_language"]), None
    instance_id = str(row["dataset_row_id"])
    raw = raw_multiswe.get(instance_id)
    if raw is None:
        raise ValueError("pinned dataset API row missing")
    if sha256_bytes(canonical_json(raw).encode()) != row.get("source_record_sha256"):
        raise ValueError("pinned dataset API row hash mismatch")
    language = str(raw.get("lang", ""))
    expected = "javascript_typescript" if language in {"js", "ts"} else language
    if expected != row.get("primary_language"):
        raise ValueError("source language does not match frozen candidate")
    return language, raw


def _pull_request(row: Mapping[str, Any], language: str) -> PullRequest:
    repository = _repository(row)
    org, repo = repository.split("/", 1)
    instance_id = str(row["dataset_row_id"])
    statement = str(row["problem_statement"])
    return PullRequest(
        org=org,
        repo=repo,
        number=int(instance_id.rsplit("-", 1)[1]),
        state="closed",
        title=statement.splitlines()[0],
        body=statement,
        base=Base(label="", ref="", sha=str(row["base_commit"])),
        resolved_issues=[],
        fix_patch=str(row["target_patch"]),
        test_patch=str(row["test_patch"]),
        lang=language,
    )


def derive_test_command(row: Mapping[str, Any], language: str) -> tuple[str, str]:
    """Derive the pinned upstream run script and its repository workdir."""
    if importlib.metadata.version("multi-swe-bench") != HARNESS_VERSION:
        raise RuntimeError("multi-swe-bench version drift")
    instance = Instance.create(
        _pull_request(row, language),
        Config(need_clone=False, global_env=None, clear_env=False),
    )
    scripts = [
        file.content.strip()
        for file in instance.dependency().files()
        if file.name == "run.sh"
    ]
    if len(scripts) != 1:
        raise ValueError("expected exactly one harness run.sh")
    command = scripts[0]
    if "git apply" in command or "/home/test.patch" in command:
        raise ValueError("generated test command applies patches")
    workdirs = re.findall(r"(?m)^cd\s+(/[A-Za-z0-9._/-]+)\s*$", command)
    if len(set(workdirs)) != 1:
        raise ValueError("generated test command must select one absolute workdir")
    return command, workdirs[0]


def _execution_command(command: str, language: str) -> str:
    if language == "rust":
        # Docker's `bash -l` replaces the image PATH even though the immutable
        # image config pins Cargo under /usr/local/cargo/bin.
        return "export PATH=/usr/local/cargo/bin:$PATH\n" + command
    if language == "go":
        # The Go images likewise pin their toolchain in Config.Env, which a
        # login shell replaces before the harness command starts.
        return "export PATH=/go/bin:/usr/local/go/bin:$PATH\n" + command
    return command


def _test_names(value: Any) -> list[str]:
    """Read dataset-server's column-oriented test-result representation."""
    if not isinstance(value, Mapping):
        raise ValueError("test evidence must be a mapping")
    names = value.get("name")
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("test evidence names missing")
    return names


def _validate_source_row(
    control_root: Path,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if row.get("source_id") != config["source_id"]:
        raise ValueError("source identity mismatch")
    if row.get("dataset_id") != config["dataset_id"]:
        raise ValueError("dataset identity mismatch")
    if row.get("dataset_revision") != config["dataset_revision"]:
        raise ValueError("dataset revision mismatch")
    if row.get("primary_language") not in config["languages"]:
        raise ValueError("language is outside adapter scope")
    if not FULL_COMMIT.fullmatch(str(row.get("base_commit", ""))):
        raise ValueError("immutable 40-character base commit required")
    if _repository(row) in MULTISWE_HELD_OUT_REPOSITORIES:
        raise ValueError("held-out repository is forbidden")
    if row.get("license_spdx") not in PERMISSIVE_SOURCE_LICENSES:
        raise ValueError("permissive repository license required")
    license_path = control_root / str(row.get("license_evidence_path", ""))
    if not license_path.is_file():
        raise ValueError("frozen repository license evidence missing")
    image = row.get("container_image", {})
    digest = str(image.get("repo_digest", ""))
    if not IMAGE_DIGEST.fullmatch(digest):
        raise ValueError("immutable image repo digest required")
    if image.get("manifest_digest") != digest.rsplit("@", 1)[1]:
        raise ValueError("image manifest digest mismatch")
    for name in ("target_patch", "test_patch"):
        value = str(row.get(name, ""))
        if not value.strip():
            raise ValueError(f"{name} missing")
        if sha256_bytes(value.encode()) != row.get(f"{name}_sha256"):
            raise ValueError(f"{name} hash mismatch")


def _materialize_artifacts(
    control_root: Path, row: dict[str, Any], source_name: str
) -> None:
    root = (
        control_root
        / "source-cache"
        / source_name
        / str(row["dataset_revision"])
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
            raise RuntimeError(f"{row['case_id']}: frozen artifact drift at {name}")
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


def _localize_exact_image(
    expected: Mapping[str, Any], *, pull_timeout: int, allow_pull: bool
) -> tuple[dict[str, Any] | None, bool]:
    repo_tag = str(expected["repo_tag"])
    repo_digest = str(expected["repo_digest"])
    actual = _docker_image_identity(repo_tag)
    if actual is not None and actual["repo_digest"] == repo_digest:
        return actual, False
    if not allow_pull:
        return None, False
    pull = subprocess.run(
        ["docker", "pull", repo_digest],
        capture_output=True,
        text=True,
        timeout=pull_timeout,
        check=False,
    )
    if pull.returncode:
        raise RuntimeError(pull.stderr[-2000:] or pull.stdout[-2000:])
    inspect = subprocess.run(
        ["docker", "image", "inspect", repo_digest],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if inspect.returncode:
        raise RuntimeError(inspect.stderr[-2000:])
    payload = json.loads(inspect.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("ambiguous digest-pulled image identity")
    image = payload[0]
    if repo_digest not in image.get("RepoDigests", []):
        raise RuntimeError("digest-pulled image does not expose expected RepoDigest")
    image_id = str(image.get("Id", ""))
    tag = subprocess.run(
        ["docker", "tag", image_id, repo_tag],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if tag.returncode:
        raise RuntimeError(tag.stderr[-2000:])
    actual = _docker_image_identity(repo_tag)
    if actual is None or actual["repo_digest"] != repo_digest:
        raise RuntimeError("localized image digest mismatch")
    return actual, True


def _prepare_row(
    control_root: Path,
    raw: Mapping[str, Any],
    source_name: str,
    config: Mapping[str, Any],
    raw_multiswe: Mapping[str, Mapping[str, Any]],
    *,
    pull_timeout: int,
    allow_pull: bool,
) -> tuple[dict[str, Any] | None, bool]:
    _validate_source_row(control_root, raw, config)
    language, source_record = _source_language(raw, raw_multiswe)
    command, workdir = derive_test_command(raw, language)
    command = _execution_command(command, language)
    expected = dict(raw["container_image"])
    actual, pulled = _localize_exact_image(
        expected, pull_timeout=pull_timeout, allow_pull=allow_pull
    )
    if actual is None:
        return None, False
    row = dict(raw)
    _materialize_artifacts(control_root, row, source_name)
    target_patch = control_root / str(row["target_patch_path"])
    license_path = control_root / str(row["license_evidence_path"])
    fail_to_pass = list(
        row.get("fail_to_pass")
        or _test_names((source_record or {}).get("f2p_tests"))
    )
    pass_to_pass = list(
        row.get("pass_to_pass")
        or _test_names((source_record or {}).get("p2p_tests"))
    )
    if not fail_to_pass or not pass_to_pass:
        raise ValueError("FAIL_TO_PASS and PASS_TO_PASS evidence required")
    row.update(
        {
            "_manifest_dir": str(control_root.resolve()),
            "container_image": actual,
            "container_workdir": workdir,
            "verification_backend": "docker",
            "allowed_paths": _patch_paths(target_patch.read_text(encoding="utf-8")),
            "install_command": "true",
            "compile_or_typecheck_command": "true",
            "targeted_test_command": command,
            "full_regression_command": "true",
            "test_command_sha256": sha256_bytes(command.encode()),
            "test_set_hash": sha256_bytes(
                canonical_json(
                    {
                        "fail_to_pass": fail_to_pass,
                        "pass_to_pass": pass_to_pass,
                        "test_command_sha256": sha256_bytes(command.encode()),
                    }
                ).encode()
            ),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "license_evidence_sha256": sha256_file(license_path),
            "network_policy": "disabled",
            "gpu_required": False,
            "benchmark_exclusion_gate": "ready",
            "harness": {
                "package": "multi-swe-bench",
                "version": HARNESS_VERSION,
                "upstream_revision": raw.get("upstream_adapter_revision"),
            },
            "timeouts": {
                "checkout_dependency": 120,
                "parent": 3600,
                "target": 3600,
                "regression": 600,
                "fresh_total": 10800,
            },
        }
    )
    return row, pulled


def prepare_executable_candidates(
    control_root: Path,
    source_name: str,
    *,
    limit: int | None = None,
    case_ids: Sequence[str] = (),
    max_pulls: int = 4,
    pull_timeout: int = 1800,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare a deterministic bounded slice, preserving prior rows on resume."""
    if source_name not in SOURCE_CONFIGS:
        raise ValueError(f"unknown source adapter: {source_name}")
    if max_pulls < 0:
        raise ValueError("max_pulls must be non-negative")
    config = SOURCE_CONFIGS[source_name]
    source_path = control_root / str(config["source_manifest"])
    all_rows = [
        row
        for row in read_jsonl(source_path)
        if row.get("primary_language") in config["languages"]
    ]
    source_order = {str(row["case_id"]): index for index, row in enumerate(all_rows)}
    rows = list(all_rows)
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if str(row["case_id"]) in requested]
        missing = sorted(requested - {str(row["case_id"]) for row in rows})
        if missing:
            raise ValueError("unknown case IDs: " + ", ".join(missing))
    if limit is not None:
        rows = rows[:limit]
    manifest = control_root / str(config["executable_manifest"])
    existing = read_jsonl(manifest) if resume and manifest.is_file() else []
    prepared_by_id = {str(row["case_id"]): row for row in existing}
    raw_multiswe = _raw_multiswe_rows(control_root) if source_name == "multiswe" else {}
    rejected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    prepared_scope: set[str] = set()
    pulls = 0
    for raw in rows:
        case_id = str(raw["case_id"])
        try:
            prepared, pulled = _prepare_row(
                control_root,
                raw,
                "multi-swe-rl-verified"
                if source_name == "multiswe"
                else "bytedance-multi-swe-rl",
                config,
                raw_multiswe,
                pull_timeout=pull_timeout,
                allow_pull=pulls < max_pulls,
            )
            if prepared is None:
                deferred.append(
                    {"case_id": case_id, "reason": "bounded_pull_budget_exhausted"}
                )
                continue
            pulls += int(pulled)
            prepared_by_id[case_id] = prepared
            prepared_scope.add(case_id)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            rejected.append({"case_id": case_id, "detail": str(error)[-2000:]})
    frozen = sorted(
        (
            row
            for case_id, row in prepared_by_id.items()
            if case_id in source_order
        ),
        key=lambda row: source_order[str(row["case_id"])],
    )
    write_jsonl(manifest, frozen)
    write_jsonl(control_root / str(config["rejections"]), rejected)
    language_counts = Counter(
        row["primary_language"] for row in frozen
    )
    repository_counts = Counter(_repository(row) for row in frozen)
    report = {
        "schema_version": "general_coding_replay_multisource_images_v1",
        "source_adapter": source_name,
        "source_capacity": len(all_rows),
        "requested": len(rows),
        "strict_executable_plan_capacity": len(rows) - len(rejected),
        "prepared_in_scope": len(prepared_scope),
        "prepared_scope_case_ids": sorted(prepared_scope),
        "prepared_total": len(frozen),
        "pulled_by_digest": pulls,
        "reused_local_or_resumed": len(prepared_scope) - pulls,
        "deferred": len(deferred),
        "rejected": len(rejected),
        "image_availability": (
            "all_requested_local"
            if len(prepared_scope) == len(rows)
            else "partial_or_unavailable"
        ),
        "prepared_by_language": dict(sorted(language_counts.items())),
        "prepared_by_repository": dict(sorted(repository_counts.items())),
        "manifest": {
            "path": manifest.name,
            "rows": len(frozen),
            "sha256": sha256_file(manifest),
        },
        "bounded_pull_limit": max_pulls,
        "resume": resume,
        "deferred_cases": deferred,
    }
    write_json(control_root / str(config["image_report"]), report)
    return report


def verify_executable_candidates(
    control_root: Path,
    source_name: str,
    **prepare_options: Any,
) -> dict[str, Any]:
    config = SOURCE_CONFIGS[source_name]
    image_report = prepare_executable_candidates(
        control_root, source_name, **prepare_options
    )
    if image_report["prepared_in_scope"] != image_report["requested"]:
        return {"status": "blocked_image_gate", "image_report": image_report}
    executable_manifest = control_root / str(config["executable_manifest"])
    selected_ids = set(image_report["prepared_scope_case_ids"])
    selected_rows = [
        row
        for row in read_jsonl(executable_manifest)
        if str(row["case_id"]) in selected_ids
    ]
    verification_manifest = executable_manifest.with_name(
        executable_manifest.stem + "-preflight.jsonl"
    )
    write_jsonl(verification_manifest, selected_rows)
    verification = verify_candidates(
        verification_manifest,
        control_root / str(config["verification_dir"]) / "verified",
        resume=True,
    )
    report = {
        "schema_version": "general_coding_replay_multisource_preflight_v1",
        "status": "passed" if verification["rejected"] == 0 else "failed",
        "image_report": image_report,
        "verification_manifest": {
            "path": verification_manifest.name,
            "rows": len(selected_rows),
            "sha256": sha256_file(verification_manifest),
        },
        "verification": verification,
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / str(config["preflight_report"]), report)
    return report


def build_capacity_report(control_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "general_coding_replay_multisource_capacity_v1",
        "accepted_jsonl_mutated": False,
        "starting_verified_deficit": 370,
        "sources": {},
    }
    quota_ceiling = 0
    for source_name, config in SOURCE_CONFIGS.items():
        source_rows = [
            row
            for row in read_jsonl(control_root / str(config["source_manifest"]))
            if row.get("primary_language") in config["languages"]
        ]
        manifest = control_root / str(config["executable_manifest"])
        executable = read_jsonl(manifest) if manifest.is_file() else []
        image_report_path = control_root / str(config["image_report"])
        image_report = json.loads(image_report_path.read_text(encoding="utf-8"))
        verified_root = control_root / str(config["verification_dir"]) / "verified"
        verified = list(verified_root.glob("*/verified.json"))
        strict_capacity = int(image_report["strict_executable_plan_capacity"])
        source_quota_ceiling = min(int(config["quota_capacity"]), strict_capacity)
        quota_ceiling += source_quota_ceiling
        report["sources"][source_name] = {
            "source_candidates": len(source_rows),
            "strict_executable_plan_capacity": strict_capacity,
            "quota_admissible_ceiling": source_quota_ceiling,
            "executable_prepared": len(executable),
            "double_replay_verified": len(verified),
            "remaining_unprepared": len(source_rows) - len(executable),
        }
    report["combined_strict_executable_plan_capacity"] = sum(
        source["strict_executable_plan_capacity"]
        for source in report["sources"].values()
    )
    report["combined_quota_admissible_ceiling"] = quota_ceiling
    report["best_case_remaining_verified_deficit"] = (
        report["starting_verified_deficit"] - quota_ceiling
    )
    write_json(control_root / "multisource-executable-capacity-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--source", choices=tuple(SOURCE_CONFIGS), required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--limit", type=int)
        command.add_argument("--case-id", action="append", default=[])
        command.add_argument("--max-pulls", type=int, default=4)
        command.add_argument("--pull-timeout", type=int, default=1800)
        command.add_argument("--resume", action="store_true")
    subparsers.add_parser("capacity")
    args = parser.parse_args(argv)
    if args.command == "capacity":
        result = build_capacity_report(args.control_root)
    else:
        options = {
            "limit": args.limit,
            "case_ids": args.case_id,
            "max_pulls": args.max_pulls,
            "pull_timeout": args.pull_timeout,
            "resume": args.resume,
        }
        if args.command == "prepare":
            result = prepare_executable_candidates(
                args.control_root, args.source, **options
            )
        else:
            result = verify_executable_candidates(
                args.control_root, args.source, **options
            )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
