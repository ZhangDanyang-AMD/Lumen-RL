"""Pinned SWE-rebench V2 C++ fallback freeze and offline verification."""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from .general_coding_quota_override import (
    _docker_hub_digest,
    _exclusion_dimensions,
    _get_json,
    overlap_reasons,
)
from .general_coding_replay import (
    PERMISSIVE_SOURCE_LICENSES,
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


DATASET_ID = "nebius/SWE-rebench-V2"
DATASET_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
PARQUET_SHA256 = "0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad"
DATASET_LICENSE_SHA256 = "fe2ae9b4d7a68d4e4bf48bbc103318eb7b697455d56fd78c9ffe87a34da5668c"
SOURCE_ID = "swe_rebench_v2_fallback"
TARGET_ROWS = 32
REPOSITORY_CAP = 15
LINEAGE_CAP = 2
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ZEek_LICENSE = {
    "repository": "zeek/zeek",
    "spdx": "BSD-3-Clause",
    "license_file": "COPYING",
}


def _write_frozen_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(row) + "\n" for row in rows).encode()
    if path.is_file() and path.read_bytes() != content:
        raise RuntimeError(f"frozen manifest drift: {path}")
    if not path.is_file():
        atomic_write(path, content)


def _normalized_patch(value: str) -> str:
    return "\n".join(
        line.rstrip() for line in value.replace("\r\n", "\n").splitlines()
    )


def _supplemental_exclusions(control_root: Path) -> list[dict[str, Any]]:
    rows = []
    paths = (
        control_root / "general-coding-dev-reservations.jsonl",
        control_root.parent / "general-coding-held-out-v1" / "reservations.jsonl",
    )
    for path in paths:
        if not path.is_file():
            continue
        for raw in read_jsonl(path):
            protected = raw.get("protected") or {}
            rows.append(
                {
                    "upstream_repository": raw.get("upstream_repository")
                    or raw.get("repository"),
                    "problem_id": raw.get("problem_id"),
                    "source_lineage_id": raw.get("source_lineage_id"),
                    "normalized_patch_hash": raw.get("normalized_patch_hash")
                    or protected.get("normalized_solution_patch_sha256"),
                    "test_set_hash": raw.get("test_set_hash")
                    or protected.get("test_patch_sha256"),
                }
            )
    return rows


def adapt_rebench_cpp_row(
    row: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Adapt one pinned row without claiming image or replay verification."""
    if row.get("language") != "cpp":
        raise ValueError("row is not C++")
    instance_id = str(row.get("instance_id", ""))
    repository = str(row.get("repo", ""))
    commit = str(row.get("base_commit", ""))
    patch = str(row.get("patch", ""))
    test_patch = str(row.get("test_patch", ""))
    f2p = [str(value) for value in (row.get("FAIL_TO_PASS") or [])]
    p2p = [str(value) for value in (row.get("PASS_TO_PASS") or [])]
    install_config = row.get("install_config") or {}
    test_command = str(install_config.get("test_cmd", "")).strip()
    if not instance_id or "/" not in repository:
        raise ValueError("immutable row identity missing")
    if not FULL_COMMIT.fullmatch(commit):
        raise ValueError("exact base commit missing")
    if not patch.strip() or not test_patch.strip():
        raise ValueError("gold patch or test patch missing")
    if not f2p or not p2p or not test_command:
        raise ValueError("executable F2P/P2P evidence missing")
    image_name = str(row.get("image_name", "")).removeprefix("docker.io/")
    if not image_name or ":" not in image_name:
        raise ValueError("container image tag missing")
    task_type, task_evidence = derive_repository_task_type(
        {
            "title": str(row.get("problem_statement", "")).splitlines()[0],
            "body": str(row.get("problem_statement", "")),
            "fix_patch": patch,
            "test_patch": test_patch,
        }
    )
    target_hash = sha256_bytes(patch.encode())
    test_patch_hash = sha256_bytes(test_patch.encode())
    normalized_patch_hash = sha256_bytes(_normalized_patch(patch).encode())
    lineage = f"{DATASET_ID}@{DATASET_REVISION}:{instance_id}"
    return {
        "schema_version": "general_coding_replay_candidate_source_v3",
        "case_id": f"gc-replay-{SOURCE_ID}-{instance_id}",
        "source_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_row_id": instance_id,
        "source_record_sha256": sha256_bytes(canonical_json(row).encode()),
        "source_file_sha256": source_sha256,
        "source_lineage_id": lineage,
        "upstream_repository": f"https://github.com/{repository}.git",
        "repository": repository,
        "base_commit": commit,
        "problem_id": instance_id,
        "problem_statement": str(row.get("problem_statement", "")),
        "target_patch": patch,
        "target_patch_sha256": target_hash,
        "normalized_patch_hash": normalized_patch_hash,
        "test_patch": test_patch,
        "test_patch_sha256": test_patch_hash,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "test_set_hash": sha256_bytes(
            canonical_json(
                {
                    "FAIL_TO_PASS": f2p,
                    "PASS_TO_PASS": p2p,
                    "test_patch_sha256": test_patch_hash,
                }
            ).encode()
        ),
        "primary_language": "cpp",
        "primary_task_type": task_type,
        "task_type_evidence": task_evidence,
        "dataset_license_spdx": str(row.get("license", "")),
        "image_name": image_name,
        "test_command": test_command,
        "status": "candidate_pending_image_and_local_double_replay",
        "verified": False,
        "local_replay_passes": 0,
        "network_policy": "disabled",
        "gpu_required": False,
    }


def _candidate_order(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        sha256_bytes(
            f"{row['dataset_revision']}\0{row['problem_id']}\0"
            f"{row['normalized_patch_hash']}".encode()
        ),
        str(row["case_id"]),
    )


def _license_claim(row: Mapping[str, Any]) -> tuple[str, str | None]:
    repository = str(row["repository"])
    claimed = str(row["dataset_license_spdx"])
    if claimed in PERMISSIVE_SOURCE_LICENSES:
        return claimed, None
    if repository == ZEek_LICENSE["repository"]:
        return str(ZEek_LICENSE["spdx"]), str(ZEek_LICENSE["license_file"])
    raise ValueError(f"{repository}: non-allowlisted license claim {claimed!r}")


def _freeze_license(
    control_root: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    repository = str(row["repository"])
    commit = str(row["base_commit"])
    spdx, fixed_file = _license_claim(row)
    evidence_root = control_root / "license-evidence" / "swe-rebench-v2"
    evidence_root.mkdir(parents=True, exist_ok=True)
    stem = repository.replace("/", "__")
    if fixed_file:
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{fixed_file}"
        response = requests.get(url, timeout=(30, 120))
        response.raise_for_status()
        content = response.content
        source = {
            "method": "immutable_known_license_file",
            "source_url": url,
            "license_file": fixed_file,
        }
    else:
        api_path = evidence_root / f"{stem}.{commit}.github-license-api.json"
        payload = _get_json(
            f"https://api.github.com/repos/{repository}/license?ref={commit}",
            api_path,
        )
        observed = str((payload.get("license") or {}).get("spdx_id", ""))
        if observed != spdx or observed not in PERMISSIVE_SOURCE_LICENSES:
            raise RuntimeError(
                f"{repository}@{commit}: license {observed!r} does not match {spdx!r}"
            )
        content = base64.b64decode(str(payload["content"]), validate=False)
        source = {
            "method": "github_license_api_at_commit",
            "api_evidence_path": api_path.relative_to(control_root).as_posix(),
            "api_evidence_sha256": sha256_file(api_path),
            "license_file": str(payload.get("path", "")),
            "source_url": str(payload.get("html_url", "")),
        }
    license_path = evidence_root / f"{stem}.{commit}.LICENSE"
    if license_path.is_file() and license_path.read_bytes() != content:
        raise RuntimeError(f"{repository}@{commit}: frozen license drift")
    if not license_path.is_file():
        atomic_write(license_path, content)
    provenance = {
        "schema_version": "immutable_upstream_license_evidence_v1",
        "repository": repository,
        "base_commit": commit,
        "spdx": spdx,
        "license_evidence_path": license_path.relative_to(control_root).as_posix(),
        "license_evidence_sha256": sha256_file(license_path),
        **source,
    }
    provenance_path = evidence_root / f"{stem}.{commit}.license-source.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError(f"{repository}@{commit}: license provenance drift")
    else:
        write_json(provenance_path, provenance)
    return provenance


def _materialize_artifacts(control_root: Path, row: dict[str, Any]) -> None:
    artifact_root = (
        control_root
        / "source-cache"
        / "swe-rebench-v2"
        / DATASET_REVISION
        / "candidates"
        / str(row["problem_id"])
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
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


def freeze_rebench_cpp_candidates(control_root: Path) -> dict[str, Any]:
    """Freeze an exact 32-row metadata-ready pool and prove bounded capacity."""
    cache = control_root / "source-cache" / "swe-rebench-v2" / DATASET_REVISION
    dataset_api = _get_json(
        f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{DATASET_REVISION}",
        cache / "dataset-api.json",
    )
    if dataset_api.get("sha") != DATASET_REVISION:
        raise RuntimeError("SWE-rebench V2 dataset revision drift")
    parquet = cache / "train-00000-of-00001.parquet"
    if not parquet.is_file() or sha256_file(parquet) != PARQUET_SHA256:
        raise RuntimeError("pinned SWE-rebench V2 parquet missing or drifted")
    dataset_license = cache / "DATASET.LICENSE"
    if (
        not dataset_license.is_file()
        or sha256_file(dataset_license) != DATASET_LICENSE_SHA256
    ):
        raise RuntimeError("pinned SWE-rebench V2 dataset license missing or drifted")
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for SWE-rebench freeze") from exc
    table = pq.read_table(parquet)
    cpp_rows = table.filter(pc.equal(table["language"], "cpp")).to_pylist()
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    dimensions = _exclusion_dimensions(
        registry, _supplemental_exclusions(control_root)
    )
    eligible = []
    rejections = []
    for raw in cpp_rows:
        try:
            candidate = adapt_rebench_cpp_row(raw, source_sha256=PARQUET_SHA256)
            _license_claim(candidate)
        except ValueError as exc:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "schema_or_license",
                    "detail": str(exc),
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
    remaining = sorted(eligible, key=_candidate_order)
    selected = []
    repository_counts: Counter[str] = Counter()
    lineage_counts: Counter[str] = Counter()
    while remaining:
        allowed = [
            row
            for row in remaining
            if repository_counts[str(row["upstream_repository"])] < REPOSITORY_CAP
            and lineage_counts[str(row["source_lineage_id"])] < LINEAGE_CAP
        ]
        if not allowed:
            break
        chosen = min(
            allowed,
            key=lambda row: (
                repository_counts[str(row["upstream_repository"])],
                _candidate_order(row),
            ),
        )
        remaining.remove(chosen)
        repository_counts[str(chosen["upstream_repository"])] += 1
        lineage_counts[str(chosen["source_lineage_id"])] += 1
        selected.append(chosen)
    if len(selected) < TARGET_ROWS:
        raise RuntimeError(
            f"SWE-rebench strict capacity is {len(selected)}, expected {TARGET_ROWS}"
        )
    primary_selected = selected[:TARGET_ROWS]
    reserve_selected = selected[TARGET_ROWS:]
    frozen = []
    license_cache: dict[tuple[str, str], dict[str, Any]] = {}
    image_cache: dict[str, dict[str, str]] = {}
    for candidate in selected:
        license_key = (
            str(candidate["repository"]),
            str(candidate["base_commit"]),
        )
        if license_key not in license_cache:
            license_cache[license_key] = _freeze_license(control_root, candidate)
        license_evidence = license_cache[license_key]
        image_name = str(candidate.pop("image_name"))
        if image_name not in image_cache:
            image_cache[image_name] = _docker_hub_digest(image_name)
        candidate["container_image"] = image_cache[image_name]
        candidate["license_spdx"] = license_evidence["spdx"]
        candidate["license_evidence_path"] = license_evidence[
            "license_evidence_path"
        ]
        candidate["license_evidence_sha256"] = license_evidence[
            "license_evidence_sha256"
        ]
        _materialize_artifacts(control_root, candidate)
        candidate["_manifest_dir"] = str(control_root.resolve())
        frozen.append(candidate)
    primary_ids = {str(row["case_id"]) for row in primary_selected}
    frozen_primary = sorted(
        (row for row in frozen if str(row["case_id"]) in primary_ids),
        key=lambda row: str(row["case_id"]),
    )
    frozen_reserve = sorted(
        (row for row in frozen if str(row["case_id"]) not in primary_ids),
        key=lambda row: str(row["case_id"]),
    )
    manifest = control_root / "swe-rebench-v2-cpp-candidate-source-manifest.jsonl"
    _write_frozen_jsonl(manifest, frozen_primary)
    reserve_manifest = (
        control_root / "swe-rebench-v2-cpp-reserve-candidate-source-manifest.jsonl"
    )
    _write_frozen_jsonl(reserve_manifest, frozen_reserve)
    write_jsonl(
        control_root / "swe-rebench-v2-cpp-freeze-rejections.jsonl", rejections
    )
    report = {
        "schema_version": "general_coding_replay_swe_rebench_v2_freeze_v1",
        "status": "exact_candidate_capacity_ready_for_bounded_preflight",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_api_sha256": sha256_file(cache / "dataset-api.json"),
        "parquet_sha256": PARQUET_SHA256,
        "dataset_license_sha256": DATASET_LICENSE_SHA256,
        "cpp_source_rows": len(cpp_rows),
        "strict_eligible_rows_before_caps": len(eligible),
        "strict_capacity_after_caps": len(frozen),
        "selected_rows": len(frozen_primary),
        "reserve_rows": len(frozen_reserve),
        "selected_by_repository": dict(
            sorted(
                Counter(
                    str(row["upstream_repository"]) for row in frozen_primary
                ).items()
            )
        ),
        "repository_cap": REPOSITORY_CAP,
        "max_repository_count": max(repository_counts.values()),
        "lineage_cap": LINEAGE_CAP,
        "max_lineage_count": max(lineage_counts.values()),
        "overlap_rejections": sum(
            item["reason"] == "overlap" for item in rejections
        ),
        "schema_or_license_rejections": sum(
            item["reason"] == "schema_or_license" for item in rejections
        ),
        "manifest": {
            "path": manifest.name,
            "rows": len(frozen_primary),
            "sha256": sha256_file(manifest),
        },
        "reserve_manifest": {
            "path": reserve_manifest.name,
            "rows": len(frozen_reserve),
            "sha256": sha256_file(reserve_manifest),
        },
        "immutable_image_digests": len(image_cache),
        "commit_bound_license_records": len(license_cache),
        "verified_rows": 0,
        "required_local_replay_passes": 2,
    }
    write_json(control_root / "swe-rebench-v2-cpp-freeze-report.json", report)
    return report


def freeze_rebench_cpp_wave2(control_root: Path) -> dict[str, Any]:
    """Freeze strict-eligible rows omitted from the original capped pool."""
    cache = control_root / "source-cache" / "swe-rebench-v2" / DATASET_REVISION
    parquet = cache / "train-00000-of-00001.parquet"
    dataset_license = cache / "DATASET.LICENSE"
    if not parquet.is_file() or sha256_file(parquet) != PARQUET_SHA256:
        raise RuntimeError("pinned SWE-rebench V2 parquet missing or drifted")
    if (
        not dataset_license.is_file()
        or sha256_file(dataset_license) != DATASET_LICENSE_SHA256
    ):
        raise RuntimeError("pinned SWE-rebench V2 dataset license missing or drifted")
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for SWE-rebench freeze") from exc
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    dimensions = _exclusion_dimensions(
        registry, _supplemental_exclusions(control_root)
    )
    table = pq.read_table(parquet)
    cpp_rows = table.filter(pc.equal(table["language"], "cpp")).to_pylist()
    frozen_ids = {
        str(row["case_id"])
        for name in (
            "swe-rebench-v2-cpp-candidate-source-manifest.jsonl",
            "swe-rebench-v2-cpp-reserve-candidate-source-manifest.jsonl",
        )
        for row in read_jsonl(control_root / name)
    }
    selected = []
    for raw in cpp_rows:
        try:
            candidate = adapt_rebench_cpp_row(
                raw, source_sha256=PARQUET_SHA256
            )
            _license_claim(candidate)
        except ValueError:
            continue
        if (
            str(candidate["case_id"]) not in frozen_ids
            and not overlap_reasons(candidate, dimensions)
        ):
            selected.append(candidate)
    selected.sort(key=lambda row: str(row["case_id"]))
    frozen = []
    license_cache: dict[tuple[str, str], dict[str, Any]] = {}
    image_cache: dict[str, dict[str, str]] = {}
    for candidate in selected:
        license_key = (
            str(candidate["repository"]),
            str(candidate["base_commit"]),
        )
        if license_key not in license_cache:
            license_cache[license_key] = _freeze_license(control_root, candidate)
        license_evidence = license_cache[license_key]
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
    manifest = control_root / "swe-rebench-v2-cpp-wave2-source-manifest.jsonl"
    _write_frozen_jsonl(manifest, frozen)
    report = {
        "schema_version": "general_coding_replay_swe_rebench_v2_wave2_v1",
        "status": "strict_unfrozen_pool_ready",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "rows": len(frozen),
        "selected_by_repository": dict(
            sorted(
                Counter(str(row["upstream_repository"]) for row in frozen).items()
            )
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
    write_json(control_root / "swe-rebench-v2-cpp-wave2-report.json", report)
    return report


def prepare_rebench_images(
    control_root: Path,
    *,
    limit: int | None = None,
    case_ids: Sequence[str] = (),
    source_manifest: Path | None = None,
    executable_manifest: Path | None = None,
) -> dict[str, Any]:
    source = source_manifest or (
        control_root / "swe-rebench-v2-cpp-candidate-source-manifest.jsonl"
    )
    rows = read_jsonl(source)
    if case_ids:
        requested_ids = set(case_ids)
        rows = [row for row in rows if str(row["case_id"]) in requested_ids]
        found_ids = {str(row["case_id"]) for row in rows}
        missing = sorted(requested_ids - found_ids)
        if missing:
            raise ValueError("unknown SWE-rebench case IDs: " + ", ".join(missing))
    if limit is not None:
        rows = rows[:limit]
    prepared = []
    rejected = []
    for row in rows:
        expected = dict(row["container_image"])
        pull = subprocess.run(
            ["docker", "pull", str(expected["repo_tag"])],
            capture_output=True,
            text=True,
            timeout=1800,
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
        row["container_image"] = actual
        row["container_workdir"] = str(actual["working_dir"])
        patch_path = control_root / str(row["target_patch_path"])
        test_command = str(row.pop("test_command"))
        row.update(
            {
                "verification_backend": "docker",
                "allowed_paths": _patch_paths(
                    patch_path.read_text(encoding="utf-8")
                ),
                "install_command": "true",
                "compile_or_typecheck_command": "true",
                "targeted_test_command": test_command,
                # The target command is already the repository's complete suite and
                # therefore executes both F2P and P2P. Avoid running that same full
                # suite twice in each fresh container.
                "full_regression_command": "true",
                "benchmark_exclusion_gate": "ready",
                "timeouts": {
                    "checkout_dependency": 120,
                    "parent": 600,
                    "target": 600,
                    "regression": 600,
                    "fresh_total": 2400,
                },
            }
        )
        prepared.append(row)
    output = executable_manifest or (
        control_root / "swe-rebench-v2-cpp-executable-manifest.jsonl"
    )
    write_jsonl(output, prepared)
    write_jsonl(
        control_root / "swe-rebench-v2-cpp-image-rejections.jsonl", rejected
    )
    report = {
        "schema_version": "general_coding_replay_swe_rebench_v2_images_v1",
        "requested": len(rows),
        "prepared": len(prepared),
        "rejected": len(rejected),
        "manifest": {
            "path": output.name,
            "rows": len(prepared),
            "sha256": sha256_file(output),
        },
    }
    write_json(control_root / "swe-rebench-v2-cpp-image-report.json", report)
    return report


def verify_rebench(
    control_root: Path,
    *,
    limit: int | None,
    case_ids: Sequence[str] = (),
    source_manifest: Path | None = None,
    executable_manifest: Path | None = None,
) -> dict[str, Any]:
    image_report = prepare_rebench_images(
        control_root,
        limit=limit,
        case_ids=case_ids,
        source_manifest=source_manifest,
        executable_manifest=executable_manifest,
    )
    if image_report["prepared"] != image_report["requested"]:
        return {
            "status": "blocked_image_gate",
            "image_report": image_report,
            "verified": 0,
        }
    output = control_root / "swe-rebench-v2-verification" / "verified"
    verification = verify_candidates(
        executable_manifest
        or control_root / "swe-rebench-v2-cpp-executable-manifest.jsonl",
        output,
        resume=True,
    )
    report = {
        "schema_version": "general_coding_replay_swe_rebench_v2_preflight_v1",
        "status": (
            "passed"
            if verification["verified"] == image_report["prepared"]
            and verification["rejected"] == 0
            else "failed"
        ),
        "scope": "bounded" if limit is not None else "full_32",
        "image_report": image_report,
        "verification": verification,
        "verified_rows": verification["verified"],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "swe-rebench-v2-cpp-preflight-report.json", report)
    return report


def build_rebench_capacity_report(control_root: Path) -> dict[str, Any]:
    """Count only double-replayed rows toward the existing 468-row capacity."""
    verified_root = control_root / "swe-rebench-v2-verification" / "verified"
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
    repository_counts: Counter[str] = Counter()
    lineage_counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: str(item["case_id"])):
        repository = str(row["upstream_repository"])
        lineage = str(row["source_lineage_id"])
        if (
            repository_counts[repository] >= REPOSITORY_CAP
            or lineage_counts[lineage] >= LINEAGE_CAP
        ):
            continue
        admitted.append(row)
        repository_counts[repository] += 1
        lineage_counts[lineage] += 1
        if len(admitted) == TARGET_ROWS:
            break
    admitted_manifest = control_root / "swe-rebench-v2-cpp-admitted.jsonl"
    write_jsonl(admitted_manifest, admitted)
    existing = json.loads(
        (control_root / "quota-override-deficit-report.json").read_text(
            encoding="utf-8"
        )
    )
    prior_capacity = int(existing["source_language_capacity_upper_bound"])
    recovered = len(admitted)
    report = {
        "schema_version": "general_coding_replay_swe_rebench_v2_capacity_v1",
        "status": "exact_32_ready_for_formal_integration"
        if recovered == TARGET_ROWS
        else "verified_capacity_deficit",
        "prior_formal_capacity": prior_capacity,
        "verified_rows_available": len(rows),
        "admitted_capacity": recovered,
        "formal_capacity_with_admitted_rows": prior_capacity + recovered,
        "remaining_cpp_deficit": TARGET_ROWS - recovered,
        "selected_by_repository": dict(sorted(repository_counts.items())),
        "repository_cap": REPOSITORY_CAP,
        "lineage_cap": LINEAGE_CAP,
        "max_lineage_count": max(lineage_counts.values(), default=0),
        "manifest": {
            "path": admitted_manifest.name,
            "rows": recovered,
            "sha256": sha256_file(admitted_manifest),
        },
        "formal_quota_files_mutated": False,
        "note": (
            "The existing formal quota report remains unchanged until all 32 rows "
            "pass two fresh offline replays; integration is atomic."
        ),
    }
    write_json(control_root / "swe-rebench-v2-cpp-capacity-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("freeze-wave2")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--limit", type=int)
    verify.add_argument("--case-id", action="append", default=[])
    verify.add_argument("--source-manifest", type=Path)
    verify.add_argument("--executable-manifest", type=Path)
    subparsers.add_parser("capacity")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = freeze_rebench_cpp_candidates(args.control_root)
    elif args.command == "freeze-wave2":
        report = freeze_rebench_cpp_wave2(args.control_root)
    elif args.command == "verify":
        report = verify_rebench(
            args.control_root,
            limit=args.limit,
            case_ids=args.case_id,
            source_manifest=args.source_manifest,
            executable_manifest=args.executable_manifest,
        )
    else:
        report = build_rebench_capacity_report(args.control_root)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
