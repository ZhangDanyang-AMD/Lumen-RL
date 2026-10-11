"""General Coding repository held-out controls.

The preferred path freezes a public, eval-only Multi-SWE-bench reservation and
publishes only hashes for solution/test patches and oracle results. Legacy
private mutation controls remain for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests


SCHEMA_VERSION = "general_coding_repository_held_out_v1"
TARGET_COUNT = 40
LANGUAGE_TARGETS = {
    "python": 8,
    "cpp": 8,
    "go": 8,
    "javascript_typescript": 8,
    "rust": 8,
}
PERMISSIVE_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
    "Unlicense",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TASK_ID = re.compile(r"gc-ho-[0-9a-f]{24}\Z")
FORBIDDEN_PUBLIC_KEYS = {
    "hidden_tests",
    "hidden_tests_path",
    "oracle",
    "oracle_path",
    "target_patch",
    "trajectory",
    "sft_events",
}
PUBLIC_BENCH_DATASET_ID = "ByteDance-Seed/Multi-SWE-bench"
PUBLIC_BENCH_REVISION = "56ff018c04a38e27ada1e9d0a6d5839a51f88f0d"
PUBLIC_BENCH_LICENSE = "CC0 with ByteDance notice; repository licenses control code"
PUBLIC_BENCH_UPSTREAM_REPOSITORY = "https://github.com/multi-swe-bench/multi-swe-bench"
PUBLIC_BENCH_UPSTREAM_REVISION = "24f493f8a103e72312ded4f6b9c89f081d69cb09"
PUBLIC_BENCH_UPSTREAM_LICENSE = "Apache-2.0"
PUBLIC_HELD_OUT_SOURCES = {
    "cpp": ("cpp/fmtlib__fmt_dataset.jsonl",),
    "go": ("go/grpc__grpc-go_dataset.jsonl",),
    "javascript_typescript": (
        "ts/darkreader__darkreader_dataset.jsonl",
        "js/axios__axios_dataset.jsonl",
        "ts/vuejs__core_dataset.jsonl",
    ),
    "rust": ("rust/tokio-rs__tokio_dataset.jsonl",),
}
PUBLIC_HELD_OUT_TARGETS = {language: 10 for language in PUBLIC_HELD_OUT_SOURCES}


class HeldOutInventoryError(ValueError):
    """Raised when private held-out evidence is incomplete or unsafe."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def task_id_for(
    repository: str,
    base_commit: str,
    source_lineage_id: str,
    mutation_id: str,
) -> str:
    """Derive an opaque, deterministic ID from immutable task identity."""
    digest = stable_hash(
        {
            "repository": repository,
            "base_commit": base_commit,
            "source_lineage_id": source_lineage_id,
            "mutation_id": mutation_id,
        }
    )
    return f"gc-ho-{digest[:24]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise HeldOutInventoryError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _normalized_patch_hash(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"
    return sha256_bytes(text.encode())


def _fetch_public_bytes(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=(30, 300),
        headers={"User-Agent": "general-coding-held-out-freezer/2"},
    )
    response.raise_for_status()
    return response.content


def _dockerhub_image_identity(org: str, repo: str, number: int) -> dict[str, str]:
    image_repository = f"mswebench/{org.lower()}_m_{repo.lower()}"
    tag = f"pr-{number}"
    token_response = requests.get(
        "https://auth.docker.io/token",
        params={
            "service": "registry.docker.io",
            "scope": f"repository:{image_repository}:pull",
        },
        timeout=(30, 120),
    )
    token_response.raise_for_status()
    token = token_response.json()["token"]
    manifest = requests.head(
        f"https://registry-1.docker.io/v2/{image_repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        },
        timeout=(30, 120),
    )
    manifest.raise_for_status()
    digest = manifest.headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise HeldOutInventoryError(f"immutable image digest missing for {image_repository}:{tag}")
    return {
        "repo_tag": f"{image_repository}:{tag}",
        "repo_digest": f"{image_repository}@{digest}",
        "manifest_digest": digest,
    }


def _github_license_evidence(
    repository: str,
    base_commit: str,
    evidence_root: Path,
) -> dict[str, Any]:
    path = evidence_root / f"{repository.replace('/', '__')}-{base_commit}.json"
    if path.is_file():
        return {**json.loads(path.read_text(encoding="utf-8")), "evidence_path": str(path)}
    response = requests.get(
        f"https://api.github.com/repos/{repository}/license",
        params={"ref": base_commit},
        timeout=(30, 120),
        headers={"User-Agent": "general-coding-held-out-freezer/2"},
    )
    import base64

    payload = response.json() if response.status_code == 200 else {}
    license_value = payload.get("license") or {}
    spdx = str(license_value.get("spdx_id", ""))
    content = str(payload.get("content", "")).replace("\n", "")
    license_bytes = base64.b64decode(content, validate=True) if content else b""
    if not license_bytes:
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst", "COPYING"):
            raw = requests.get(
                f"https://raw.githubusercontent.com/{repository}/{base_commit}/{name}",
                timeout=(30, 120),
                headers={"User-Agent": "general-coding-held-out-freezer/2"},
            )
            if raw.status_code == 200 and raw.content:
                license_bytes = raw.content
                break
    normalized = re.sub(
        r"\s+", " ", license_bytes.decode("utf-8", errors="ignore")
    ).casefold()
    if spdx not in PERMISSIVE_LICENSES:
        if "apache license" in normalized and "version 2.0" in normalized:
            spdx = "Apache-2.0"
        elif "permission is hereby granted, free of charge" in normalized:
            spdx = "MIT"
        elif "mozilla public license" in normalized and "2.0" in normalized:
            spdx = "MPL-2.0"
        elif "redistribution and use in source and binary forms" in normalized:
            spdx = "BSD-3-Clause"
    if spdx not in PERMISSIVE_LICENSES or not license_bytes:
        raise HeldOutInventoryError(f"{repository}@{base_commit}: unacceptable license {spdx!r}")
    evidence = {
        "repository": repository,
        "base_commit": base_commit,
        "spdx": spdx,
        "license_blob_sha": payload.get("sha"),
        "license_blob_sha256": sha256_bytes(license_bytes),
    }
    write_json(path, evidence)
    return {**evidence, "evidence_path": str(path)}


def freeze_public_repository_held_out(
    replay_registry_path: Path,
    replay_control_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze an authoritative public eval-only held-out without exposing oracle bytes."""
    registry = json.loads(replay_registry_path.read_text(encoding="utf-8"))
    excluded_problem_ids = {str(value) for value in registry.get("problem_ids", [])}
    excluded_repositories = {str(value).removesuffix(".git") for value in registry.get("repositories", [])}
    excluded_lineages = {str(value) for value in registry.get("source_lineages", [])}
    excluded_patches = {str(value) for value in registry.get("normalized_patch_hashes", [])}
    train_manifest = replay_control_root / "multi-swe-verified-candidate-source-manifest.jsonl"
    if train_manifest.is_file():
        for row in read_jsonl(train_manifest):
            excluded_problem_ids.add(str(row.get("problem_id", "")))
            excluded_repositories.add(str(row.get("upstream_repository", "")).removesuffix(".git"))
            excluded_lineages.add(str(row.get("source_lineage_id", "")))
            excluded_patches.add(_normalized_patch_hash(row.get("target_patch", "")))

    output_root.mkdir(parents=True, exist_ok=True)
    license_root = output_root / "license-evidence"
    reservations = []
    source_receipts = []
    rejected = Counter()
    for language, source_paths in PUBLIC_HELD_OUT_SOURCES.items():
        accepted = 0
        for source_path in source_paths:
            url = (
                f"https://huggingface.co/datasets/{PUBLIC_BENCH_DATASET_ID}/resolve/"
                f"{PUBLIC_BENCH_REVISION}/{source_path}"
            )
            source_bytes = _fetch_public_bytes(url)
            source_receipts.append(
                {
                    "path": source_path,
                    "url": url,
                    "bytes": len(source_bytes),
                    "sha256": sha256_bytes(source_bytes),
                    "retained": False,
                }
            )
            rows = [
                json.loads(line)
                for line in source_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
            for row in sorted(rows, key=lambda value: str(value.get("instance_id", ""))):
                if accepted >= PUBLIC_HELD_OUT_TARGETS[language]:
                    break
                instance_id = str(row.get("instance_id", ""))
                repository_name = f"{row.get('org')}/{row.get('repo')}"
                repository = f"https://github.com/{repository_name}"
                base = row.get("base") if isinstance(row.get("base"), Mapping) else {}
                base_commit = str(base.get("sha", ""))
                lineage = (
                    f"{PUBLIC_BENCH_DATASET_ID}@{PUBLIC_BENCH_REVISION}:{instance_id}"
                )
                patch_hash = _normalized_patch_hash(row.get("fix_patch", ""))
                if instance_id in excluded_problem_ids:
                    rejected["problem_id"] += 1
                    continue
                if repository in excluded_repositories:
                    rejected["repository"] += 1
                    continue
                if lineage in excluded_lineages:
                    rejected["lineage"] += 1
                    continue
                if patch_hash in excluded_patches:
                    rejected["patch"] += 1
                    continue
                if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
                    rejected["base_commit"] += 1
                    continue
                fix_patch = str(row.get("fix_patch", ""))
                test_patch = str(row.get("test_patch", ""))
                if not fix_patch or not test_patch:
                    rejected["oracle_or_tests_missing"] += 1
                    continue
                image = _dockerhub_image_identity(
                    str(row["org"]), str(row["repo"]), int(row["number"])
                )
                license_info = _github_license_evidence(
                    repository_name, base_commit, license_root
                )
                task_id = "gc-ho-public-" + sha256_bytes(
                    _canonical(
                        {
                            "dataset_revision": PUBLIC_BENCH_REVISION,
                            "instance_id": instance_id,
                            "base_commit": base_commit,
                        }
                    )
                )[:24]
                reservations.append(
                    {
                        "schema_version": "general_coding_repository_public_held_out_v1",
                        "task_id": task_id,
                        "split": "held_out",
                        "eval_only": True,
                        "sft_enabled": False,
                        "privacy_claim": "public_eval_only_repository_held_out",
                        "base_model_pretraining_exposure": "possible",
                        "dataset_id": PUBLIC_BENCH_DATASET_ID,
                        "dataset_revision": PUBLIC_BENCH_REVISION,
                        "problem_id": instance_id,
                        "repository": repository,
                        "base_commit": base_commit,
                        "source_lineage_id": lineage,
                        "primary_language": language,
                        "license": {
                            "spdx": license_info["spdx"],
                            "evidence_sha256": sha256_file(
                                Path(str(license_info["evidence_path"]))
                            ),
                        },
                        "problem": {
                            "title": str(row.get("title", "")),
                            "body": str(row.get("body", "")),
                            "resolved_issues_sha256": sha256_bytes(
                                _canonical(row.get("resolved_issues"))
                            ),
                        },
                        "protected": {
                            "solution_patch_sha256": sha256_bytes(fix_patch.encode()),
                            "normalized_solution_patch_sha256": patch_hash,
                            "test_patch_sha256": sha256_bytes(test_patch.encode()),
                            "oracle_results_sha256": sha256_bytes(
                                _canonical(
                                    {
                                        "run_result": row.get("run_result"),
                                        "test_patch_result": row.get("test_patch_result"),
                                        "fix_patch_result": row.get("fix_patch_result"),
                                        "fixed_tests": row.get("fixed_tests"),
                                    }
                                )
                            ),
                        },
                        "container_image": image,
                    }
                )
                accepted += 1
            if accepted >= PUBLIC_HELD_OUT_TARGETS[language]:
                break
        if accepted != PUBLIC_HELD_OUT_TARGETS[language]:
            raise HeldOutInventoryError(
                f"{language}: needed {PUBLIC_HELD_OUT_TARGETS[language]} disjoint rows, found {accepted}"
            )

    reservations.sort(key=lambda row: row["task_id"])
    reservations_path = output_root / "reservations.jsonl"
    write_jsonl(reservations_path, reservations)
    public_text = reservations_path.read_text(encoding="utf-8")
    if '"fix_patch"' in public_text or '"test_patch"' in public_text:
        raise HeldOutInventoryError("public held-out output contains oracle patch bytes")
    counts = dict(sorted(Counter(row["primary_language"] for row in reservations).items()))
    component = {
        "schema_version": "general_coding_repository_public_held_out_exclusion_v1",
        "domain": "general_coding_repository_held_out",
        "task_count": len(reservations),
        "complete": len(reservations) == TARGET_COUNT,
        "privacy_claim": "public eval-only repository held-out; possible base-model pretraining exposure",
        "dataset": {
            "id": PUBLIC_BENCH_DATASET_ID,
            "revision": PUBLIC_BENCH_REVISION,
            "license": PUBLIC_BENCH_LICENSE,
            "source_files": source_receipts,
        },
        "upstream_repository": {
            "url": PUBLIC_BENCH_UPSTREAM_REPOSITORY,
            "revision": PUBLIC_BENCH_UPSTREAM_REVISION,
            "license": PUBLIC_BENCH_UPSTREAM_LICENSE,
        },
        "entries": [
            {
                "task_id": row["task_id"],
                "problem_id": row["problem_id"],
                "repository": row["repository"],
                "base_commit": row["base_commit"],
                "source_lineage_id": row["source_lineage_id"],
                "primary_language": row["primary_language"],
                **row["protected"],
                "container_image_repo_digest": row["container_image"]["repo_digest"],
            }
            for row in reservations
        ],
        "source": {
            "path": "reservations.jsonl",
            "rows": len(reservations),
            "sha256": sha256_file(reservations_path),
        },
    }
    component["component_sha256"] = stable_hash(component)
    write_json(output_root / "exclusion-component.json", component)
    report = {
        "schema_version": "general_coding_repository_public_held_out_freeze_report_v1",
        "status": "ready" if component["complete"] else "deficit",
        "target": TARGET_COUNT,
        "held_out": len(reservations),
        "language_distribution": counts,
        "language_targets": PUBLIC_HELD_OUT_TARGETS,
        "python_balance_note": (
            "The authoritative multilingual benchmark's original seven-language release "
            "does not include Python; four in-scope quota languages are balanced at 10 each."
        ),
        "disjointness": {
            "problem_id": True,
            "repository": True,
            "source_lineage": True,
            "normalized_patch": True,
            "rejected_counts": dict(sorted(rejected.items())),
        },
        "protected_material": "hash_only_in_public_outputs",
        "container_images": {
            "digest_frozen": len(reservations),
            "downloaded": 0,
        },
        "source_files": source_receipts,
        "reservations_sha256": sha256_file(reservations_path),
        "component_sha256": component["component_sha256"],
    }
    write_json(output_root / "inventory-report.json", report)
    write_checksums(output_root)
    return report


def update_replay_registry_from_public_held_out(
    registry_path: Path,
    component_path: Path,
) -> dict[str, Any]:
    """Replace only the unavailable repository-held-out blocker."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    component = json.loads(component_path.read_text(encoding="utf-8"))
    entries = component.get("entries", [])
    if component.get("complete") is not True or len(entries) != TARGET_COUNT:
        raise HeldOutInventoryError("refusing registry update: public held-out is incomplete")
    registry["general_coding_repository_held_out"] = {
        "classification": "public_eval_only_repository_held_out",
        "base_model_pretraining_exposure": "possible",
        "task_count": TARGET_COUNT,
        "component_sha256": component["component_sha256"],
        "source_sha256": component["source"]["sha256"],
        "task_ids": sorted(row["task_id"] for row in entries),
        "problem_ids": sorted(row["problem_id"] for row in entries),
        "source_lineages": sorted(row["source_lineage_id"] for row in entries),
        "normalized_patch_hashes": sorted(
            row["normalized_solution_patch_sha256"] for row in entries
        ),
        "oracle_hashes": sorted(row["solution_patch_sha256"] for row in entries),
        "test_patch_hashes": sorted(row["test_patch_sha256"] for row in entries),
        "container_image_digests": sorted(
            row["container_image_repo_digest"] for row in entries
        ),
    }
    registry["problem_ids"] = sorted(
        set(registry.get("problem_ids", []))
        | {row["problem_id"] for row in entries}
    )
    registry["repositories"] = sorted(
        set(registry.get("repositories", []))
        | {row["repository"] for row in entries}
    )
    registry["source_lineages"] = sorted(
        set(registry.get("source_lineages", []))
        | {row["source_lineage_id"] for row in entries}
    )
    registry["normalized_patch_hashes"] = sorted(
        set(registry.get("normalized_patch_hashes", []))
        | {row["normalized_solution_patch_sha256"] for row in entries}
    )
    registry["blockers"] = [
        blocker
        for blocker in registry.get("blockers", [])
        if blocker.get("id") != "general_coding_repository_held_out"
    ]
    registry["pending"] = [item["id"] for item in registry["blockers"]]
    registry["status"] = "frozen" if not registry["blockers"] else "frozen_with_pending_domains"
    write_json(registry_path, registry)
    return registry


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    private_root = root.resolve(strict=True)
    if resolved != private_root and private_root not in resolved.parents:
        raise HeldOutInventoryError(f"protected path escapes private root: {path}")
    if not resolved.is_file():
        raise HeldOutInventoryError(f"protected path is not a file: {path}")
    return resolved


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("baseline", {}).get("passed") is not True:
        raise HeldOutInventoryError("baseline must pass before mutation")
    if receipt.get("defect", {}).get("passed") is not False:
        raise HeldOutInventoryError("defect must fail protected tests")
    if receipt.get("oracle", {}).get("passed") is not True:
        raise HeldOutInventoryError("oracle must pass protected tests")
    if receipt.get("network") != "disabled":
        raise HeldOutInventoryError("validation network must be disabled")
    if receipt.get("cpu_only") is not True:
        raise HeldOutInventoryError("validation must be CPU-only")
    if receipt.get("oracle_access") != {
        "authorized": True,
        "scope": "eval_only",
    }:
        raise HeldOutInventoryError("oracle access receipt must be eval-only")
    return receipt


def validate_private_candidate(
    candidate: Mapping[str, Any],
    private_root: Path,
) -> dict[str, dict[str, Any]]:
    """Validate one fully frozen task and split public from private metadata."""
    required = {
        "task_id",
        "repository",
        "base_commit",
        "source_lineage_id",
        "primary_language",
        "mutation_id",
        "license",
        "problem_contract",
        "parent_source_refs",
        "protected",
        "evaluation",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise HeldOutInventoryError(f"missing fields: {missing}")
    task_id = str(candidate["task_id"])
    if not TASK_ID.fullmatch(task_id):
        raise HeldOutInventoryError(f"non-opaque task_id: {task_id!r}")
    commit = str(candidate["base_commit"])
    if not COMMIT.fullmatch(commit):
        raise HeldOutInventoryError("base_commit must be an exact 40-char lowercase commit")
    repository = str(candidate["repository"])
    if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+(?:\.git)?", repository):
        raise HeldOutInventoryError("repository must be an immutable GitHub source URL")
    language = str(candidate["primary_language"])
    if language not in LANGUAGE_TARGETS:
        raise HeldOutInventoryError(f"unsupported language: {language}")
    expected_task_id = task_id_for(
        repository,
        commit,
        str(candidate["source_lineage_id"]),
        str(candidate["mutation_id"]),
    )
    if task_id != expected_task_id:
        raise HeldOutInventoryError("task_id does not match deterministic immutable identity")

    license_info = candidate["license"]
    if not isinstance(license_info, Mapping):
        raise HeldOutInventoryError("license must be an object")
    spdx = str(license_info.get("spdx", ""))
    if spdx not in PERMISSIVE_LICENSES:
        raise HeldOutInventoryError(f"non-permissive or unknown license: {spdx!r}")
    license_path = _inside(Path(str(license_info.get("evidence_path", ""))), private_root)
    evidence = json.loads(license_path.read_text(encoding="utf-8"))
    expected_evidence = {
        "repository": repository,
        "base_commit": commit,
        "spdx": spdx,
    }
    if any(evidence.get(key) != value for key, value in expected_evidence.items()):
        raise HeldOutInventoryError("license evidence is not frozen to this repository revision")
    if not SHA256.fullmatch(str(evidence.get("license_blob_sha256", ""))):
        raise HeldOutInventoryError("license evidence lacks a frozen license blob checksum")

    refs = candidate["parent_source_refs"]
    if not isinstance(refs, list) or not refs:
        raise HeldOutInventoryError("parent_source_refs must be non-empty")
    for ref in refs:
        if (
            not isinstance(ref, Mapping)
            or not str(ref.get("path", "")).strip()
            or not SHA256.fullmatch(str(ref.get("sha256", "")))
        ):
            raise HeldOutInventoryError("invalid parent source reference")

    evaluation = candidate["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise HeldOutInventoryError("evaluation must be an object")
    command = evaluation.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise HeldOutInventoryError("evaluation command must be a non-empty argv list")
    if evaluation.get("network") != "disabled" or evaluation.get("cpu_only") is not True:
        raise HeldOutInventoryError("evaluation must be network-disabled and CPU-only")
    timeout = evaluation.get("timeout_seconds")
    if not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise HeldOutInventoryError("timeout_seconds must be in [1, 3600]")

    protected = candidate["protected"]
    if not isinstance(protected, Mapping):
        raise HeldOutInventoryError("protected must be an object")
    hidden = _inside(Path(str(protected.get("hidden_tests_path", ""))), private_root)
    oracle = _inside(Path(str(protected.get("oracle_path", ""))), private_root)
    receipt_path = _inside(
        Path(str(protected.get("validation_receipt_path", ""))), private_root
    )
    _validate_receipt(receipt_path)

    contract = candidate["problem_contract"]
    if not isinstance(contract, Mapping) or not contract.get("description"):
        raise HeldOutInventoryError("model-visible problem contract is incomplete")
    if FORBIDDEN_PUBLIC_KEYS.intersection(_walk_keys(contract)):
        raise HeldOutInventoryError("problem contract contains protected answer/test material")

    hashes = {
        "hidden_tests_sha256": sha256_file(hidden),
        "oracle_sha256": sha256_file(oracle),
        "validation_receipt_sha256": sha256_file(receipt_path),
        "oracle_access_receipt_sha256": sha256_file(receipt_path),
    }
    reservation = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "split": "held_out",
        "eval_only": True,
        "sft_enabled": False,
        "repository": repository,
        "base_commit": commit,
        "source_lineage_id": str(candidate["source_lineage_id"]),
        "mutation_id": str(candidate["mutation_id"]),
        "primary_language": language,
        "license": {
            "spdx": spdx,
            "evidence_sha256": sha256_file(license_path),
        },
        "problem_contract": dict(contract),
        "parent_source_refs": [dict(ref) for ref in refs],
        "protected": hashes,
        "evaluation": dict(evaluation),
    }
    leaked = sorted(FORBIDDEN_PUBLIC_KEYS.intersection(_walk_keys(reservation)))
    if leaked:
        raise HeldOutInventoryError(f"public reservation leaks protected fields: {leaked}")
    private = {
        "task_id": task_id,
        "hidden_tests_path": str(hidden),
        "oracle_path": str(oracle),
        "validation_receipt_path": str(receipt_path),
        "license_evidence_path": str(license_path),
        **hashes,
    }
    return {"reservation": reservation, "private": private}


def _language_deficits(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        language: max(0, target - counts.get(language, 0))
        for language, target in LANGUAGE_TARGETS.items()
    }


def build_inventory(
    candidates_path: Path,
    private_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Deterministically freeze only candidates with complete validation evidence."""
    candidates = read_jsonl(candidates_path)
    accepted: list[dict[str, dict[str, Any]]] = []
    rejected = []
    seen_ids: set[str] = set()
    seen_lineages: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda row: (
            str(row.get("task_id", "")),
            str(row.get("source_lineage_id", "")),
        ),
    ):
        try:
            frozen = validate_private_candidate(candidate, private_root)
            reservation = frozen["reservation"]
            if reservation["task_id"] in seen_ids:
                raise HeldOutInventoryError("duplicate task_id")
            if reservation["source_lineage_id"] in seen_lineages:
                raise HeldOutInventoryError("duplicate source lineage")
            seen_ids.add(reservation["task_id"])
            seen_lineages.add(reservation["source_lineage_id"])
            accepted.append(frozen)
        except (HeldOutInventoryError, OSError, json.JSONDecodeError) as exc:
            rejected.append(
                {
                    "task_id": candidate.get("task_id"),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    reservations = [item["reservation"] for item in accepted]
    private_index = [item["private"] for item in accepted]
    output_root.mkdir(parents=True, exist_ok=True)
    reservations_path = output_root / "reservations.jsonl"
    private_path = output_root / "private-index.jsonl"
    write_jsonl(reservations_path, reservations)
    write_jsonl(private_path, private_index)
    write_jsonl(output_root / "rejected.jsonl", rejected)
    counts = dict(sorted(Counter(row["primary_language"] for row in reservations).items()))
    deficits = _language_deficits(counts)
    ready = (
        len(reservations) == TARGET_COUNT
        and not any(deficits.values())
        and counts == LANGUAGE_TARGETS
    )
    report = {
        "schema_version": "general_coding_repository_held_out_inventory_report_v1",
        "status": "ready" if ready else "deficit",
        "target": TARGET_COUNT,
        "materialized": len(reservations),
        "reserved": len(reservations),
        "rejected": len(rejected),
        "deficit": max(0, TARGET_COUNT - len(reservations)),
        "language_targets": LANGUAGE_TARGETS,
        "language_distribution": counts,
        "language_deficits": deficits,
        "source_deficits": {
            "fully_frozen_permissive_oss_revision_with_private_validation": max(
                0, TARGET_COUNT - len(reservations)
            )
        },
        "source": {
            "sha256": sha256_file(candidates_path),
            "rows": len(candidates),
        },
        "reservations_sha256": sha256_file(reservations_path),
        "private_index_sha256": sha256_file(private_path),
        "privacy_invariants": {
            "eval_only": all(row["eval_only"] for row in reservations),
            "sft_disabled": all(row["sft_enabled"] is False for row in reservations),
            "hash_only_protected_refs": all(
                not FORBIDDEN_PUBLIC_KEYS.intersection(_walk_keys(row))
                for row in reservations
            ),
            "network_disabled": all(
                row["evaluation"]["network"] == "disabled" for row in reservations
            ),
            "cpu_only": all(row["evaluation"]["cpu_only"] for row in reservations),
        },
    }
    write_json(output_root / "inventory-report.json", report)
    write_checksums(output_root)
    return report


def export_exclusion_component(
    reservations_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export identity/hash reservations without contracts or secret paths."""
    rows = read_jsonl(reservations_path)
    entries = []
    for row in rows:
        protected = row.get("protected", {})
        required_hashes = (
            "hidden_tests_sha256",
            "oracle_sha256",
            "oracle_access_receipt_sha256",
        )
        if not all(SHA256.fullmatch(str(protected.get(key, ""))) for key in required_hashes):
            raise HeldOutInventoryError(f"{row.get('task_id')}: protected hashes incomplete")
        entries.append(
            {
                "task_id": row["task_id"],
                "repository": row["repository"],
                "base_commit": row["base_commit"],
                "source_lineage_id": row["source_lineage_id"],
                "primary_language": row["primary_language"],
                "hidden_tests_sha256": protected["hidden_tests_sha256"],
                "oracle_sha256": protected["oracle_sha256"],
                "oracle_access_receipt_sha256": protected[
                    "oracle_access_receipt_sha256"
                ],
            }
        )
    component = {
        "schema_version": "general_coding_repository_held_out_exclusion_v1",
        "domain": "general_coding_repository_held_out",
        "task_count": len(entries),
        "complete": len(entries) == TARGET_COUNT,
        "entries": sorted(entries, key=lambda item: item["task_id"]),
        "source": {
            "sha256": sha256_file(reservations_path),
            "rows": len(rows),
        },
    }
    component["component_sha256"] = stable_hash(component)
    write_json(output_root / "exclusion-component.json", component)
    write_checksums(output_root)
    return component


def audit_replay_exclusion(
    reservations_path: Path,
    replay_artifacts: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Detect held-out identity/hash leakage into Replay/Train visible artifacts."""
    rows = read_jsonl(reservations_path)
    needles = set()
    for row in rows:
        needles.update(
            str(value)
            for value in (
                row.get("task_id"),
                row.get("source_lineage_id"),
                row.get("repository"),
            )
            if value
        )
        needles.update(
            str(value)
            for key, value in (row.get("protected") or {}).items()
            if key.endswith("_sha256") and value
        )
    hits = []
    checked = []
    for artifact in replay_artifacts:
        paths = [artifact] if artifact.is_file() else sorted(artifact.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            data = path.read_bytes()
            checked.append({"path": str(path), "sha256": sha256_bytes(data)})
            text = data.decode("utf-8", errors="ignore")
            for needle in sorted(needles):
                if needle and needle in text:
                    hits.append(
                        {
                            "path": str(path),
                            "kind": "held_out_identity_or_hash",
                            "value_sha256": sha256_bytes(needle.encode()),
                        }
                    )
    report = {
        "schema_version": "general_coding_repository_held_out_leakage_audit_v1",
        "status": "pass" if not hits else "fail",
        "held_out_tasks": len(rows),
        "checked_files": len(checked),
        "checked": checked,
        "hit_count": len(hits),
        "hits": hits,
    }
    write_json(output_path, report)
    return report


def update_replay_registry(
    registry_path: Path,
    component_path: Path,
) -> dict[str, Any]:
    """Close the Replay blocker only for a complete immutable 40-task component."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    component = json.loads(component_path.read_text(encoding="utf-8"))
    entries = component.get("entries", [])
    counts = Counter(item.get("primary_language") for item in entries)
    if (
        component.get("complete") is not True
        or len(entries) != TARGET_COUNT
        or dict(counts) != LANGUAGE_TARGETS
    ):
        raise HeldOutInventoryError("refusing registry update: held-out component is incomplete")
    registry["general_coding_repository_held_out"] = {
        "task_count": TARGET_COUNT,
        "component_sha256": component["component_sha256"],
        "source_sha256": component["source"]["sha256"],
        "task_ids": sorted(item["task_id"] for item in entries),
        "source_lineages": sorted(item["source_lineage_id"] for item in entries),
        "hidden_test_hashes": sorted(item["hidden_tests_sha256"] for item in entries),
        "oracle_access_receipt_hashes": sorted(
            item["oracle_access_receipt_sha256"] for item in entries
        ),
    }
    registry["blockers"] = [
        item
        for item in registry.get("blockers", [])
        if item.get("id") != "general_coding_repository_held_out"
    ]
    registry["pending"] = [item["id"] for item in registry["blockers"]]
    registry["status"] = "frozen" if not registry["blockers"] else "frozen_with_pending_domains"
    write_json(registry_path, registry)
    return registry


def package_private_domain(
    control_root: Path,
    private_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create an eval-operator package; never creates a model-visible package."""
    reservations = read_jsonl(control_root / "reservations.jsonl")
    private_rows = read_jsonl(control_root / "private-index.jsonl")
    by_id = {row["task_id"]: row for row in private_rows}
    if {row["task_id"] for row in reservations} != set(by_id):
        raise HeldOutInventoryError("private/public task identity mismatch")
    if output_root.exists():
        raise HeldOutInventoryError(f"private package already exists: {output_root}")
    protected_root = output_root / "protected"
    for row in reservations:
        task_id = row["task_id"]
        source = by_id[task_id]
        destination = protected_root / task_id
        destination.mkdir(parents=True)
        for key, name, hash_key in (
            ("hidden_tests_path", "hidden-tests.bundle", "hidden_tests_sha256"),
            ("oracle_path", "oracle.patch", "oracle_sha256"),
            (
                "validation_receipt_path",
                "validation-receipt.json",
                "validation_receipt_sha256",
            ),
            ("license_evidence_path", "LICENSE.evidence", None),
        ):
            path = _inside(Path(source[key]), private_root)
            shutil.copyfile(path, destination / name)
            if hash_key and sha256_file(destination / name) != source[hash_key]:
                raise HeldOutInventoryError(f"{task_id}: protected copy checksum mismatch")
    shutil.copyfile(control_root / "reservations.jsonl", output_root / "reservations.jsonl")
    manifest = {
        "schema_version": "general_coding_repository_held_out_private_package_v1",
        "eval_only": True,
        "model_visible": False,
        "contains_sft_trajectories": False,
        "task_count": len(reservations),
        "reservations_sha256": sha256_file(output_root / "reservations.jsonl"),
    }
    write_json(output_root / "manifest.json", manifest)
    write_checksums(output_root)
    return manifest


def run_eval_task(
    reservation: Mapping[str, Any],
    private_entry: Mapping[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    """Run one protected test command with CPU/GPU and network isolation."""
    if reservation.get("eval_only") is not True or reservation.get("sft_enabled") is not False:
        raise HeldOutInventoryError("task is not eval-only")
    hidden = Path(str(private_entry["hidden_tests_path"])).resolve(strict=True)
    if sha256_file(hidden) != reservation["protected"]["hidden_tests_sha256"]:
        raise HeldOutInventoryError("hidden test checksum mismatch")
    evaluation = reservation["evaluation"]
    command: Sequence[str] = list(evaluation["command"])
    if shutil.which("unshare"):
        command = ["unshare", "--net", "--", *command]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "GENERAL_CODING_HIDDEN_TESTS": str(hidden),
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    result = subprocess.run(
        command,
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=int(evaluation["timeout_seconds"]),
        check=False,
    )
    return {
        "task_id": reservation["task_id"],
        "returncode": result.returncode,
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "network": "disabled_linux_namespace" if command[0] == "unshare" else "unavailable",
        "cpu_only": True,
    }


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != checksum_path:
                handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
