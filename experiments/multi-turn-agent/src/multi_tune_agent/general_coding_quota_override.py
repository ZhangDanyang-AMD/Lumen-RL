"""Auditable General Coding Replay source-quota override.

This module never promotes source metadata to ``verified``.  It freezes
candidate evidence and emits commands for the existing fresh, offline,
double-replay verifier.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
import yaml

from .general_coding_replay import (
    ALLOWED_TASK_TYPES,
    MULTISWE_HELD_OUT_REPOSITORIES,
    PHASE1_LANGUAGE_TARGETS,
    PHASE1_TASK_TYPE_TARGETS,
    PERMISSIVE_SOURCE_LICENSES,
    atomic_write,
    canonical_json,
    derive_repository_task_type,
    read_jsonl,
    select_replay,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)


OVERRIDE_SCHEMA = "general_coding_replay_quota_override_v1"
SWE_SMITH_SOURCES = {
    "cpp": {
        "dataset_id": "SWE-bench/SWE-smith-cpp",
        "revision": "b1554a5f12603c825f9b48b361b9b97032347e48",
        "rows": 5123,
        "parquet_sha256": "006e532392f1033630e89c615cd75071255cbc7b6f969bc4da4c7feaca1ce03e",
    },
    "rust": {
        "dataset_id": "SWE-bench/SWE-smith-rs",
        "revision": "1ae20d63edd611309d86133466247e3a9723a27d",
        "rows": 5311,
        "parquet_sha256": "33ee3cb7956cccc91e8af697c9894b4ca6e00cc16e502ffb62ef6dca9b154c00",
    },
}
BYTEDANCE_DATASET_ID = "ByteDance-Seed/Multi-SWE-RL"
BYTEDANCE_REVISION = "9777648932daa214ba18c70c81e85821b5836f32"
BYTEDANCE_SOURCE_TARGETS = {"cpp": 75, "rust": 50}
EFFECTIVE_SOURCE_TARGETS = {
    "swe_gym": 250,
    "multi_swe_rl_verified": 125,
    "bytedance_multi_swe_rl_fallback": 93,
    "swe_rebench_v2_fallback": 10,
    "swe_bench_live_multilang_cpp": 12,
    "menvdata_swe_cpp": 10,
}
SOURCE_LANGUAGE_TARGETS = {
    "swe_gym": {"python": 250},
    "multi_swe_rl_verified": {"go": 50, "javascript_typescript": 75},
    "bytedance_multi_swe_rl_fallback": {"cpp": 43, "rust": 50},
    "swe_rebench_v2_fallback": {"cpp": 10},
    "swe_bench_live_multilang_cpp": {"cpp": 12},
    "menvdata_swe_cpp": {"cpp": 10},
}
BYTEDANCE_FILES = {
    "cpp": (
        "data_20240601_20250331/cpp/CGAL__cgal_dataset.jsonl",
        "data_20240601_20250331/cpp/bitcoin__bitcoin_dataset.jsonl",
        "data_20240601_20250331/cpp/catchorg__Catch2_dataset.jsonl",
        "data_20240601_20250331/cpp/fmtlib__fmt_dataset.jsonl",
        "data_20240601_20250331/cpp/halide__Halide_dataset.jsonl",
        "data_20240601_20250331/cpp/nlohmann__json_dataset.jsonl",
        "data_20240601_20250331/cpp/root-project__root_dataset.jsonl",
        "data_20240601_20250331/cpp/simdjson__simdjson_dataset.jsonl",
        "data_20240601_20250331/cpp/yhirose__cpp-httplib_dataset.jsonl",
    ),
    "rust": (
        "data_20240601_20250331/rust/BurntSushi__ripgrep_dataset.jsonl",
        "data_20240601_20250331/rust/alacritty__alacritty_dataset.jsonl",
        "data_20240601_20250331/rust/clap-rs__clap_dataset.jsonl",
        "data_20240601_20250331/rust/fish-shell__fish-shell_dataset.jsonl",
        "data_20240601_20250331/rust/helix-editor__helix_dataset.jsonl",
        "data_20240601_20250331/rust/nushell__nushell_dataset.jsonl",
        "data_20240601_20250331/rust/rusqlite__rusqlite_dataset.jsonl",
        "data_20240601_20250331/rust/rust-lang__mdBook_dataset.jsonl",
        "data_20240601_20250331/rust/serde-rs__serde_dataset.jsonl",
        "data_20240601_20250331/rust/sharkdp__bat_dataset.jsonl",
        "data_20240601_20250331/rust/sharkdp__fd_dataset.jsonl",
        "data_20240601_20250331/rust/tokio-rs__bytes_dataset.jsonl",
        "data_20240601_20250331/rust/tokio-rs__tokio_dataset.jsonl",
        "data_20240601_20250331/rust/tokio-rs__tracing_dataset.jsonl",
    ),
}
BYTEDANCE_IMMUTABLE_LICENSE_FILES = {
    "BurntSushi/ripgrep": ("MIT", "LICENSE-MIT"),
    "alacritty/alacritty": ("Apache-2.0", "LICENSE-APACHE"),
    "bitcoin/bitcoin": ("MIT", "COPYING"),
    "catchorg/Catch2": ("BSL-1.0", "LICENSE.txt"),
    "clap-rs/clap": ("MIT", "LICENSE-MIT"),
    "fmtlib/fmt": ("MIT", "LICENSE"),
    "halide/Halide": ("MIT", "LICENSE.txt"),
    "nushell/nushell": ("MIT", "LICENSE"),
    "serde-rs/serde": ("MIT", "LICENSE-MIT"),
    "sharkdp/bat": ("MIT", "LICENSE-MIT"),
    "sharkdp/fd": ("MIT", "LICENSE-MIT"),
    "tokio-rs/bytes": ("MIT", "LICENSE"),
    "tokio-rs/tokio": ("MIT", "LICENSE"),
    "tokio-rs/tracing": ("MIT", "LICENSE"),
    "yhirose/cpp-httplib": ("MIT", "LICENSE"),
}
BYTEDANCE_INCOMPATIBLE_LICENSES = {
    "CGAL/cgal": "GPL-3.0-or-later/LGPL-3.0-or-later",
    "fish-shell/fish-shell": "GPL-2.0-only",
    "root-project/root": "LGPL-2.1-or-later",
    "rust-lang/mdBook": "MPL-2.0",
}
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
IMMUTABLE_IMAGE = re.compile(r".+@sha256:[0-9a-f]{64}\Z")
SWE_REBENCH_V2_ALTERNATIVE = {
    "dataset_id": "nebius/SWE-rebench-V2",
    "revision": "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0",
    "language": "C++",
    "status": "requires_independent_freeze_and_overlap_audit",
    "reason": (
        "public executable C++ rows expose base_commit, gold patch, test patch, "
        "FAIL_TO_PASS, PASS_TO_PASS, image_name, repository license, and install config"
    ),
}


def _get_json(url: str, destination: Path) -> dict[str, Any]:
    if destination.is_file():
        value = json.loads(destination.read_text(encoding="utf-8"))
    else:
        response = requests.get(url, timeout=(30, 300))
        response.raise_for_status()
        value = response.json()
        write_json(destination, value)
    if not isinstance(value, dict):
        raise RuntimeError(f"{url}: expected an object")
    return value


def _download(url: str, destination: Path) -> dict[str, Any]:
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=(30, 600))
        response.raise_for_status()
        atomic_write(destination, response.content)
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def validate_override_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the override preserves every requested total."""
    if document.get("schema_version") != OVERRIDE_SCHEMA:
        raise ValueError("unsupported quota override schema")
    if document.get("status") != "approved_source_infeasibility_override":
        raise ValueError("quota override is not formally approved")
    effective = document.get("effective_quotas")
    if not isinstance(effective, Mapping):
        raise ValueError("effective_quotas must be a mapping")
    expected = {
        "source_targets": EFFECTIVE_SOURCE_TARGETS,
        "language_targets": PHASE1_LANGUAGE_TARGETS,
        "task_type_targets": PHASE1_TASK_TYPE_TARGETS,
        "source_language_targets": SOURCE_LANGUAGE_TARGETS,
    }
    for key, value in expected.items():
        if effective.get(key) != value:
            raise ValueError(f"quota override changed {key}")
    for key in ("source_targets", "language_targets", "task_type_targets"):
        if sum(effective[key].values()) != 500:
            raise ValueError(f"{key} does not sum to 500")
    matrix_total = sum(
        amount
        for languages in effective["source_language_targets"].values()
        for amount in languages.values()
    )
    if matrix_total != 500:
        raise ValueError("source-language cross quotas do not sum to 500")
    return dict(effective)


def _normalized_repository(value: str) -> str:
    value = value.strip().lower().removesuffix(".git").rstrip("/")
    return value.replace("git@github.com:", "https://github.com/")


def _exclusion_dimensions(
    registry: Mapping[str, Any],
    supplemental_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, set[str]]:
    dimensions = {
        "repository": {
            _normalized_repository(str(value)) for value in registry.get("repositories", [])
        },
        "problem": {str(value) for value in registry.get("problem_ids", [])},
        "lineage": {str(value) for value in registry.get("source_lineages", [])},
        "patch": {str(value) for value in registry.get("normalized_patch_hashes", [])},
        "test": {str(value) for value in registry.get("test_set_hashes", [])},
    }
    for row in supplemental_rows:
        dimensions["repository"].add(
            _normalized_repository(str(row.get("upstream_repository", "")))
        )
        dimensions["problem"].add(str(row.get("problem_id", "")))
        dimensions["lineage"].add(str(row.get("source_lineage_id", "")))
        dimensions["patch"].add(str(row.get("normalized_patch_hash", "")))
        dimensions["test"].add(str(row.get("test_set_hash", "")))
    for values in dimensions.values():
        values.discard("")
    return dimensions


def overlap_reasons(
    row: Mapping[str, Any], dimensions: Mapping[str, set[str]]
) -> list[str]:
    checks = {
        "repository": _normalized_repository(str(row.get("upstream_repository", ""))),
        "problem": str(row.get("problem_id", "")),
        "lineage": str(row.get("source_lineage_id", "")),
        "patch": str(row.get("normalized_patch_hash", "")),
        "test": str(row.get("test_set_hash", "")),
    }
    return sorted(key for key, value in checks.items() if value and value in dimensions[key])


def swe_smith_admission_decision(
    row: Mapping[str, Any], probe: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed when the immutable image is not the executable buggy parent."""
    image_name = str(row.get("image_name", ""))
    mutation_patch = str(row.get("patch", ""))
    head = str(probe.get("image_head", ""))
    repo_digest = str(probe.get("repo_digest", ""))
    missing = []
    if not image_name:
        missing.append("image_name")
    if not mutation_patch.strip():
        missing.append("bug_introducing_patch")
    if not FULL_COMMIT.fullmatch(head):
        missing.append("image_head")
    if not IMMUTABLE_IMAGE.fullmatch(repo_digest):
        missing.append("repo_digest")
    for key in (
        "patch_applies_to_image_head",
        "bug_present_only_after_patch",
        "inverse_patch_restores_tests",
        "license_evidence_sha256",
    ):
        if not probe.get(key):
            missing.append(key)
    if missing:
        return {
            "status": "rejected_incomplete_preflight",
            "missing_or_invalid": sorted(missing),
            "candidate_rows_admitted": 0,
        }
    return {
        "status": "rejected_no_exact_buggy_parent_commit",
        "reason": (
            "SWE-smith patch is the bug-introducing mutation. The immutable image HEAD "
            "is the pristine state; the executable task parent is HEAD plus an "
            "uncommitted mutation, so /testbed HEAD cannot be represented as the "
            "task base_commit without reversing patch semantics."
        ),
        "image_head": head,
        "repo_digest": repo_digest,
        "pristine_head_recovered": True,
        "executable_buggy_parent_is_git_commit": False,
        "candidate_rows_admitted": 0,
    }


def _docker_hub_digest(image_name: str) -> dict[str, str]:
    image_name = image_name.removeprefix("docker.io/")
    repository, separator, tag = image_name.partition(":")
    if not separator:
        tag = "latest"
    token_response = requests.get(
        "https://auth.docker.io/token",
        params={
            "service": "registry.docker.io",
            "scope": f"repository:{repository}:pull",
        },
        timeout=(30, 120),
    )
    token_response.raise_for_status()
    token = str(token_response.json()["token"])
    response = requests.head(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        },
        timeout=(30, 120),
    )
    response.raise_for_status()
    digest = response.headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(f"{image_name}: registry returned no immutable digest")
    return {
        "repo_tag": f"{repository}:{tag}",
        "repo_digest": f"{repository}@{digest}",
        "manifest_digest": digest,
    }


def _docker_run_probe(
    repo_digest: str,
    command: str,
    *,
    patch_path: Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--env",
        "HIP_VISIBLE_DEVICES=",
        "--env",
        "ROCR_VISIBLE_DEVICES=",
    ]
    if patch_path is not None:
        args.extend(["--volume", f"{patch_path}:/evidence/mutation.patch:ro"])
    args.extend(["--entrypoint", "/bin/bash", repo_digest, "-lc", command])
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def preflight_swe_smith_cpp(
    control_root: Path,
    *,
    limit: int = 1,
) -> dict[str, Any]:
    """Probe a bounded number of pinned C++ images without admitting any row."""
    if limit < 1:
        raise ValueError("SWE-smith preflight limit must be positive")
    spec = SWE_SMITH_SOURCES["cpp"]
    cache = control_root / "source-cache" / "swe-smith" / spec["revision"]
    parquet = cache / "train-00000-of-00001.parquet"
    receipt = _download(
        f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/"
        f"{spec['revision']}/data/train-00000-of-00001.parquet",
        parquet,
    )
    if receipt["sha256"] != spec["parquet_sha256"]:
        raise RuntimeError("SWE-smith C++ parquet checksum drift")
    report_path = control_root / "swe-smith-cpp-preflight-report.json"
    existing_deficit_path = control_root / "quota-override-deficit-report.json"
    existing_deficit = (
        json.loads(existing_deficit_path.read_text(encoding="utf-8"))
        if existing_deficit_path.is_file()
        else {}
    )
    previous_cpp_capacity = int(
        existing_deficit.get("candidate_capacity_by_source_language", {}).get(
            "bytedance_multi_swe_rl_fallback|cpp", 0
        )
    )
    formal_capacity = int(
        existing_deficit.get("source_language_capacity_upper_bound", 0)
    )
    current_capacity_fields = {
        "formal_capacity_before_probe": formal_capacity,
        "formal_capacity_after_probe": formal_capacity,
        "bytedance_cpp_capacity": previous_cpp_capacity,
        "cpp_target": BYTEDANCE_SOURCE_TARGETS["cpp"],
        "remaining_cpp_deficit": (
            BYTEDANCE_SOURCE_TARGETS["cpp"] - previous_cpp_capacity
        ),
        "caps_preserved": {"repository": 15, "lineage": 2},
        "split_isolation_preserved": True,
        "overlap_rows_admitted": 0,
    }
    if report_path.is_file():
        frozen = json.loads(report_path.read_text(encoding="utf-8"))
        reusable = (
            frozen.get("dataset_revision") == spec["revision"]
            and frozen.get("parquet_sha256") == receipt["sha256"]
            and int(frozen.get("bounded_probe_limit", 0)) >= limit
            and len(frozen.get("probes", [])) >= limit
            and all(
                probe.get("admission", {}).get("status")
                == "rejected_no_exact_buggy_parent_commit"
                for probe in frozen.get("probes", [])[:limit]
            )
        )
        if reusable:
            frozen.update(current_capacity_fields)
            write_json(report_path, frozen)
            return frozen
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for SWE-smith preflight") from exc
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    probes = []
    seen_images = set()
    evidence_root = control_root / "license-evidence" / "swe-smith-cpp"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_name = str(row.get("image_name", ""))
        if not image_name or image_name in seen_images:
            continue
        seen_images.add(image_name)
        identity = _docker_hub_digest(image_name)
        subprocess.run(
            ["docker", "pull", identity["repo_digest"]],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        inspect = subprocess.run(
            ["docker", "image", "inspect", identity["repo_digest"], "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        patch_path = cache / "preflight" / f"{row['instance_id']}.mutation.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_bytes = str(row["patch"]).encode()
        if patch_path.is_file() and patch_path.read_bytes() != patch_bytes:
            raise RuntimeError(f"{row['instance_id']}: frozen mutation patch drift")
        if not patch_path.is_file():
            atomic_write(patch_path, patch_bytes)
        metadata = _docker_run_probe(
            identity["repo_digest"],
            (
                "set -eu; "
                "git -C /testbed rev-parse HEAD; "
                "git -C /testbed remote get-url origin; "
                "git -C /testbed apply --check /evidence/mutation.patch"
            ),
            patch_path=patch_path,
        )
        metadata_lines = metadata.stdout.splitlines()
        if metadata.returncode or len(metadata_lines) < 2:
            raise RuntimeError(
                f"{row['instance_id']}: image metadata/patch probe failed: {metadata.stderr}"
            )
        image_head, origin = metadata_lines[:2]
        license_result = _docker_run_probe(
            identity["repo_digest"],
            (
                "set -eu; "
                "for f in /testbed/LICENSE /testbed/LICENSE.txt /testbed/COPYING; do "
                "if [ -f \"$f\" ]; then cat \"$f\"; exit 0; fi; done; exit 1"
            ),
        )
        if license_result.returncode:
            raise RuntimeError(f"{row['instance_id']}: immutable image license missing")
        license_bytes = license_result.stdout.encode()
        license_path = evidence_root / f"{row['instance_id']}.{image_head}.LICENSE"
        if license_path.is_file() and license_path.read_bytes() != license_bytes:
            raise RuntimeError(f"{row['instance_id']}: frozen image license drift")
        if not license_path.is_file():
            atomic_write(license_path, license_bytes)
        test_name = str((row.get("FAIL_TO_PASS") or [""])[0])
        test_regex = f"^{re.escape(test_name)}$"
        cycle_command = (
            "set -u; cd /testbed; "
            f"ctest --test-dir build -R {shlex.quote(test_regex)} --output-on-failure; "
            "parent_rc=$?; "
            "git apply /evidence/mutation.patch; "
            "cmake --build build -j2; bug_build_rc=$?; "
            f"ctest --test-dir build -R {shlex.quote(test_regex)} --output-on-failure; "
            "bug_rc=$?; "
            "git apply -R /evidence/mutation.patch; "
            "cmake --build build -j2; restored_build_rc=$?; "
            f"ctest --test-dir build -R {shlex.quote(test_regex)} --output-on-failure; "
            "restored_rc=$?; "
            "printf '\\nPROBE_RC %s %s %s %s %s\\n' "
            "\"$parent_rc\" \"$bug_build_rc\" \"$bug_rc\" "
            "\"$restored_build_rc\" \"$restored_rc\""
        )
        cycles = [
            _docker_run_probe(
                identity["repo_digest"],
                cycle_command,
                patch_path=patch_path,
                timeout=600,
            )
            for _ in range(2)
        ]
        replay_codes = []
        for cycle in cycles:
            match = re.search(r"PROBE_RC (\d+) (\d+) (\d+) (\d+) (\d+)", cycle.stdout)
            if cycle.returncode or not match:
                replay_codes.append(None)
            else:
                replay_codes.append([int(value) for value in match.groups()])
        bug_direction_confirmed = (
            len(replay_codes) == 2
            and all(
                codes is not None
                and codes[0] == 0
                and codes[1] == 0
                and codes[2] != 0
                and codes[3:] == [0, 0]
                for codes in replay_codes
            )
        )
        probe = {
            **identity,
            "image_id": inspect.stdout.strip(),
            "image_head": image_head,
            "origin": origin,
            "instance_id": row["instance_id"],
            "repository": row["repo"],
            "test_name": test_name,
            "fresh_offline_replay_codes": replay_codes,
            "patch_applies_to_image_head": True,
            "bug_present_only_after_patch": bug_direction_confirmed,
            "inverse_patch_restores_tests": bug_direction_confirmed,
            "license_evidence_path": license_path.relative_to(control_root).as_posix(),
            "license_evidence_sha256": sha256_file(license_path),
            "mutation_patch_sha256": sha256_file(patch_path),
        }
        probe["admission"] = swe_smith_admission_decision(row, probe)
        probes.append(probe)
        if len(probes) >= limit:
            break
    report = {
        "schema_version": "general_coding_replay_swe_smith_cpp_preflight_v1",
        "status": "rejected_no_exact_buggy_parent_commit",
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "parquet_sha256": receipt["sha256"],
        "rows": spec["rows"],
        "bounded_probe_limit": limit,
        "probes": probes,
        "candidate_rows_admitted": 0,
        "recovered_capacity": 0,
        **current_capacity_fields,
        "alternative_source": SWE_REBENCH_V2_ALTERNATIVE,
    }
    write_json(report_path, report)
    return report


def audit_swe_smith_metadata(control_root: Path) -> dict[str, Any]:
    """Freeze the primary-source rejection without guessing missing evidence."""
    sources: dict[str, Any] = {}
    for language, spec in SWE_SMITH_SOURCES.items():
        cache = control_root / "source-cache" / "swe-smith" / spec["revision"]
        api = _get_json(
            f"https://huggingface.co/api/datasets/{spec['dataset_id']}/revision/{spec['revision']}",
            cache / "dataset-api.json",
        )
        features = [
            item.get("name")
            for item in ((api.get("cardData") or {}).get("dataset_info") or {}).get("features", [])
            if isinstance(item, Mapping)
        ]
        dataset_license = (api.get("cardData") or {}).get("license")
        missing = [
            name
            for name, present in {
                "full_base_commit": "base_commit" in features,
                "per_repository_license": False,
                "dataset_license": bool(dataset_license),
            }.items()
            if not present
        ]
        sources[language] = {
            "dataset_id": spec["dataset_id"],
            "revision": spec["revision"],
            "rows": spec["rows"],
            "parquet_sha256": spec["parquet_sha256"],
            "features": features,
            "dataset_license": dataset_license,
            "status": "rejected_incomplete_provenance" if missing else "admissible",
            "missing_evidence": missing,
            "candidate_rows_admitted": 0 if missing else None,
            "dataset_api_sha256": sha256_file(cache / "dataset-api.json"),
        }
    report = {
        "schema_version": "general_coding_replay_swe_smith_evidence_v1",
        "status": "rejected_incomplete_provenance",
        "reason": (
            "language-specific rows omit the executable buggy parent commit. Image "
            "HEAD can recover the pristine synthetic repository commit, but SWE-smith "
            "patches introduce bugs after that commit; the resulting task parent is "
            "an uncommitted state. Per-repository license evidence is also absent "
            "from dataset metadata."
        ),
        "sources": sources,
        "alternative_source": SWE_REBENCH_V2_ALTERNATIVE,
        "tests_executed": 0,
        "verified_rows": 0,
    }
    write_json(control_root / "swe-smith-source-evidence.json", report)
    return report


def _test_names(value: Any, state: str) -> list[str]:
    if isinstance(value, Mapping):
        return sorted(
            str(name)
            for name, result in value.items()
            if isinstance(result, Mapping) and result.get("fix") == state
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sorted(str(item) for item in value)
    return []


def adapt_bytedance_row(
    row: Mapping[str, Any],
    language: str,
    *,
    source_file: str,
    source_file_sha256: str,
) -> dict[str, Any]:
    """Adapt one original ByteDance row without asserting local verification."""
    base = row.get("base") if isinstance(row.get("base"), Mapping) else {}
    commit = str(base.get("sha", ""))
    patch = str(row.get("fix_patch", ""))
    test_patch = str(row.get("test_patch", ""))
    instance_id = str(row.get("instance_id", ""))
    repository = f"{row.get('org')}/{row.get('repo')}"
    fail_to_pass = _test_names(row.get("f2p_tests"), "PASS")
    pass_to_pass = _test_names(row.get("p2p_tests"), "PASS")
    if not FULL_COMMIT.fullmatch(commit):
        raise ValueError(f"{instance_id}: exact base commit missing")
    # PASS_TO_PASS is legitimately empty for some upstream executable tasks;
    # preserve that empty list rather than fabricating regression labels.
    if not patch.strip() or not fail_to_pass:
        raise ValueError(f"{instance_id}: executable patch/test evidence missing")
    task_type, task_evidence = derive_repository_task_type(row)
    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(f"{instance_id}: unsupported evidence-derived task type")
    normalized_patch_hash = sha256_bytes(
        "\n".join(line.rstrip() for line in patch.replace("\r\n", "\n").splitlines()).encode()
    )
    test_set_hash = sha256_bytes(
        canonical_json({"FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass}).encode()
    )
    lineage = f"{BYTEDANCE_DATASET_ID}@{BYTEDANCE_REVISION}:{instance_id}"
    return {
        "schema_version": "general_coding_replay_candidate_source_v3",
        "case_id": f"gc-replay-bytedance_multi_swe_rl_fallback-{instance_id}",
        "source_id": "bytedance_multi_swe_rl_fallback",
        "dataset_id": BYTEDANCE_DATASET_ID,
        "dataset_revision": BYTEDANCE_REVISION,
        "dataset_row_id": instance_id,
        "source_file": source_file,
        "source_file_sha256": source_file_sha256,
        "source_record_sha256": sha256_bytes(canonical_json(row).encode()),
        "source_lineage_id": lineage,
        "upstream_repository": f"https://github.com/{repository}.git",
        "base_commit": commit,
        "problem_id": instance_id,
        "problem_statement": "\n\n".join(
            value for value in (str(row.get("title", "")), str(row.get("body", ""))) if value
        ),
        "target_patch": patch,
        "target_patch_sha256": sha256_bytes(patch.encode()),
        "normalized_patch_hash": normalized_patch_hash,
        "test_patch": test_patch,
        "test_patch_sha256": sha256_bytes(test_patch.encode()),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "test_set_hash": test_set_hash,
        "primary_language": language,
        "primary_task_type": task_type,
        "task_type_evidence": task_evidence,
        "status": "candidate_pending_local_double_replay",
        "verified": False,
        "local_replay_passes": 0,
        "network_policy": "disabled_for_verification",
        "gpu_required": False,
    }


def _stable_candidate_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if (row.get("container_image") or {}).get("repo_digest") else 1,
        str(row.get("primary_task_type")),
        sha256_bytes(
            f"{row.get('dataset_revision')}\0{row.get('problem_id')}\0"
            f"{row.get('normalized_patch_hash')}".encode()
        ),
        str(row.get("case_id")),
    )


def select_fallback_candidates(
    rows: Iterable[Mapping[str, Any]],
    targets: Mapping[str, int] = BYTEDANCE_SOURCE_TARGETS,
    *,
    repository_cap: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministically while preserving repository diversity."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[str(row["primary_language"])].append(dict(row))
    selected: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    repo_counts: Counter[str] = Counter()
    for language, target in targets.items():
        task_counts: Counter[str] = Counter()
        remaining = sorted(pools.get(language, []), key=_stable_candidate_order)
        while len([row for row in selected if row["primary_language"] == language]) < target:
            eligible = [
                row
                for row in remaining
                if repo_counts[str(row["upstream_repository"])] < repository_cap
            ]
            if not eligible:
                break
            # Prefer scarce task types, immutable images, then a stable hash.
            availability = Counter(str(row["primary_task_type"]) for row in eligible)
            chosen = min(
                eligible,
                key=lambda row: (
                    task_counts[str(row["primary_task_type"])],
                    availability[str(row["primary_task_type"])],
                    _stable_candidate_order(row),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
            repo_counts[str(chosen["upstream_repository"])] += 1
            task_counts[str(chosen["primary_task_type"])] += 1
        count = sum(row["primary_language"] == language for row in selected)
        if count < target:
            deficits[language] = target - count
    selected.sort(key=lambda row: str(row["case_id"]))
    report = {
        "status": "exact" if not deficits else "quota_deficits",
        "targets": dict(targets),
        "selected": len(selected),
        "selected_by_language": dict(Counter(row["primary_language"] for row in selected)),
        "selected_by_task_type": dict(Counter(row["primary_task_type"] for row in selected)),
        "selected_by_repository": dict(Counter(row["upstream_repository"] for row in selected)),
        "deficits": deficits,
        "repository_cap": repository_cap,
        "deterministic": True,
        "verified_rows": 0,
    }
    return selected, report


def _github_license_evidence(
    control_root: Path,
    repository: str,
    base_commit: str,
) -> dict[str, Any]:
    slug = repository.removeprefix("https://github.com/").removesuffix(".git")
    evidence_dir = control_root / "license-evidence" / "bytedance-multi-swe-rl"
    stem = slug.replace("/", "__")
    incompatible = BYTEDANCE_INCOMPATIBLE_LICENSES.get(slug)
    if incompatible:
        raise RuntimeError(
            f"{slug}@{base_commit}: incompatible upstream license {incompatible!r}"
        )
    immutable_file = BYTEDANCE_IMMUTABLE_LICENSE_FILES.get(slug)
    if immutable_file:
        spdx, license_file = immutable_file
        if spdx not in PERMISSIVE_SOURCE_LICENSES:
            raise RuntimeError(f"{slug}@{base_commit}: license policy rejected {spdx!r}")
        source_url = (
            f"https://raw.githubusercontent.com/{slug}/{base_commit}/{license_file}"
        )
        license_file_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", license_file)
        text_path = evidence_dir / (
            f"{stem}.{base_commit}.{license_file_stem}.LICENSE"
        )
        receipt = _download(source_url, text_path)
        provenance = {
            "schema_version": "immutable_upstream_license_evidence_v1",
            "repository": slug,
            "base_commit": base_commit,
            "license_file": license_file,
            "source_url": source_url,
            "spdx": spdx,
            "license_evidence_sha256": receipt["sha256"],
        }
        provenance_path = evidence_dir / (
            f"{stem}.{base_commit}.{license_file_stem}.license-source.json"
        )
        if provenance_path.is_file():
            frozen = json.loads(provenance_path.read_text(encoding="utf-8"))
            if frozen != provenance:
                raise RuntimeError(f"{slug}@{base_commit}: frozen license provenance drift")
        else:
            write_json(provenance_path, provenance)
        return {
            "repository": slug,
            "base_commit": base_commit,
            "spdx": spdx,
            "license_file": license_file,
            "source_url": source_url,
            "api_evidence_path": provenance_path.relative_to(control_root).as_posix(),
            "api_evidence_sha256": sha256_file(provenance_path),
            "license_evidence_path": text_path.relative_to(control_root).as_posix(),
            "license_evidence_sha256": receipt["sha256"],
        }
    api_path = evidence_dir / f"{stem}.{base_commit}.github-license-api.json"
    payload = _get_json(
        f"https://api.github.com/repos/{slug}/license?ref={base_commit}",
        api_path,
    )
    spdx = str((payload.get("license") or {}).get("spdx_id", ""))
    if spdx not in PERMISSIVE_SOURCE_LICENSES:
        raise RuntimeError(f"{slug}@{base_commit}: incompatible or unknown license {spdx!r}")
    try:
        content = base64.b64decode(str(payload["content"]), validate=False)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"{slug}@{base_commit}: license content missing") from exc
    text_path = evidence_dir / f"{stem}.{base_commit}.LICENSE"
    if text_path.is_file() and text_path.read_bytes() != content:
        raise RuntimeError(f"{slug}@{base_commit}: frozen license drift")
    if not text_path.is_file():
        atomic_write(text_path, content)
    return {
        "repository": slug,
        "base_commit": base_commit,
        "spdx": spdx,
        "api_evidence_path": api_path.relative_to(control_root).as_posix(),
        "api_evidence_sha256": sha256_file(api_path),
        "license_evidence_path": text_path.relative_to(control_root).as_posix(),
        "license_evidence_sha256": sha256_file(text_path),
    }


def _registry_image_identity(repository: str, problem_id: str) -> dict[str, str]:
    slug = repository.removeprefix("https://github.com/").removesuffix(".git")
    org, repo = slug.split("/", 1)
    number = problem_id.rsplit("-", 1)[-1]
    if not number.isdigit():
        raise RuntimeError(f"{problem_id}: numeric pull request suffix missing")
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
    token = str(token_response.json()["token"])
    response = requests.head(
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
    response.raise_for_status()
    digest = response.headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(f"{problem_id}: registry returned no immutable image digest")
    return {
        "repo_tag": f"{image_repository}:{tag}",
        "repo_digest": f"{image_repository}@{digest}",
        "manifest_digest": digest,
    }


def freeze_bytedance_fallback(control_root: Path) -> dict[str, Any]:
    """Selectively freeze C++/Rust fallback rows and immutable dependencies."""
    cache = control_root / "source-cache" / "bytedance-multi-swe-rl" / BYTEDANCE_REVISION
    dataset_api = _get_json(
        f"https://huggingface.co/api/datasets/{BYTEDANCE_DATASET_ID}/revision/"
        f"{BYTEDANCE_REVISION}",
        cache / "dataset-api.json",
    )
    if dataset_api.get("sha") != BYTEDANCE_REVISION:
        raise RuntimeError("ByteDance fallback dataset revision drift")
    readme = _download(
        f"https://huggingface.co/datasets/{BYTEDANCE_DATASET_ID}/resolve/"
        f"{BYTEDANCE_REVISION}/README.md",
        cache / "README.md",
    )
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    supplemental = []
    for path in (
        control_root / "general-coding-dev-reservations.jsonl",
        control_root / "multi-swe-bench-eval-only-reservations.jsonl",
    ):
        if path.is_file():
            supplemental.extend(read_jsonl(path))
    dimensions = _exclusion_dimensions(registry, supplemental)
    candidates = []
    source_files = []
    overlap_rejections = []
    schema_rejections = []
    for language, paths in BYTEDANCE_FILES.items():
        for remote_path in paths:
            local_path = cache / remote_path
            receipt = _download(
                f"https://huggingface.co/datasets/{BYTEDANCE_DATASET_ID}/resolve/"
                f"{BYTEDANCE_REVISION}/{remote_path}",
                local_path,
            )
            source_files.append(
                {
                    **receipt,
                    "path": local_path.relative_to(control_root).as_posix(),
                    "remote_path": remote_path,
                }
            )
            for raw in read_jsonl(local_path):
                try:
                    candidate = adapt_bytedance_row(
                        raw,
                        language,
                        source_file=remote_path,
                        source_file_sha256=receipt["sha256"],
                    )
                except ValueError as exc:
                    schema_rejections.append(
                        {"instance_id": raw.get("instance_id"), "detail": str(exc)}
                    )
                    continue
                slug = candidate["upstream_repository"].removeprefix(
                    "https://github.com/"
                ).removesuffix(".git")
                reasons = overlap_reasons(candidate, dimensions)
                if slug in MULTISWE_HELD_OUT_REPOSITORIES:
                    reasons.append("multi_swe_bench_eval_only_repository")
                if reasons:
                    overlap_rejections.append(
                        {
                            "case_id": candidate["case_id"],
                            "reasons": sorted(set(reasons)),
                        }
                    )
                    continue
                candidates.append(candidate)
    prelicense_selected, prelicense_selection = select_fallback_candidates(candidates)
    license_by_revision = {}
    rejected_license_repositories = {}
    license_errors = []
    image_errors = []
    license_ready = []
    for candidate in candidates:
        repository = str(candidate["upstream_repository"])
        base_commit = str(candidate["base_commit"])
        key = (repository, base_commit)
        if repository in rejected_license_repositories:
            continue
        if key not in license_by_revision:
            try:
                license_by_revision[key] = _github_license_evidence(
                    control_root, repository, base_commit
                )
            except (requests.RequestException, RuntimeError) as exc:
                rejected_license_repositories[repository] = str(exc)
                license_ready = [
                    row
                    for row in license_ready
                    if row["upstream_repository"] != repository
                ]
                license_by_revision = {
                    frozen_key: evidence
                    for frozen_key, evidence in license_by_revision.items()
                    if frozen_key[0] != repository
                }
                license_errors.append(
                    {
                        "repository": repository,
                        "detail": str(exc),
                        "candidate_rows": sum(
                            row["upstream_repository"] == repository
                            for row in candidates
                        ),
                    }
                )
                continue
        license_evidence = license_by_revision.get(key)
        if not license_evidence:
            continue
        candidate["license_spdx"] = license_evidence["spdx"]
        candidate["license_evidence_path"] = license_evidence["license_evidence_path"]
        license_ready.append(candidate)
    selected, selection = select_fallback_candidates(license_ready)
    write_json(control_root / "fallback-selection-deficits.json", selection)

    for candidate in selected:
        repository = str(candidate["upstream_repository"])
        try:
            candidate["container_image"] = _registry_image_identity(
                repository, str(candidate["problem_id"])
            )
        except (requests.RequestException, RuntimeError) as exc:
            image_errors.append(
                {"case_id": candidate["case_id"], "detail": str(exc)}
            )
    if image_errors:
        write_jsonl(control_root / "fallback-image-rejections.jsonl", image_errors)
    if license_errors:
        write_jsonl(control_root / "fallback-license-rejections.jsonl", license_errors)
    image_ready = [
        row
        for row in selected
        if row.get("license_spdx") and row.get("container_image")
    ]
    image_ready_counts = Counter(row["primary_language"] for row in image_ready)
    final_deficits = {
        language: target - image_ready_counts[language]
        for language, target in BYTEDANCE_SOURCE_TARGETS.items()
        if image_ready_counts[language] < target
    }

    manifest_path = (
        control_root
        / "bytedance-multi-swe-rl-fallback-candidate-source-manifest.jsonl"
    )
    write_jsonl(manifest_path, image_ready)
    write_jsonl(
        control_root / "fallback-overlap-rejections.jsonl", overlap_rejections
    )
    write_jsonl(
        control_root / "fallback-schema-rejections.jsonl", schema_rejections
    )
    report = {
        "schema_version": "general_coding_replay_bytedance_fallback_freeze_v1",
        "status": (
            "candidate_metadata_ready_local_double_replay_not_run"
            if not final_deficits
            else "quota_deficits_after_overlap_and_image_gates"
        ),
        "dataset_id": BYTEDANCE_DATASET_ID,
        "dataset_revision": BYTEDANCE_REVISION,
        "dataset_api_sha256": sha256_file(cache / "dataset-api.json"),
        "readme": readme,
        "source_files": source_files,
        "selective_download": True,
        "full_corpus_downloaded": False,
        "candidate_rows_before_selection": len(candidates),
        "prelicense_selection": prelicense_selection,
        "license_admissible_candidate_rows": len(license_ready),
        "license_rejected_candidate_rows": len(candidates) - len(license_ready),
        "prelicense_selected_rows_dropped_by_license": sum(
            row["upstream_repository"] in rejected_license_repositories
            for row in prelicense_selected
        ),
        "selection": selection,
        "image_available_counts": dict(sorted(image_ready_counts.items())),
        "final_language_deficits": final_deficits,
        "image_rejections": len(image_errors),
        "license_rejections": len(license_errors),
        "overlap_rejections": len(overlap_rejections),
        "schema_rejections": len(schema_rejections),
        "licenses": [
            license_by_revision[key]
            for key in sorted(license_by_revision)
            if license_by_revision[key]
        ],
        "manifest": {
            "path": manifest_path.name,
            "rows": len(image_ready),
            "sha256": sha256_file(manifest_path),
        },
        "verified_rows": 0,
        "local_replay_passes": 0,
        "required_local_replay_passes": 2,
        "network_policy_for_verification": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "bytedance-fallback-source-freeze-report.json", report)
    return report


def build_override_deficit_report(control_root: Path) -> dict[str, Any]:
    """Run the deterministic selector over provenance-ready candidate metadata."""
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    dimensions = _exclusion_dimensions(registry)
    manifests = (
        control_root / "swe-gym-candidate-source-manifest.jsonl",
        control_root / "multi-swe-verified-candidate-source-manifest.jsonl",
        control_root
        / "bytedance-multi-swe-rl-fallback-candidate-source-manifest.jsonl",
    )
    rows = []
    excluded = Counter()
    capacities = Counter()
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for raw in read_jsonl(manifest):
            patch = str(raw.get("target_patch", ""))
            patch_hash = str(raw.get("normalized_patch_hash", "")) or sha256_bytes(
                "\n".join(
                    line.rstrip() for line in patch.replace("\r\n", "\n").splitlines()
                ).encode()
            )
            row = {
                "case_id": raw["case_id"],
                "source_id": raw["source_id"],
                "coding_task_type": raw["primary_task_type"],
                "primary_language": raw["primary_language"],
                "upstream_repository": raw["upstream_repository"],
                "problem_id": raw["problem_id"],
                "source_lineage_id": raw.get("source_lineage_id", ""),
                "source_hash": raw.get("source_record_sha256", ""),
                "normalized_patch_hash": patch_hash,
                "ast_fingerprint": "",
                "diff_hunk_hash": patch_hash,
                "test_set_hash": raw.get("test_set_hash")
                or raw.get("test_patch_sha256", ""),
                "verified_runtime_seconds": 0,
                "status": "candidate_only_not_verified",
            }
            reasons = overlap_reasons(row, dimensions)
            if reasons:
                excluded[str(row["source_id"])] += 1
                continue
            rows.append(row)
            capacities[
                f"{row['source_id']}|{row['primary_language']}"
            ] += 1
    quotas = {
        "source_targets": EFFECTIVE_SOURCE_TARGETS,
        "language_targets": PHASE1_LANGUAGE_TARGETS,
        "task_type_targets": PHASE1_TASK_TYPE_TARGETS,
    }
    _selected, selector = select_replay(rows, quotas, 500)
    raw_counts = {
        "source": Counter(str(row["source_id"]) for row in rows),
        "language": Counter(str(row["primary_language"]) for row in rows),
        "task_type": Counter(str(row["coding_task_type"]) for row in rows),
    }
    capacity_deficits = {
        dimension: {
            key: target - raw_counts[dimension][key]
            for key, target in quotas[f"{dimension}_targets"].items()
            if raw_counts[dimension][key] < target
        }
        for dimension in ("source", "language", "task_type")
    }
    source_language_upper_bound = sum(
        min(
            target,
            capacities[f"{source}|{language}"],
        )
        for source, languages in SOURCE_LANGUAGE_TARGETS.items()
        for language, target in languages.items()
    )
    report = {
        "schema_version": "general_coding_replay_quota_override_deficits_v1",
        "status": selector["status"],
        "scope": "provenance_and_immutable_image_ready_candidates_not_verified",
        "candidate_rows": len(rows),
        "candidate_capacity_by_source_language": dict(sorted(capacities.items())),
        "candidate_capacity_counts": {
            dimension: dict(sorted(counts.items()))
            for dimension, counts in raw_counts.items()
        },
        "proven_capacity_deficits": capacity_deficits,
        "source_language_capacity_upper_bound": source_language_upper_bound,
        "source_language_cross_quota_deficit": 500 - source_language_upper_bound,
        "excluded_by_registry": dict(sorted(excluded.items())),
        "selected_at_stall": selector["selected"],
        "counts_at_stall": selector["counts"],
        "deficits": selector["deficits"],
        "candidate_cross_distribution": selector["candidate_cross_distribution"],
        "verified_rows": 0,
        "note": "No candidate counts toward Replay 500 before two fresh local replays.",
    }
    write_json(control_root / "quota-override-deficit-report.json", report)
    return report


def candidate_preflight(
    rows: Sequence[Mapping[str, Any]],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    effective = validate_override_document(override)
    errors = []
    for row in rows:
        missing = [
            key
            for key in (
                "dataset_revision",
                "base_commit",
                "target_patch_sha256",
                "test_patch_sha256",
                "fail_to_pass",
                "pass_to_pass",
                "license_spdx",
                "license_evidence_path",
                "source_lineage_id",
                "normalized_patch_hash",
                "test_set_hash",
            )
            if key not in row or (key != "pass_to_pass" and not row.get(key))
        ]
        image = row.get("container_image") or {}
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", str(image.get("repo_digest", ""))):
            missing.append("container_image.repo_digest")
        if row.get("verified") is not False or row.get("local_replay_passes") != 0:
            missing.append("unverified_status")
        if missing:
            errors.append({"case_id": row.get("case_id"), "missing_or_invalid": sorted(missing)})
    counts = {
        "language": dict(Counter(str(row.get("primary_language")) for row in rows)),
        "task_type": dict(Counter(str(row.get("primary_task_type")) for row in rows)),
        "source": dict(Counter(str(row.get("source_id")) for row in rows)),
    }
    task_capacity_deficits = {
        task: max(0, target - counts["task_type"].get(task, 0))
        for task, target in effective["task_type_targets"].items()
        if target > counts["task_type"].get(task, 0)
    }
    return {
        "schema_version": "general_coding_replay_quota_override_preflight_v1",
        "status": "ready_for_local_double_replay" if not errors else "blocked_metadata",
        "candidate_rows": len(rows),
        "metadata_errors": errors,
        "counts": counts,
        "candidate_only_task_type_deficits": task_capacity_deficits,
        "verified_rows": 0,
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }


def build_override_evidence(control_root: Path, override_path: Path) -> dict[str, Any]:
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    if not isinstance(override, Mapping):
        raise ValueError("quota override must be a mapping")
    effective = validate_override_document(override)
    swe_smith = audit_swe_smith_metadata(control_root)
    report = {
        "schema_version": "general_coding_replay_quota_override_evidence_v1",
        "status": "fallback_required",
        "proven_infeasibility": {
            "source": "PrimeIntellect/Multi-SWE-RL-Verified",
            "revision": "80de95c62ac792c99dcfa8e26569bcd7d036bdc3",
            "cpp_rows": 0,
            "rust_rows": 0,
            "cross_quota_deficit": 125,
        },
        "effective_quotas": effective,
        "primary_mutation_source": swe_smith,
        "fallback": {
            "dataset_id": BYTEDANCE_DATASET_ID,
            "revision": BYTEDANCE_REVISION,
            "language_targets": BYTEDANCE_SOURCE_TARGETS,
            "status": "candidate_freeze_pending",
            "verified_rows": 0,
        },
        "override_sha256": sha256_file(override_path),
    }
    write_json(control_root / "quota-override-evidence.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze General Coding Replay quota-override evidence.")
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--quota-override", type=Path, required=True)
    parser.add_argument("--freeze-fallback", action="store_true")
    parser.add_argument(
        "--preflight-swe-smith-cpp-limit",
        type=int,
        default=0,
        help="Pull and probe this many distinct pinned SWE-smith C++ images.",
    )
    args = parser.parse_args(argv)
    report = build_override_evidence(args.control_root, args.quota_override)
    if args.preflight_swe_smith_cpp_limit:
        report["primary_mutation_source"]["cpp_preflight"] = preflight_swe_smith_cpp(
            args.control_root,
            limit=args.preflight_swe_smith_cpp_limit,
        )
    if args.freeze_fallback:
        report["fallback"] = freeze_bytedance_fallback(args.control_root)
        report["deficit_report"] = build_override_deficit_report(args.control_root)
    write_json(args.control_root / "quota-override-evidence.json", report)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
