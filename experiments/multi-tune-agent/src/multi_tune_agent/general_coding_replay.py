"""General Coding Replay v1 inventory, verification, selection, and packaging.

The verifier is intentionally Linux-only: every untrusted command runs in a
fresh user/network namespace with GPU visibility removed.  It never fetches a
repository; source manifests must point at an already materialized local git
repository and pin full immutable commit IDs.
"""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
import requests
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SCHEMA_VERSION = "general_coding_replay_v1"
VERIFIER_VERSION = "general_coding_replay_verifier_v6"
SWE_GYM_DATASET_ID = "SWE-Gym/SWE-Gym"
SWE_GYM_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SWE_GYM_ROWS = 2438
SWE_GYM_PARQUET_SHA256 = "60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97"
OPENCODE_DATASET_ID = "nvidia/OpenCodeInstruct"
OPENCODE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
OPENCODE_ROWS = 5_000_000
MULTISWE_VERIFIED_DATASET_ID = "PrimeIntellect/Multi-SWE-RL-Verified"
MULTISWE_VERIFIED_REVISION = "80de95c62ac792c99dcfa8e26569bcd7d036bdc3"
MULTISWE_VERIFIED_ROWS = 2232
MULTISWE_VALIDATION_LANGUAGE_COUNTS = {
    "c": 83,
    "cpp": 0,
    "go": 901,
    "java": 691,
    "js": 294,
    "rust": 0,
    "ts": 263,
}
MULTISWE_TRAIN_FETCH_TARGETS = {"go": 100, "js": 75, "ts": 75}
MULTISWE_HELD_OUT_REPOSITORIES = {
    "fmtlib/fmt",
    "grpc/grpc-go",
    "axios/axios",
    "darkreader/darkreader",
    "vuejs/core",
    "tokio-rs/tokio",
}
MULTISWE_UPSTREAM_REPOSITORY = "https://github.com/multi-swe-bench/multi-swe-bench"
MULTISWE_UPSTREAM_REVISION = "24f493f8a103e72312ded4f6b9c89f081d69cb09"
MULTISWE_UPSTREAM_LICENSE = "Apache-2.0"
HUMANEVAL_DATASET_ID = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
HUMANEVAL_TEST_SHA256 = "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"
HUMANEVAL_TEST_ROWS = 164
MBPP_DATASET_ID = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_SANITIZED_TEST_SHA256 = "e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850"
MBPP_SANITIZED_TEST_ROWS = 257
LIVECODEBENCH_DATASET_ID = "livecodebench/code_generation_lite"
LIVECODEBENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LIVECODEBENCH_SOURCE_FILE = "test6.jsonl"
LIVECODEBENCH_SOURCE_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
LIVECODEBENCH_SOURCE_SIZE = 134_303_240
LIVECODEBENCH_WINDOW = ("2025-01-01T00:00:00", "2025-04-30T23:59:59")
PERMISSIVE_SOURCE_LICENSES = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "ISC",
        "MIT",
        "PSF-2.0",
    }
)
OPENCODE_SCHEMA = (
    "id",
    "input",
    "output",
    "domain",
    "generation_algorithm",
    "llm_judgement",
    "unit_tests",
    "tests_execution_status",
    "average_test_score",
)
PHASE1_SOURCE_TARGETS = {"swe_gym": 250, "multi_swe_rl_verified": 250}
PHASE1_LANGUAGE_TARGETS = {
    "python": 250,
    "cpp": 75,
    "go": 50,
    "javascript_typescript": 75,
    "rust": 50,
}
PHASE1_TASK_TYPE_TARGETS = {
    "repository_bug_fix_or_test_repair": 220,
    "function_implementation": 100,
    "refactor_or_api_adaptation": 80,
    "build_dependency_or_config": 50,
    "non_kernel_performance_repair": 30,
    "typing_validation_or_docs_with_test": 20,
}
FULL_SHA = re.compile(r"^[0-9a-f]{40,64}$")
CASE_ID = re.compile(r"^gc-replay-[a-z0-9_-]+-[a-zA-Z0-9._-]+$")
ALLOWED_LANGUAGES = {
    "python",
    "cpp",
    "javascript_typescript",
    "go",
    "rust",
}
ALLOWED_TASK_TYPES = {
    "repository_bug_fix_or_test_repair",
    "function_implementation",
    "refactor_or_api_adaptation",
    "build_dependency_or_config",
    "non_kernel_performance_repair",
    "typing_validation_or_docs_with_test",
}
REJECTION_REASONS = frozenset(
    {
        "license_unknown",
        "license_incompatible",
        "parent_revision_missing",
        "parent_not_reproducible",
        "target_patch_missing",
        "patch_apply_failed",
        "tests_missing",
        "dependency_restore_failed",
        "parent_already_passes",
        "targeted_tests_failed",
        "regression_tests_failed",
        "test_timeout",
        "fresh_replay_mismatch",
        "forbidden_path_modified",
        "benchmark_overlap",
        "split_leakage",
        "duplicate_source",
        "duplicate_patch",
        "gpu_required",
        "network_isolation_unavailable",
        "compile_or_typecheck_failed",
        "immutable_revision_required",
        "secret_detected",
    }
)
REQUIRED_SOURCE_FIELDS = (
    "source_id",
    "dataset_id",
    "dataset_revision",
    "upstream_repository",
    "base_commit",
    "problem_id",
    "license_spdx",
    "license_evidence_path",
    "primary_language",
    "primary_task_type",
    "problem_statement_path",
    "target_patch_path",
    "allowed_paths",
    "install_command",
    "compile_or_typecheck_command",
    "targeted_test_command",
    "full_regression_command",
    "network_policy",
    "gpu_required",
)
DEFAULT_TIMEOUTS = {
    "checkout_dependency": 30 * 60,
    "parent": 30 * 60,
    "target": 30 * 60,
    "regression": 90 * 60,
    "fresh_total": 180 * 60,
}
GPU_PATCH_SUFFIXES = {".cu", ".cuh", ".hip"}
SECRET_NAME = re.compile(
    r"(^|/)(\.env($|\.)|id_rsa|id_ed25519|credentials\.json|secrets?($|[._-]))",
    re.IGNORECASE,
)
SECRET_TEXT = re.compile(
    r"(-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|"
    r"(?:aws_secret_access_key|api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


class ReplayRejected(RuntimeError):
    """A standardized, auditable verification rejection."""

    def __init__(self, reason: str, detail: str):
        if reason not in REJECTION_REASONS:
            raise ValueError(f"non-standard rejection reason: {reason}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    kwargs = {} if isinstance(data, bytes) else {"encoding": "utf-8"}
    with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **kwargs) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _load_document(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _manifest_entries(document: Any) -> list[dict[str, Any]]:
    if document is None:
        return []
    raw = document.get("sources", []) if isinstance(document, Mapping) else document
    if not isinstance(raw, list):
        raise ValueError("source manifest must be a list or contain a sources list")
    return [dict(item) for item in raw]


def _resolve_evidence(entry: Mapping[str, Any], field: str, manifest_dir: Path) -> Path:
    raw = str(entry[field])
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (manifest_dir / path).resolve()


def validate_source_entry(entry: Mapping[str, Any], manifest_dir: Path) -> None:
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in entry]
    if missing:
        raise ValueError("missing source fields: " + ", ".join(missing))
    if not FULL_SHA.fullmatch(str(entry["dataset_revision"])):
        raise ReplayRejected("immutable_revision_required", "dataset_revision is not a full commit")
    if not re.fullmatch(r"[0-9a-f]{40}", str(entry["base_commit"])):
        raise ReplayRejected("immutable_revision_required", "base_commit is not a 40-character commit")
    if entry["primary_language"] not in ALLOWED_LANGUAGES:
        raise ValueError(f"unsupported primary_language: {entry['primary_language']}")
    if entry["primary_task_type"] not in ALLOWED_TASK_TYPES:
        raise ValueError(f"unsupported primary_task_type: {entry['primary_task_type']}")
    if entry["network_policy"] != "disabled":
        raise ValueError("network_policy must be disabled")
    if entry["gpu_required"] is not False:
        raise ReplayRejected("gpu_required", "General Coding Replay forbids GPU cases")
    if not isinstance(entry["allowed_paths"], list) or not entry["allowed_paths"]:
        raise ValueError("allowed_paths must be a non-empty list")
    if str(entry["license_spdx"]).upper() in {"", "UNKNOWN", "NOASSERTION"}:
        raise ReplayRejected("license_unknown", "a known SPDX license is required")
    for field in ("license_evidence_path", "problem_statement_path", "target_patch_path"):
        if not _resolve_evidence(entry, field, manifest_dir).is_file():
            reason = "target_patch_missing" if field == "target_patch_path" else "parent_not_reproducible"
            raise ReplayRejected(reason, f"missing {field}")
    if not all(str(entry[field]).strip() for field in (
        "install_command",
        "compile_or_typecheck_command",
        "targeted_test_command",
        "full_regression_command",
    )):
        raise ReplayRejected("tests_missing", "all install/compile/test commands are required")


def _registry_fingerprints(registry: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "problem_ids",
        "repositories",
        "prompt_hashes",
        "text_hashes",
        "ast_fingerprints",
        "signature_fingerprints",
        "normalized_patch_hashes",
        "source_lineages",
        "contract_hashes",
        "contract_terms",
        "symbols",
    ):
        for item in registry.get(key, []):
            values.add(str(item))
    for entry in registry.get("entries", []):
        if isinstance(entry, Mapping):
            values.update(str(value) for value in entry.values() if isinstance(value, (str, int)))
    for entry in registry.get("mirror_rewrite_fingerprints", []):
        if isinstance(entry, Mapping) and entry.get("fingerprint"):
            values.add(str(entry["fingerprint"]))
    return values


def benchmark_overlap(entry: Mapping[str, Any], registry: Mapping[str, Any]) -> bool:
    frozen = _registry_fingerprints(registry)
    candidates = {
        str(entry.get("problem_id", "")),
        str(entry.get("upstream_repository", "")),
        str(entry.get("source_lineage_id", "")),
        str(entry.get("normalized_prompt_hash", "")),
        str(entry.get("text_hash", "")),
        str(entry.get("ast_fingerprint", "")),
        str(entry.get("signature_fingerprint", "")),
        str(entry.get("normalized_patch_hash", "")),
    }
    return bool((candidates - {""}).intersection(frozen))


def build_inventory(source_manifest: Path, exclusion_registry: Path, output: Path) -> dict[str, Any]:
    entries = _manifest_entries(_load_document(source_manifest))
    registry = _load_document(exclusion_registry)
    if not isinstance(registry, Mapping):
        raise ValueError("exclusion registry must be an object")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in entries:
        source_id = str(raw.get("source_id", "unknown"))
        stable_id = str(raw.get("stable_id") or raw.get("problem_id") or source_id)
        case_id = str(raw.get("case_id") or f"gc-replay-{source_id}-{stable_id}")
        try:
            validate_source_entry(raw, source_manifest.parent)
            if not CASE_ID.fullmatch(case_id):
                raise ValueError(f"invalid case_id: {case_id}")
            if case_id in seen_ids:
                raise ReplayRejected("duplicate_source", "duplicate case_id in source manifest")
            if benchmark_overlap(raw, registry):
                raise ReplayRejected("benchmark_overlap", "candidate matches frozen exclusion registry")
            seen_ids.add(case_id)
            entry = dict(raw)
            entry.update(
                {
                    "case_id": case_id,
                    "status": "inventoried",
                    "source_lineage_id": str(raw.get("source_lineage_id") or f"{source_id}:{stable_id}"),
                    "source_hash": None,
                    "normalized_patch_hash": None,
                    "test_set_hash": None,
                    "parent_expected_failures": list(raw.get("parent_expected_failures", [])),
                    "target_test_receipt": None,
                    "fresh_verify_receipt": None,
                    "rejection_reason": None,
                    "split": "train",
                    "sample_domain": "general_coding",
                    "task_type": "general_coding_replay",
                    "coding_task_type": raw["primary_task_type"],
                    "primary_language": raw["primary_language"],
                    "_manifest_dir": str(source_manifest.parent.resolve()),
                }
            )
            accepted.append(entry)
        except ReplayRejected as exc:
            rejected.append({"case_id": case_id, "status": "rejected", "rejection_reason": exc.reason, "detail": exc.detail})
    accepted.sort(key=lambda row: row["case_id"])
    write_jsonl(output, accepted)
    write_jsonl(output.with_name("inventory-rejections.jsonl"), rejected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "inventoried": len(accepted),
        "rejected": len(rejected),
        "source_manifest_sha256": sha256_file(source_manifest),
        "exclusion_registry_sha256": sha256_file(exclusion_registry),
        "candidates_sha256": sha256_file(output),
    }
    write_json(output.with_name("inventory-report.json"), report)
    return report


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _copy_git_revision(repo: Path, commit: str, destination: Path) -> None:
    actual = _git(repo, "rev-parse", f"{commit}^{{commit}}").stdout.strip()
    if actual != commit:
        raise ReplayRejected("parent_revision_missing", f"base commit mismatch: {actual}")
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(repo), str(destination)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    _git(destination, "checkout", "--quiet", "--detach", commit)
    if _git(destination, "status", "--porcelain").stdout.strip():
        raise ReplayRejected("parent_not_reproducible", "fresh checkout is dirty")


def _isolated_command(command: str, cwd: Path, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "*",
        }
    )
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--net", "sh", "-lc", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReplayRejected("test_timeout", f"command timed out after {timeout}s: {command}") from exc
    elapsed = time.monotonic() - started
    if result.returncode == 1 and "unshare failed" in result.stderr.lower():
        raise ReplayRejected("network_isolation_unavailable", result.stderr.strip())
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_seconds": elapsed,
        "network": "disabled_linux_namespace",
        "gpu_visibility": "disabled",
        "ok": result.returncode == 0,
    }


def _normalized_patch(patch: str) -> str:
    lines = [line.rstrip() for line in patch.replace("\r\n", "\n").splitlines()]
    return "\n".join(line for line in lines if not line.startswith(("index ", "diff --git "))) + "\n"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReplayRejected("forbidden_path_modified", f"unsafe path: {value}")
    return path.as_posix()


def _untracked_paths(workspace: Path) -> set[str]:
    return set(_git(workspace, "ls-files", "--others", "--exclude-standard").stdout.splitlines())


def _changed_paths(workspace: Path, baseline_untracked: set[str]) -> list[str]:
    result = _git(workspace, "diff", "--name-only", "--diff-filter=ACMRTUXB")
    changed = {line for line in result.stdout.splitlines() if line}
    changed.update(_untracked_paths(workspace) - baseline_untracked)
    return sorted(changed)


def _check_changed_paths(changed: Sequence[str], allowed: Sequence[str], patch_text: str) -> None:
    allowed_paths = [_safe_relative_path(value) for value in allowed]
    forbidden = [
        path
        for path in changed
        if not any(path == item or path.startswith(item.rstrip("/") + "/") for item in allowed_paths)
    ]
    if forbidden:
        raise ReplayRejected("forbidden_path_modified", ", ".join(forbidden))
    if any(Path(path).suffix.lower() in GPU_PATCH_SUFFIXES for path in changed) or re.search(
        r"\b(?:cuda|hipLaunchKernel|triton\.jit)\b", patch_text, re.IGNORECASE
    ):
        raise ReplayRejected("gpu_required", "patch modifies or introduces GPU code")


def _source_tree_hash(workspace: Path) -> str:
    files = _git(workspace, "ls-files").stdout.splitlines()
    digest = hashlib.sha256()
    for relative in sorted(files):
        path = workspace / relative
        if path.is_file():
            digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _ast_fingerprint(workspace: Path, changed: Sequence[str]) -> str:
    parts: list[str] = []
    for relative in changed:
        path = workspace / relative
        if path.suffix == ".py" and path.is_file():
            try:
                parts.append(ast.dump(ast.parse(path.read_text(encoding="utf-8")), include_attributes=False))
            except (SyntaxError, UnicodeDecodeError):
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.is_file():
            parts.append(re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="replace")).strip())
    return sha256_bytes("\0".join(parts).encode())


def _run_replay(candidate: Mapping[str, Any], workspace: Path, patch_path: Path, *, fresh: bool) -> dict[str, Any]:
    timeouts = {**DEFAULT_TIMEOUTS, **candidate.get("timeouts", {})}
    receipts: dict[str, Any] = {}
    started = time.monotonic()
    test_patch_raw = candidate.get("test_patch_path")
    if test_patch_raw:
        manifest_dir = Path(str(candidate.get("_manifest_dir", "."))).resolve()
        test_patch_path = _resolve_evidence(candidate, "test_patch_path", manifest_dir)
        if not test_patch_path.is_file():
            raise ReplayRejected("tests_missing", f"missing test patch: {test_patch_path}")
        test_patch_result = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", str(test_patch_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        receipts["test_patch_apply"] = {
            "returncode": test_patch_result.returncode,
            "stdout": test_patch_result.stdout,
            "stderr": test_patch_result.stderr,
            "ok": test_patch_result.returncode == 0,
        }
        if test_patch_result.returncode:
            raise ReplayRejected("tests_missing", test_patch_result.stderr[-2000:])
    receipts["install"] = _isolated_command(str(candidate["install_command"]), workspace, int(timeouts["checkout_dependency"]))
    if not receipts["install"]["ok"]:
        raise ReplayRejected("dependency_restore_failed", receipts["install"]["stderr"][-2000:])
    receipts["parent_compile"] = _isolated_command(
        str(candidate["compile_or_typecheck_command"]), workspace, int(timeouts["parent"])
    )
    if not receipts["parent_compile"]["ok"]:
        raise ReplayRejected("parent_not_reproducible", receipts["parent_compile"]["stderr"][-2000:])
    receipts["parent_tests"] = _isolated_command(
        str(candidate["targeted_test_command"]), workspace, int(timeouts["parent"])
    )
    parent_may_pass = candidate.get("parent_expectation") == "passes"
    if receipts["parent_tests"]["ok"] and not parent_may_pass:
        raise ReplayRejected("parent_already_passes", "targeted tests passed on parent")
    patch_result = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", str(patch_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipts["patch_apply"] = {
        "returncode": patch_result.returncode,
        "stdout": patch_result.stdout,
        "stderr": patch_result.stderr,
        "ok": patch_result.returncode == 0,
    }
    if patch_result.returncode:
        raise ReplayRejected("patch_apply_failed", patch_result.stderr[-2000:])
    patch_text = patch_path.read_text(encoding="utf-8")
    changed = _patch_paths(patch_text)
    _check_changed_paths(changed, list(candidate["allowed_paths"]), patch_text)
    receipts["target_compile"] = _isolated_command(
        str(candidate["compile_or_typecheck_command"]), workspace, int(timeouts["target"])
    )
    if not receipts["target_compile"]["ok"]:
        raise ReplayRejected("compile_or_typecheck_failed", receipts["target_compile"]["stderr"][-2000:])
    receipts["targeted_tests"] = _isolated_command(
        str(candidate["targeted_test_command"]), workspace, int(timeouts["target"])
    )
    if not receipts["targeted_tests"]["ok"]:
        raise ReplayRejected("targeted_tests_failed", receipts["targeted_tests"]["stderr"][-2000:])
    receipts["regression_tests"] = _isolated_command(
        str(candidate["full_regression_command"]), workspace, int(timeouts["regression"])
    )
    if not receipts["regression_tests"]["ok"]:
        raise ReplayRejected("regression_tests_failed", receipts["regression_tests"]["stderr"][-2000:])
    if fresh and time.monotonic() - started > int(timeouts["fresh_total"]):
        raise ReplayRejected("test_timeout", "fresh replay exceeded total timeout")
    receipts["changed_paths"] = changed
    receipts["source_hash"] = _source_tree_hash(workspace)
    receipts["ast_fingerprint"] = _ast_fingerprint(workspace, changed)
    receipts["total_duration_seconds"] = time.monotonic() - started
    return receipts


def _docker_image_identity(repo_tag: str) -> dict[str, Any] | None:
    docker = shutil.which("docker")
    if not docker:
        return None
    result = subprocess.run(
        [docker, "image", "inspect", repo_tag],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"ambiguous Docker image identity: {repo_tag}")
    image = payload[0]
    repo_digests = sorted(str(value) for value in image.get("RepoDigests", []))
    matching = [value for value in repo_digests if value.split("@", 1)[0] == repo_tag.rsplit(":", 1)[0]]
    if len(matching) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("Id", ""))):
        raise RuntimeError(f"image lacks one immutable local identity: {repo_tag}")
    return {
        "repo_tag": repo_tag,
        "repo_digest": matching[0],
        "image_id": str(image["Id"]),
        "working_dir": str(image.get("Config", {}).get("WorkingDir") or "/testbed"),
    }


def _assert_docker_image_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = _docker_image_identity(str(expected["repo_tag"]))
    if actual is None:
        raise ReplayRejected("dependency_restore_failed", f"container image missing: {expected['repo_tag']}")
    for field in ("repo_tag", "repo_digest", "image_id"):
        if actual[field] != expected[field]:
            raise ReplayRejected(
                "dependency_restore_failed",
                f"container image {field} mismatch: expected {expected[field]}, found {actual[field]}",
            )
    return actual


def _docker_create_args(
    identity: Mapping[str, Any],
    target_patch: Path,
    test_patch: Path | None,
    workdir: str = "/testbed",
) -> list[str]:
    docker = shutil.which("docker") or "docker"
    args = [
        docker,
        "create",
        "--network",
        "none",
        "--workdir",
        workdir,
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--env",
        "HIP_VISIBLE_DEVICES=",
        "--env",
        "ROCR_VISIBLE_DEVICES=",
        "--volume",
        f"{target_patch.resolve()}:/replay/target.patch:ro",
    ]
    if test_patch is not None:
        args.extend(["--volume", f"{test_patch.resolve()}:/replay/test.patch:ro"])
    args.extend([str(identity["image_id"]), "sleep", "infinity"])
    return args


def _validate_container_policy(inspect: Mapping[str, Any]) -> None:
    host = inspect.get("HostConfig", {})
    config = inspect.get("Config", {})
    if host.get("NetworkMode") != "none":
        raise ReplayRejected("network_isolation_unavailable", "container network mode is not none")
    if host.get("Devices") or host.get("DeviceRequests"):
        raise ReplayRejected("gpu_required", "container has host device or GPU requests")
    environment = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in config.get("Env", [])
        if "=" in item
    }
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if environment.get(name) != "":
            raise ReplayRejected("gpu_required", f"{name} must be empty")


def _validate_container_head(receipt: Mapping[str, Any], expected_commit: str) -> None:
    actual = str(receipt.get("stdout", "")).strip()
    if not receipt.get("ok") or actual != expected_commit:
        raise ReplayRejected(
            "parent_revision_missing",
            f"/testbed HEAD mismatch: expected {expected_commit}, found {actual}",
        )


def _docker_exec(container: str, command: str, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            # SWE-Gym/SWE-bench images activate their pinned test environment
            # from Bash login startup. POSIX sh bypasses that activation and can
            # silently select the base Python instead of the testbed Python.
            ["docker", "exec", container, "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReplayRejected("test_timeout", f"container command timed out: {command}") from exc
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_seconds": time.monotonic() - started,
        "network": "docker_none",
        "gpu_visibility": "disabled",
        "ok": result.returncode == 0,
    }


def _chunk_pytest_command(command: str, *, max_chars: int = 24_576) -> list[str]:
    """Split oversized pytest node-id lists below the host/container ARG_MAX."""
    tokens = shlex.split(command)
    test_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if ".py" in token and not token.startswith("-")
        ),
        None,
    )
    if test_index is None or len(command) <= max_chars:
        return [command]
    prefix = tokens[:test_index]
    tests = tokens[test_index:]
    chunks: list[str] = []
    current: list[str] = []
    for test in tests:
        proposed = shlex.join([*prefix, *current, test])
        if current and len(proposed) > max_chars:
            chunks.append(shlex.join([*prefix, *current]))
            current = [test]
        else:
            current.append(test)
    if current:
        chunks.append(shlex.join([*prefix, *current]))
    return chunks


def _docker_exec_pytest(
    container: str, command: str, timeout: int
) -> dict[str, Any]:
    chunks = _chunk_pytest_command(command)
    if len(chunks) == 1:
        return _docker_exec(container, command, timeout)
    started = time.monotonic()
    receipts: list[dict[str, Any]] = []
    for chunk in chunks:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise ReplayRejected("test_timeout", "chunked pytest command timed out")
        receipt = _docker_exec(container, chunk, max(1, int(remaining)))
        receipts.append(receipt)
        if not receipt["ok"]:
            break
    return {
        "command": command,
        "returncode": receipts[-1]["returncode"],
        "stdout": "".join(str(receipt["stdout"]) for receipt in receipts),
        "stderr": "".join(str(receipt["stderr"]) for receipt in receipts),
        "duration_seconds": time.monotonic() - started,
        "network": "docker_none",
        "gpu_visibility": "disabled",
        "ok": all(receipt["ok"] for receipt in receipts),
        "chunk_count": len(chunks),
        "chunks_executed": len(receipts),
    }


def _run_container_replay(
    candidate: Mapping[str, Any],
    identity: Mapping[str, Any],
    patch_path: Path,
    test_patch_path: Path | None,
) -> dict[str, Any]:
    timeouts = {**DEFAULT_TIMEOUTS, **candidate.get("timeouts", {})}
    create = subprocess.run(
        _docker_create_args(
            identity,
            patch_path,
            test_patch_path,
            str(candidate.get("container_workdir") or identity.get("working_dir") or "/testbed"),
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if create.returncode:
        raise ReplayRejected("dependency_restore_failed", create.stderr[-2000:])
    container = create.stdout.strip()
    receipts: dict[str, Any] = {
        "container_image": dict(identity),
        "container_id": container,
        "network": "none",
        "gpu_devices_mounted": False,
        "gpu_visibility": {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
    }
    started = time.monotonic()
    try:
        subprocess.run(["docker", "start", container], check=True, capture_output=True, text=True, timeout=120)
        inspected = subprocess.run(
            ["docker", "inspect", container],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = json.loads(inspected.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise ReplayRejected("parent_not_reproducible", "invalid container inspection")
        _validate_container_policy(payload[0])
        receipts["policy_checked"] = True
        receipts["parent_head"] = _docker_exec(container, "git rev-parse HEAD", 120)
        _validate_container_head(receipts["parent_head"], str(candidate["base_commit"]))
        reset_command = candidate.get("container_reset_command")
        if reset_command:
            expected_reset = f"git reset --hard {candidate['base_commit']}"
            if reset_command != expected_reset:
                raise ReplayRejected(
                    "parent_not_reproducible",
                    "container reset command is not pinned to base_commit",
                )
            receipts["parent_reset"] = _docker_exec(container, expected_reset, 120)
            if not receipts["parent_reset"]["ok"]:
                raise ReplayRejected(
                    "parent_not_reproducible",
                    receipts["parent_reset"]["stderr"][-2000:],
                )
        clean_command = candidate.get("container_clean_command")
        if clean_command:
            if clean_command != "git clean -fd":
                raise ReplayRejected(
                    "parent_not_reproducible",
                    "container clean command is not the bounded untracked-file cleanup",
                )
            receipts["parent_clean"] = _docker_exec(container, clean_command, 120)
            if not receipts["parent_clean"]["ok"]:
                raise ReplayRejected(
                    "parent_not_reproducible",
                    receipts["parent_clean"]["stderr"][-2000:],
                )
        if test_patch_path is not None and test_patch_path.stat().st_size:
            receipts["test_patch_apply"] = _docker_exec(
                container, "git apply --whitespace=nowarn /replay/test.patch", 120
            )
            if not receipts["test_patch_apply"]["ok"]:
                raise ReplayRejected("tests_missing", receipts["test_patch_apply"]["stderr"][-2000:])
        receipts["install"] = _docker_exec(
            container, str(candidate["install_command"]), int(timeouts["checkout_dependency"])
        )
        if not receipts["install"]["ok"]:
            raise ReplayRejected("dependency_restore_failed", receipts["install"]["stderr"][-2000:])
        receipts["parent_compile"] = _docker_exec(
            container, str(candidate["compile_or_typecheck_command"]), int(timeouts["parent"])
        )
        if not receipts["parent_compile"]["ok"]:
            raise ReplayRejected("parent_not_reproducible", receipts["parent_compile"]["stderr"][-2000:])
        receipts["parent_tests"] = _docker_exec(
            container, str(candidate["targeted_test_command"]), int(timeouts["parent"])
        )
        if receipts["parent_tests"]["ok"] and candidate.get("parent_expectation") != "passes":
            raise ReplayRejected("parent_already_passes", "targeted tests passed on parent")
        # Build and test commands can update tracked generated files. Restore
        # the exact pinned parent, then reapply only the frozen held-out tests
        # before applying the target patch.
        receipts["parent_cleanup"] = _docker_exec(
            container,
            f"git reset --hard {shlex.quote(str(candidate['base_commit']))}",
            120,
        )
        if not receipts["parent_cleanup"]["ok"]:
            raise ReplayRejected(
                "parent_not_reproducible",
                receipts["parent_cleanup"]["stderr"][-2000:],
            )
        if clean_command:
            receipts["parent_cleanup_untracked"] = _docker_exec(
                container, clean_command, 120
            )
            if not receipts["parent_cleanup_untracked"]["ok"]:
                raise ReplayRejected(
                    "parent_not_reproducible",
                    receipts["parent_cleanup_untracked"]["stderr"][-2000:],
                )
        if test_patch_path is not None and test_patch_path.stat().st_size:
            receipts["test_patch_reapply"] = _docker_exec(
                container, "git apply --whitespace=nowarn /replay/test.patch", 120
            )
            if not receipts["test_patch_reapply"]["ok"]:
                raise ReplayRejected(
                    "tests_missing",
                    receipts["test_patch_reapply"]["stderr"][-2000:],
                )
        receipts["patch_apply"] = _docker_exec(
            container, "git apply --whitespace=nowarn /replay/target.patch", 120
        )
        if not receipts["patch_apply"]["ok"]:
            raise ReplayRejected("patch_apply_failed", receipts["patch_apply"]["stderr"][-2000:])
        changed = _docker_exec(container, "git diff --name-only HEAD", 120)
        changed_paths = sorted(line for line in changed["stdout"].splitlines() if line)
        patch_text = patch_path.read_text(encoding="utf-8")
        _check_changed_paths(_patch_paths(patch_text), list(candidate["allowed_paths"]), patch_text)
        receipts["target_compile"] = _docker_exec(
            container, str(candidate["compile_or_typecheck_command"]), int(timeouts["target"])
        )
        if not receipts["target_compile"]["ok"]:
            raise ReplayRejected("compile_or_typecheck_failed", receipts["target_compile"]["stderr"][-2000:])
        receipts["targeted_tests"] = _docker_exec(
            container, str(candidate["targeted_test_command"]), int(timeouts["target"])
        )
        if not receipts["targeted_tests"]["ok"]:
            raise ReplayRejected(
                "targeted_tests_failed",
                (
                    receipts["targeted_tests"]["stdout"]
                    + receipts["targeted_tests"]["stderr"]
                )[-4000:],
            )
        receipts["regression_tests"] = _docker_exec_pytest(
            container, str(candidate["full_regression_command"]), int(timeouts["regression"])
        )
        if not receipts["regression_tests"]["ok"]:
            raise ReplayRejected(
                "regression_tests_failed",
                (
                    receipts["regression_tests"]["stdout"]
                    + receipts["regression_tests"]["stderr"]
                )[-4000:],
            )
        diff = _docker_exec(container, "git diff --binary HEAD", 120)
        receipts["changed_paths"] = changed_paths
        receipts["source_hash"] = sha256_bytes(diff["stdout"].encode())
        receipts["ast_fingerprint"] = receipts["source_hash"]
        receipts["total_duration_seconds"] = time.monotonic() - started
        return receipts
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


def _verify_container_case(
    candidate: Mapping[str, Any],
    case_dir: Path,
    status: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    identity = _assert_docker_image_identity(candidate["container_image"])
    patch_path = _resolve_evidence(candidate, "target_patch_path", manifest_dir)
    test_patch_path = (
        _resolve_evidence(candidate, "test_patch_path", manifest_dir)
        if candidate.get("test_patch_path")
        else None
    )
    first = _run_container_replay(candidate, identity, patch_path, test_patch_path)
    write_json(case_dir / "target-receipt.json", first)
    _assert_docker_image_identity(candidate["container_image"])
    second = _run_container_replay(candidate, identity, patch_path, test_patch_path)
    write_json(case_dir / "fresh-receipt.json", second)
    comparable = ("source_hash", "ast_fingerprint", "changed_paths")
    if any(first[key] != second[key] for key in comparable):
        raise ReplayRejected("fresh_replay_mismatch", "fresh container output differs")
    row = dict(candidate)
    row.update(
        {
            "status": "verified",
            "source_hash": first["source_hash"],
            "normalized_patch_hash": sha256_bytes(
                _normalized_patch(patch_path.read_text(encoding="utf-8")).encode()
            ),
            "ast_fingerprint": first["ast_fingerprint"],
            "diff_hunk_hash": first["source_hash"],
            "test_set_hash": sha256_bytes(
                canonical_json(
                    {
                        "targeted": candidate["targeted_test_command"],
                        "regression": candidate["full_regression_command"],
                        "fail_to_pass": candidate.get("fail_to_pass", []),
                        "pass_to_pass": candidate.get("pass_to_pass", []),
                        "test_patch_sha256": candidate.get(
                            "test_patch_sha256", ""
                        ),
                    }
                ).encode()
            ),
            "target_test_receipt": "target-receipt.json",
            "fresh_verify_receipt": "fresh-receipt.json",
            "verified_runtime_seconds": (
                first["total_duration_seconds"] + second["total_duration_seconds"]
            ),
        }
    )
    write_json(case_dir / "verified.json", row)
    status.update({"status": "verified", "verified_record": "verified.json"})
    return status


def _candidate_verification_digest(candidate: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json(
            {"candidate": candidate, "verifier_version": VERIFIER_VERSION}
        ).encode()
    )


def _verify_case_unlocked(candidate: Mapping[str, Any], case_dir: Path) -> dict[str, Any]:
    digest = _candidate_verification_digest(candidate)
    status_path = case_dir / "status.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("candidate_sha256") == digest and prior.get("status") in {"verified", "rejected"}:
            return prior
    case_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "case_id": candidate.get("case_id"),
        "candidate_sha256": digest,
        "status": "verifying",
        "network": "disabled",
        "gpus": "disabled",
    }
    write_json(status_path, status)
    try:
        if candidate.get("network_policy") != "disabled" or candidate.get("gpu_required") is not False:
            raise ReplayRejected("gpu_required", "candidate does not declare disabled network and GPU")
        if candidate.get("benchmark_exclusion_gate") not in {None, "ready"}:
            raise ReplayRejected(
                "benchmark_overlap",
                f"benchmark exclusion gate is {candidate.get('benchmark_exclusion_gate')}",
            )
        manifest_dir = Path(str(candidate.get("_manifest_dir", "."))).resolve()
        if candidate.get("verification_backend") == "docker":
            return _verify_container_case(candidate, case_dir, status, manifest_dir)
        repo = _resolve_evidence(candidate, "local_repository", manifest_dir)
        is_bare = (
            repo.is_dir()
            and _git(repo, "rev-parse", "--is-bare-repository", check=False).stdout.strip() == "true"
        )
        if not repo.is_dir() or (not (repo / ".git").exists() and not is_bare):
            raise ReplayRejected("parent_revision_missing", "local_repository is not a local git repository")
        commit = str(candidate.get("base_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ReplayRejected("immutable_revision_required", "base_commit is not immutable")
        patch_path = _resolve_evidence(candidate, "target_patch_path", manifest_dir)
        if not patch_path.is_file():
            raise ReplayRejected("target_patch_missing", str(patch_path))
        patch_hash = sha256_bytes(_normalized_patch(patch_path.read_text(encoding="utf-8")).encode())
        workspaces = case_dir / "workspaces"
        if workspaces.exists():
            shutil.rmtree(workspaces)
        workspaces.mkdir()
        parent = workspaces / "parent"
        first_workspace = workspaces / "first-replay"
        fresh = workspaces / "candidate"
        _copy_git_revision(repo, commit, parent)
        _copy_git_revision(repo, commit, first_workspace)
        first = _run_replay(candidate, first_workspace, patch_path, fresh=False)
        write_json(case_dir / "target-receipt.json", first)
        _copy_git_revision(repo, commit, fresh)
        second = _run_replay(candidate, fresh, patch_path, fresh=True)
        write_json(case_dir / "fresh-receipt.json", second)
        comparable = ("source_hash", "ast_fingerprint", "changed_paths")
        if any(first[key] != second[key] for key in comparable):
            raise ReplayRejected("fresh_replay_mismatch", "fresh workspace output differs")
        row = dict(candidate)
        row.update(
            {
                "status": "verified",
                "source_hash": first["source_hash"],
                "normalized_patch_hash": patch_hash,
                "test_set_hash": sha256_bytes(
                    canonical_json(
                        {
                            "targeted": candidate["targeted_test_command"],
                            "regression": candidate["full_regression_command"],
                        }
                    ).encode()
                ),
                "ast_fingerprint": first["ast_fingerprint"],
                "diff_hunk_hash": patch_hash,
                "target_test_receipt": "target-receipt.json",
                "fresh_verify_receipt": "fresh-receipt.json",
                "verified_runtime_seconds": first["total_duration_seconds"] + second["total_duration_seconds"],
                "artifact_paths": {
                    "parent_workspace": str(parent.resolve()),
                    "candidate_workspace": str(fresh.resolve()),
                    "target_patch": str(patch_path.resolve()),
                    "problem_statement": str(
                        _resolve_evidence(candidate, "problem_statement_path", manifest_dir)
                    ),
                    "license_evidence": str(
                        _resolve_evidence(candidate, "license_evidence_path", manifest_dir)
                    ),
                    "target_receipt": str((case_dir / "target-receipt.json").resolve()),
                    "fresh_receipt": str((case_dir / "fresh-receipt.json").resolve()),
                },
            }
        )
        write_json(case_dir / "verified.json", row)
        status.update({"status": "verified", "verified_record": "verified.json"})
    except ReplayRejected as exc:
        status.update({"status": "rejected", "rejection_reason": exc.reason, "detail": exc.detail})
        write_json(case_dir / "rejection.json", status)
    finally:
        write_json(status_path, status)
    return status


def verify_case(candidate: Mapping[str, Any], case_dir: Path) -> dict[str, Any]:
    """Verify one case while serializing competing processes for that case."""
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / ".verify.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _verify_case_unlocked(candidate, case_dir)


def _completed_for_candidate(candidate: Mapping[str, Any], case_dir: Path) -> bool:
    status_path = case_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        status.get("candidate_sha256") == _candidate_verification_digest(candidate)
        and status.get("status") in {"verified", "rejected"}
    )


def _write_verification_snapshot(output_dir: Path) -> dict[str, Any]:
    statuses = []
    for path in sorted(output_dir.glob("*/status.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    rejected = [item for item in statuses if item.get("status") == "rejected"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "attempted": len(statuses),
        "verified": sum(item.get("status") == "verified" for item in statuses),
        "rejected": len(rejected),
        "verifying": sum(item.get("status") == "verifying" for item in statuses),
    }
    write_jsonl(output_dir.parent / "rejections.jsonl", rejected)
    write_json(output_dir / "verification-report.json", report)
    return report


def verify_candidates(
    candidates: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    limit: int | None = None,
    workers: int = 1,
    batch_id: str | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("verification workers must be positive")
    if limit is not None and limit < 1:
        raise ValueError("verification limit must be positive")
    rows = sorted(read_jsonl(candidates), key=lambda item: str(item.get("case_id", "")))
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_completed = 0
    if resume:
        pending = []
        for row in rows:
            if _completed_for_candidate(row, output_dir / str(row["case_id"])):
                skipped_completed += 1
            else:
                pending.append(row)
        rows = pending
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        case_dir = output_dir / str(row["case_id"])
        if not resume and (case_dir / "status.json").exists():
            raise RuntimeError(f"status exists; use --resume: {case_dir}")
    def run(row: Mapping[str, Any]) -> dict[str, Any]:
        return verify_case(row, output_dir / str(row["case_id"]))

    if workers == 1:
        statuses = [run(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            statuses = list(executor.map(run, rows))
    verified = sum(item["status"] == "verified" for item in statuses)
    rejected = [item for item in statuses if item["status"] == "rejected"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "attempted": len(rows),
        "verified": verified,
        "rejected": len(rejected),
        "skipped_completed": skipped_completed,
        "workers": workers,
    }
    safe_batch_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        batch_id or f"{time.time_ns()}-{os.getpid()}",
    ).strip(".-")
    if not safe_batch_id:
        raise ValueError("batch id must contain a letter, digit, underscore, dot, or hyphen")
    batch_dir = output_dir / "batch-reports"
    write_json(batch_dir / f"{safe_batch_id}.json", report)
    write_jsonl(batch_dir / f"{safe_batch_id}-rejections.jsonl", rejected)
    with (output_dir / ".report.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        report["snapshot"] = _write_verification_snapshot(output_dir)
    return report


def load_verified(verified_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(verified_dir.glob("*/verified.json"))
    ]


def deduplicate_verified(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, dict[str, str]] = defaultdict(dict)
    dimensions = {
        "repository_problem": lambda row: canonical_json(
            [row.get("upstream_repository"), row.get("problem_id")]
        ),
        "source_hash": lambda row: str(row.get("source_hash", "")),
        "normalized_patch_hash": lambda row: str(row.get("normalized_patch_hash", "")),
        "ast_fingerprint": lambda row: str(row.get("ast_fingerprint", "")),
        "diff_hunk_hash": lambda row: str(row.get("diff_hunk_hash", "")),
        "test_set_hash": lambda row: str(row.get("test_set_hash", "")),
    }
    for raw in sorted(rows, key=lambda row: str(row["case_id"])):
        row = dict(raw)
        collision = next(
            (
                (name, key, seen[name][key])
                for name, function in dimensions.items()
                for key in [function(row)]
                if key and key in seen[name]
            ),
            None,
        )
        if collision:
            name, _key, prior = collision
            reason = "duplicate_patch" if name in {"normalized_patch_hash", "ast_fingerprint", "diff_hunk_hash"} else "duplicate_source"
            row.update({"status": "rejected", "rejection_reason": reason, "duplicate_dimension": name, "duplicate_of": prior})
            rejected.append(row)
            continue
        unique.append(row)
        for name, function in dimensions.items():
            key = function(row)
            if key:
                seen[name][key] = str(row["case_id"])
    return unique, rejected


def _quota_document(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quota file must be a mapping")
    return value


def _selection_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    result = {"source": Counter(), "task_type": Counter(), "language": Counter()}
    for row in rows:
        result["source"][str(row["source_id"])] += 1
        result["task_type"][str(row["coding_task_type"])] += 1
        result["language"][str(row["primary_language"])] += 1
    return result


def select_replay(
    rows: Iterable[Mapping[str, Any]], quotas: Mapping[str, Any], target: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool, duplicates = deduplicate_verified(rows)
    targets = {
        "source": dict(quotas["source_targets"]),
        "task_type": dict(quotas["task_type_targets"]),
        "language": dict(quotas["language_targets"]),
    }
    for dimension, values in targets.items():
        if sum(values.values()) != target:
            raise ValueError(f"{dimension} quotas sum to {sum(values.values())}, not {target}")
    source_language_targets = {
        str(source): {str(language): int(value) for language, value in languages.items()}
        for source, languages in quotas.get("source_language_targets", {}).items()
    }
    for source, languages in source_language_targets.items():
        expected = targets["source"].get(source)
        if expected is None:
            raise ValueError(f"source-language quota references unknown source: {source}")
        if sum(languages.values()) != expected:
            raise ValueError(
                f"source-language quotas for {source} sum to "
                f"{sum(languages.values())}, not {expected}"
            )
    caps = quotas.get("caps", {})
    repository_cap = int(caps.get("repository", 15))
    lineage_cap = int(caps.get("lineage", 2))
    source_cap_fraction = float(caps.get("source_fraction", 0.60))

    ordered_pool = sorted(pool, key=lambda row: str(row["case_id"]))
    constraint_specs: list[tuple[list[int], int, int]] = []

    def add_exact(indices: list[int], value: int) -> None:
        constraint_specs.append((indices, value, value))

    for field, values in (
        ("source_id", targets["source"]),
        ("coding_task_type", targets["task_type"]),
        ("primary_language", targets["language"]),
    ):
        for value, expected in values.items():
            add_exact(
                [
                    index
                    for index, row in enumerate(ordered_pool)
                    if str(row.get(field)) == value
                ],
                int(expected),
            )
    for source, languages in source_language_targets.items():
        for language, expected in languages.items():
            add_exact(
                [
                    index
                    for index, row in enumerate(ordered_pool)
                    if str(row.get("source_id")) == source
                    and str(row.get("primary_language")) == language
                ],
                expected,
            )
    for field, cap in (
        ("upstream_repository", repository_cap),
        ("source_lineage_id", lineage_cap),
    ):
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(ordered_pool):
            grouped[str(row.get(field))].append(index)
        for indices in grouped.values():
            constraint_specs.append((indices, 0, cap))
    for source in targets["source"]:
        indices = [
            index
            for index, row in enumerate(ordered_pool)
            if str(row.get("source_id")) == source
        ]
        constraint_specs.append((indices, 0, int(target * source_cap_fraction)))

    solver_status = "not_run"
    exact_selection: list[dict[str, Any]] = []
    if ordered_pool:
        matrix = lil_matrix((len(constraint_specs), len(ordered_pool)), dtype=float)
        lower = np.empty(len(constraint_specs), dtype=float)
        upper = np.empty(len(constraint_specs), dtype=float)
        for row_index, (indices, minimum, maximum) in enumerate(constraint_specs):
            matrix[row_index, indices] = 1.0
            lower[row_index] = minimum
            upper[row_index] = maximum
        objective = np.array(
            [
                float(row.get("verified_runtime_seconds", 0))
                + (index + 1) / max(1, len(ordered_pool)) * 1e-6
                for index, row in enumerate(ordered_pool)
            ]
        )
        solution = milp(
            c=objective,
            integrality=np.ones(len(ordered_pool)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": 300},
        )
        solver_status = str(solution.message)
        if solution.success and solution.x is not None:
            exact_selection = [
                dict(row)
                for row, chosen in zip(ordered_pool, solution.x, strict=True)
                if chosen >= 0.5
            ]
    if len(exact_selection) == target:
        counts = _selection_counts(exact_selection)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "exact",
            "selection_method": "scipy_milp_highs",
            "solver_status": solver_status,
            "target": target,
            "selected": len(exact_selection),
            "deduplicated_pool": len(pool),
            "duplicate_rejections": len(duplicates),
            "deficits": {"source": {}, "task_type": {}, "language": {}},
            "counts": {
                dimension: dict(sorted(value.items()))
                for dimension, value in counts.items()
            },
            "candidate_cross_distribution": dict(
                sorted(
                    Counter(
                        f"{row.get('source_id')}|{row.get('coding_task_type')}|"
                        f"{row.get('primary_language')}"
                        for row in pool
                    ).items()
                )
            ),
        }
        return exact_selection, report

    selected: list[dict[str, Any]] = []
    counts = _selection_counts([])
    repositories: Counter[str] = Counter()
    lineages: Counter[str] = Counter()
    remaining = sorted(pool, key=lambda row: (float(row.get("verified_runtime_seconds", 0)), str(row["case_id"])))
    while len(selected) < target:
        eligible = []
        for row in remaining:
            axes = {
                "source": str(row.get("source_id")),
                "task_type": str(row.get("coding_task_type")),
                "language": str(row.get("primary_language")),
            }
            if any(axes[dim] not in targets[dim] or counts[dim][axes[dim]] >= targets[dim][axes[dim]] for dim in targets):
                continue
            repository = str(row.get("upstream_repository"))
            lineage = str(row.get("source_lineage_id"))
            if repositories[repository] >= repository_cap or lineages[lineage] >= lineage_cap:
                continue
            if counts["source"][axes["source"]] >= int(target * source_cap_fraction):
                continue
            deficits = {
                dim: targets[dim][axes[dim]] - counts[dim][axes[dim]]
                for dim in targets
            }
            scarcity = tuple(
                sum(
                    1
                    for candidate in remaining
                    if str(candidate.get({"source": "source_id", "task_type": "coding_task_type", "language": "primary_language"}[dim]))
                    == axes[dim]
                )
                for dim in ("language", "task_type", "source")
            )
            eligible.append(
                (
                    tuple(-deficits[dim] for dim in ("language", "task_type", "source")),
                    scarcity,
                    repositories[repository],
                    lineages[lineage],
                    float(row.get("verified_runtime_seconds", 0)),
                    str(row["case_id"]),
                    row,
                )
            )
        if not eligible:
            break
        chosen = min(eligible)[-1]
        selected.append(dict(chosen))
        remaining.remove(chosen)
        repositories[str(chosen["upstream_repository"])] += 1
        lineages[str(chosen["source_lineage_id"])] += 1
        counts["source"][str(chosen["source_id"])] += 1
        counts["task_type"][str(chosen["coding_task_type"])] += 1
        counts["language"][str(chosen["primary_language"])] += 1
    deficits = {
        dimension: {
            key: target_value - counts[dimension][key]
            for key, target_value in values.items()
            if target_value - counts[dimension][key]
        }
        for dimension, values in targets.items()
    }
    exact = len(selected) == target and not any(deficits.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact" if exact else "quota_deficits",
        "selection_method": "greedy_diagnostic_fallback",
        "solver_status": solver_status,
        "target": target,
        "selected": len(selected),
        "deduplicated_pool": len(pool),
        "duplicate_rejections": len(duplicates),
        "deficits": deficits,
        "counts": {dimension: dict(sorted(value.items())) for dimension, value in counts.items()},
        "candidate_cross_distribution": dict(
            sorted(
                Counter(
                    f"{row.get('source_id')}|{row.get('coding_task_type')}|{row.get('primary_language')}"
                    for row in pool
                ).items()
            )
        ),
    }
    return selected if exact else [], report


def select_verified(verified_dir: Path, quota_file: Path, target: int, output: Path) -> dict[str, Any]:
    rows = load_verified(verified_dir)
    selected, report = select_replay(rows, _quota_document(quota_file), target)
    write_jsonl(output, selected)
    deficit_path = output.with_name("quota-deficits.json")
    if report["status"] == "exact":
        if deficit_path.exists():
            deficit_path.unlink()
    else:
        write_json(deficit_path, report)
    write_json(output.with_name("selection-report.json"), report)
    return report


def _copy_artifact(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise RuntimeError(f"symlink artifact forbidden: {source}")
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise RuntimeError(f"missing artifact: {source}")


def _scan_secrets(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if SECRET_NAME.search(relative):
            hits.append(relative)
            continue
        if path.stat().st_size <= 2_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if SECRET_TEXT.search(text):
                hits.append(relative)
    return hits


def _absolute_local_paths(value: Any, location: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            hits.extend(_absolute_local_paths(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_absolute_local_paths(item, f"{location}[{index}]"))
    elif isinstance(value, str) and (value.startswith("/") or "/home/" in value):
        hits.append(location)
    return hits


def package_replay(accepted: Path, hf_root: Path, *, expected_count: int = 500) -> dict[str, Any]:
    rows = read_jsonl(accepted)
    if len(rows) != expected_count:
        raise RuntimeError(f"accepted must contain exactly {expected_count} rows")
    hf_root.mkdir(parents=True, exist_ok=True)
    if any(hf_root.iterdir()):
        raise RuntimeError(f"HF root must be empty: {hf_root}")
    published: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in {"verified", "accepted"}:
            raise RuntimeError(f"{row.get('case_id')}: not verified")
        case_id = str(row["case_id"])
        artifact_root = hf_root / "artifacts" / case_id
        relative_artifacts: dict[str, str] = {}
        for name, raw in sorted(dict(row.get("artifact_paths", {})).items()):
            source = Path(str(raw)).resolve()
            destination = artifact_root / name
            _copy_artifact(source, destination)
            relative_artifacts[name] = destination.relative_to(hf_root).as_posix()
        item = {key: value for key, value in row.items() if key not in {"_manifest_dir"}}
        item.pop("local_repository", None)
        item["status"] = "accepted"
        item["artifact_paths"] = relative_artifacts
        if "target_patch" in relative_artifacts:
            item["target_patch_path"] = relative_artifacts["target_patch"]
        if "problem_statement" in relative_artifacts:
            item["problem_statement_path"] = relative_artifacts["problem_statement"]
        if "license_evidence" in relative_artifacts:
            item["license_evidence_path"] = relative_artifacts["license_evidence"]
        absolute_hits = _absolute_local_paths(item)
        if absolute_hits:
            raise RuntimeError(f"{case_id}: absolute local paths remain at {absolute_hits[:10]}")
        published.append(item)
    write_jsonl(hf_root / "samples.jsonl", published)
    counts = _selection_counts(published)
    write_json(
        hf_root / "coverage-report.json",
        {
            "sample_count": len(published),
            **{key: dict(sorted(value.items())) for key, value in counts.items()},
            "repository": dict(sorted(Counter(row["upstream_repository"] for row in published).items())),
        },
    )
    write_json(hf_root / "leakage-report.json", {"status": "pass", "overlap_count": 0})
    write_json(
        hf_root / "quality-report.json",
        {"status": "pass", "sample_count": len(published), "fresh_verifies_passed": len(published)},
    )
    general_tokens = sum(int(row.get("assistant_loss_tokens", 0)) for row in published)
    kernel_values = {int(row["kernel_assistant_loss_tokens"]) for row in published if "kernel_assistant_loss_tokens" in row}
    if not general_tokens or len(kernel_values) != 1:
        raise RuntimeError("token mix requires assistant_loss_tokens and one frozen kernel_assistant_loss_tokens value")
    kernel_tokens = next(iter(kernel_values))
    share = general_tokens / (general_tokens + kernel_tokens)
    if not 0.15 <= share <= 0.20:
        raise RuntimeError(f"General Coding assistant loss token share {share:.6f} is outside 15%-20%")
    write_json(
        hf_root / "token-mix-report.json",
        {
            "status": "pass",
            "general_coding_assistant_loss_tokens": general_tokens,
            "kernel_assistant_loss_tokens": kernel_tokens,
            "share": share,
        },
    )
    secret_hits = _scan_secrets(hf_root)
    if secret_hits:
        raise ReplayRejected("secret_detected", ", ".join(secret_hits[:20]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(published),
        "samples": {"path": "samples.jsonl", "sha256": sha256_file(hf_root / "samples.jsonl")},
        "relative_paths_only": True,
    }
    write_json(hf_root / "manifest.json", manifest)
    checksum_path = hf_root / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(hf_root.rglob("*")):
            if path.is_file() and path != checksum_path:
                handle.write(f"{sha256_file(path)}  {path.relative_to(hf_root).as_posix()}\n")
    return manifest


def validate_frozen_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict[str, Any]:
    """Reject a cache entry unless its bytes match the frozen identity."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{path}: expected {expected_size} bytes, found {size}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{path}: expected sha256 {expected_sha256}, found {actual}")
    return {"path": str(path), "bytes": size, "sha256": actual}


def audit_opencode_schema(fields: Sequence[str], row_count: int = OPENCODE_ROWS) -> dict[str, Any]:
    """Apply the strict replay gate without treating answer-derived tests as evidence."""
    available = set(fields)
    requirements = {
        "recoverable_parent_revision": {
            "required_any": [["upstream_repository", "base_commit"], ["repository", "parent_revision"]],
            "satisfied": False,
        },
        "target_diff": {
            "required_any": [["patch"], ["target_patch"], ["diff"]],
            "satisfied": bool({"patch", "target_patch", "diff"}.intersection(available)),
        },
        "clear_license": {
            "required_any": [["license_spdx"], ["repository_license"]],
            "satisfied": bool({"license_spdx", "repository_license"}.intersection(available)),
        },
        "executable_tests": {
            "required_any": [["unit_tests", "tests_execution_status"]],
            "satisfied": {"unit_tests", "tests_execution_status"}.issubset(available),
            "caveat": "tests alone cannot recover a repository parent or establish an independent target diff",
        },
    }
    requirements["recoverable_parent_revision"]["satisfied"] = (
        {"upstream_repository", "base_commit"}.issubset(available)
        or {"repository", "parent_revision"}.issubset(available)
    )
    strict = all(bool(item["satisfied"]) for item in requirements.values())
    missing_columns = sorted(
        {
            "upstream_repository",
            "base_commit",
            "target_patch",
            "repository_license",
        }
        - available
    )
    return {
        "schema_version": "opencode_strict_admissibility_v1",
        "dataset_id": OPENCODE_DATASET_ID,
        "dataset_revision": OPENCODE_REVISION,
        "dataset_rows": row_count,
        "rows_scanned": 0,
        "schema_fields": list(fields),
        "requirements": requirements,
        "strict_gate_passed": strict,
        "admissible_rows": 0 if not strict else None,
        "rejection_scope_rows": row_count if not strict else None,
        "rejection_reason": "schema_missing_replay_provenance" if not strict else None,
        "missing_columns": missing_columns,
        "blocker": (
            "The dataset schema has generated input/output and unit-test strings but no recoverable "
            "repository parent revision or target diff. Tests must not be derived from target answers."
            if not strict
            else None
        ),
    }


def _download_frozen(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict[str, Any]:
    if destination.is_file():
        actual = sha256_file(destination)
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(f"cached source checksum mismatch: {destination}")
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RuntimeError(f"cached source size mismatch: {destination}")
        return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": actual}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 300), headers={"User-Agent": "general-coding-replay-freezer/1"}) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
            temporary = Path(handle.name)
    actual = sha256_file(temporary)
    size = temporary.stat().st_size
    if expected_sha256 is not None and actual != expected_sha256:
        temporary.unlink()
        raise RuntimeError(f"downloaded source checksum mismatch for {url}: {actual}")
    if expected_size is not None and size != expected_size:
        temporary.unlink()
        raise RuntimeError(f"downloaded source size mismatch for {url}: {size}")
    os.replace(temporary, destination)
    return {"path": str(destination), "bytes": size, "sha256": actual}


def _fetch_json_frozen(url: str, destination: Path) -> dict[str, Any]:
    if destination.is_file():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"cached API response is not an object: {destination}")
        return value
    response = requests.get(url, timeout=(30, 120), headers={"User-Agent": "general-coding-replay-freezer/1"})
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"API response is not an object: {url}")
    write_json(destination, value)
    return value


def _eligible_swe_row(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    required_text = ("instance_id", "repo", "base_commit", "patch", "problem_statement")
    missing = [field for field in required_text if not str(row.get(field, "")).strip()]
    if missing:
        return False, "missing_" + "_".join(missing)
    if not re.fullmatch(r"[0-9a-f]{40}", str(row["base_commit"])):
        return False, "base_commit_not_immutable"
    pass_to_pass = row.get("PASS_TO_PASS")
    fail_to_pass = row.get("FAIL_TO_PASS")
    if not str(row.get("test_patch", "")).strip() and not pass_to_pass and not fail_to_pass:
        return False, "test_metadata_missing"
    if "/" not in str(row["repo"]):
        return False, "upstream_repository_invalid"
    return True, None


def _license_evidence(
    repo: str,
    base_commit: str,
    evidence_dir: Path,
) -> dict[str, Any]:
    safe_name = repo.replace("/", "__")
    api_path = evidence_dir / f"{safe_name}.github-license-api.json"
    url = f"https://api.github.com/repos/{repo}/license?ref={base_commit}"
    try:
        payload = _fetch_json_frozen(url, api_path)
    except requests.RequestException as exc:
        return {"repository": repo, "status": "rejected", "reason": "license_unknown", "detail": str(exc)}
    license_value = payload.get("license") if isinstance(payload.get("license"), Mapping) else {}
    spdx = str(license_value.get("spdx_id", "NOASSERTION"))
    encoded = str(payload.get("content", "")).replace("\n", "")
    if not encoded:
        return {"repository": repo, "status": "rejected", "reason": "license_unknown", "spdx": spdx}
    try:
        license_bytes = base64.b64decode(encoded, validate=True)
    except ValueError:
        return {"repository": repo, "status": "rejected", "reason": "license_unknown", "detail": "invalid API content"}
    if spdx in {"", "NOASSERTION", "null"}:
        normalized = re.sub(r"\s+", " ", license_bytes.decode("utf-8", errors="ignore")).casefold()
        if "apache license" in normalized and "version 2.0" in normalized:
            spdx = "Apache-2.0"
        elif "permission is hereby granted, free of charge" in normalized:
            spdx = "MIT"
        elif "redistribution and use in source and binary forms" in normalized:
            spdx = "BSD-3-Clause"
        else:
            return {
                "repository": repo,
                "status": "rejected",
                "reason": "license_unknown",
                "spdx": spdx,
                "license_text_sha256": sha256_bytes(license_bytes),
            }
    text_path = evidence_dir / f"{safe_name}.LICENSE"
    if text_path.is_file() and text_path.read_bytes() != license_bytes:
        raise RuntimeError(f"cached license evidence changed: {text_path}")
    if not text_path.is_file():
        atomic_write(text_path, license_bytes)
    status = "accepted" if spdx in PERMISSIVE_SOURCE_LICENSES else "rejected"
    return {
        "repository": repo,
        "status": status,
        "reason": None if status == "accepted" else "license_incompatible",
        "spdx": spdx,
        "base_commit": base_commit,
        "api_url": url,
        "license_blob_sha": payload.get("sha"),
        "api_evidence_path": api_path.name,
        "license_text_path": text_path.name,
        "license_text_sha256": sha256_file(text_path),
    }


_TASK_TITLE_RULES = (
    (
        "non_kernel_performance_repair",
        re.compile(
            r"(?i)\b(performance|perf|optimi[sz](?:e|ed|es|ing|ation)?|"
            r"speed(?:up)?|slow|latency|"
            r"throughput|allocations?|memory leak|quadratic|hot path)\b"
        ),
    ),
    (
        "refactor_or_api_adaptation",
        re.compile(
            r"(?i)(?:^|\b)(refactor(?:ed|ing)?|renam(?:e|ed|ing)|"
            r"deprecat(?:e|ed|ing|ion)|migrat(?:e|ed|ing|ion)|"
            r"adapt(?:ed|ing|ation)?|api (?:change|adaptation)|"
            r"compatib(?:le|ility)?|breaking change|cleanup|clean up|rework|"
            r"reorganiz(?:e|ed|ing|ation)|restructur(?:e|ed|ing)|"
            r"simplif(?:y|ied|ication)|extract .+ (?:into|to)|move .+ (?:into|to)|"
            r"convert .+ (?:into|to)|replace .+ with|"
            r"switch .+ (?:from .+ )?to)\b"
        ),
    ),
    (
        "function_implementation",
        re.compile(
            r"(?i)(?:^|\b)(implement(?:ed|ing|ation)?|"
            r"add(?:ing|ed)?(?: support)?|support for|"
            r"introduce|enable|expose|allow|provide|"
            r"new (?:api|method|function|feature|command)|"
            r"feature(?: request)?|create)\b"
        ),
    ),
    (
        "build_dependency_or_config",
        re.compile(
            r"(?i)\b(build|cmake|meson|bazel|dependency|dependencies|package[- ]lock|"
            r"ci|workflow|toolchain|packag(?:e|ing)|compile)\b"
        ),
    ),
    (
        "typing_validation_or_docs_with_test",
        re.compile(r"(?i)\b(type(?:script|check| hint)?|typing|validation|validator|docs?|documentation)\b"),
    ),
)


def derive_repository_task_type(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify from frozen issue and patch evidence; never assign random labels."""
    title = str(row.get("title") or str(row.get("problem_statement", "")).splitlines()[0])
    body = str(row.get("body") or row.get("problem_statement") or "")
    patch = str(row.get("fix_patch") or row.get("patch") or row.get("target_patch") or "")
    changed_paths = []
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            value = line.split(" b/", 1)[0][len("diff --git a/") :]
            if value and value not in changed_paths:
                changed_paths.append(value)
    # Issue bodies routinely mention incidental "build", "config", "docs", or
    # "type" context.  Those words previously overrode unrelated bug titles.
    # Use the title as the intent-bearing evidence and reserve patch paths for
    # exclusive docs/build changes.
    title_evidence = title or body.splitlines()[0] if body else title
    for task_type, pattern in _TASK_TITLE_RULES:
        match = pattern.search(title_evidence)
        if match:
            return task_type, {
                "method": "title_intent_and_patch_structure_v2",
                "matched_rule": task_type,
                "matched_text": match.group(0),
                "evidence_scope": "title",
                "title_sha256": sha256_bytes(title.encode()),
                "changed_paths": changed_paths,
            }
    non_test_paths = [
        path
        for path in changed_paths
        if not re.search(
            r"(?i)(^|/)(tests?|testdata|fixtures?)(/|$)|(^|/)test_", path
        )
    ]
    if non_test_paths and all(
        re.search(r"(?i)(^|/)(docs?|documentation)(/|$)|\.(md|rst)$", path)
        for path in non_test_paths
    ):
        return "typing_validation_or_docs_with_test", {
            "method": "title_intent_and_patch_structure_v2",
            "matched_rule": "exclusive_documentation_paths",
            "matched_text": None,
            "evidence_scope": "changed_paths",
            "title_sha256": sha256_bytes(title.encode()),
            "changed_paths": changed_paths,
        }
    if non_test_paths and all(
        re.search(
            r"(?i)(^|/)(\.github|ci|cmake|build|config)(/|$)|"
            r"(^|/)(CMakeLists\.txt|Dockerfile|Makefile|meson\.build|"
            r"package-lock\.json|go\.(mod|sum)|Cargo\.(toml|lock))$",
            path,
        )
        for path in non_test_paths
    ):
        return "build_dependency_or_config", {
            "method": "title_intent_and_patch_structure_v2",
            "matched_rule": "exclusive_build_or_config_paths",
            "matched_text": None,
            "evidence_scope": "changed_paths",
            "title_sha256": sha256_bytes(title.encode()),
            "changed_paths": changed_paths,
        }
    if "rename from " in patch and "rename to " in patch:
        return "refactor_or_api_adaptation", {
            "method": "title_intent_and_patch_structure_v2",
            "matched_rule": "git_rename_metadata",
            "matched_text": None,
            "evidence_scope": "patch",
            "title_sha256": sha256_bytes(title.encode()),
            "changed_paths": changed_paths,
        }
    test_evidence = bool(
        str(row.get("test_patch", "")).strip()
        or row.get("f2p_tests")
        or row.get("fixed_tests")
        or row.get("FAIL_TO_PASS")
        or row.get("fail_to_pass")
    )
    return "repository_bug_fix_or_test_repair", {
        "method": "title_intent_and_patch_structure_v2",
        "matched_rule": "test_backed_default",
        "test_evidence": test_evidence,
        "title_sha256": sha256_bytes(title.encode()),
        "changed_paths": changed_paths,
    }


def _fetch_dataset_filter_page(
    dataset_id: str,
    language: str,
    offset: int,
    length: int,
    destination: Path,
) -> dict[str, Any]:
    if destination.is_file():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"cached dataset filter response is invalid: {destination}")
        return value
    params = {
        "dataset": dataset_id,
        "config": "default",
        "split": "train",
        "where": f'"lang"=\'{language}\'',
        "orderby": '"instance_id"',
        "offset": offset,
        "length": length,
    }
    last_error = ""
    for attempt in range(6):
        response = requests.get(
            "https://datasets-server.huggingface.co/filter",
            params=params,
            timeout=(30, 300),
            headers={"User-Agent": "general-coding-replay-freezer/2"},
        )
        if response.status_code == 200:
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("dataset filter response is not an object")
            write_json(destination, value)
            return value
        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code not in {500, 502, 503, 504}:
            break
        time.sleep(2**attempt)
    raise RuntimeError(f"selective dataset filter failed for {language}: {last_error}")


def _freeze_multiswe_image_identities(
    candidates: Sequence[dict[str, Any]],
    cache_path: Path,
) -> dict[str, dict[str, str]]:
    cached = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.is_file()
        else {}
    )
    tokens: dict[str, str] = {}
    identities: dict[str, dict[str, str]] = {
        str(key): dict(value) for key, value in cached.items()
    }
    for candidate in candidates:
        case_id = str(candidate["case_id"])
        if case_id in identities:
            candidate["container_image"] = identities[case_id]
            continue
        repository = _repository_slug(str(candidate["upstream_repository"]))
        org, repo = repository.split("/", 1)
        number = int(str(candidate["problem_id"]).rsplit("-", 1)[1])
        image_repository = f"mswebench/{org.lower()}_m_{repo.lower()}"
        if image_repository not in tokens:
            token_response = requests.get(
                "https://auth.docker.io/token",
                params={
                    "service": "registry.docker.io",
                    "scope": f"repository:{image_repository}:pull",
                },
                timeout=(30, 120),
            )
            token_response.raise_for_status()
            tokens[image_repository] = str(token_response.json()["token"])
        tag = f"pr-{number}"
        manifest = requests.head(
            f"https://registry-1.docker.io/v2/{image_repository}/manifests/{tag}",
            headers={
                "Authorization": f"Bearer {tokens[image_repository]}",
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
            raise RuntimeError(f"{case_id}: container registry did not return an immutable digest")
        identity = {
            "repo_tag": f"{image_repository}:{tag}",
            "repo_digest": f"{image_repository}@{digest}",
            "manifest_digest": digest,
        }
        identities[case_id] = identity
        candidate["container_image"] = identity
    write_json(cache_path, identities)
    return identities


def freeze_multiswe_verified_source(control_root: Path) -> dict[str, Any]:
    """Freeze 250 API-selected rows without downloading the 3.56 GB corpus."""
    cache = control_root / "source-cache" / "multi-swe-rl-verified" / MULTISWE_VERIFIED_REVISION
    api = _fetch_json_frozen(
        f"https://huggingface.co/api/datasets/{MULTISWE_VERIFIED_DATASET_ID}"
        f"/revision/{MULTISWE_VERIFIED_REVISION}",
        cache / "dataset-api.json",
    )
    if api.get("sha") != MULTISWE_VERIFIED_REVISION:
        raise RuntimeError("Multi-SWE-RL-Verified did not resolve to the frozen revision")
    readme = _download_frozen(
        f"https://huggingface.co/datasets/{MULTISWE_VERIFIED_DATASET_ID}/resolve/"
        f"{MULTISWE_VERIFIED_REVISION}/README.md",
        cache / "README.md",
    )
    validation_files = {}
    for name in ("validation.jsonl", "validation-pass2.jsonl"):
        validation_files[name] = _download_frozen(
            f"https://huggingface.co/datasets/{MULTISWE_VERIFIED_DATASET_ID}/resolve/"
            f"{MULTISWE_VERIFIED_REVISION}/{name}",
            cache / name,
        )

    raw_rows = []
    response_receipts = []
    for language, target in MULTISWE_TRAIN_FETCH_TARGETS.items():
        fetch_count = target if target == 100 else min(100, target + 25)
        payload = _fetch_dataset_filter_page(
            MULTISWE_VERIFIED_DATASET_ID,
            language,
            0,
            fetch_count,
            cache / f"filter-{language}-00000-{fetch_count:05d}.json",
        )
        rows = [item["row"] for item in payload.get("rows", [])]
        if len(rows) != fetch_count or any(str(row.get("lang")) != language for row in rows):
            raise RuntimeError(
                f"selective {language} response did not contain exactly {fetch_count} rows"
            )
        raw_rows.extend(rows)
        response_path = cache / f"filter-{language}-00000-{fetch_count:05d}.json"
        response_receipts.append(
            {
                "language": language,
                "rows": len(rows),
                "path": response_path.relative_to(control_root).as_posix(),
                "sha256": sha256_file(response_path),
            }
        )

    candidates = []
    excluded_held_out_repositories = Counter()
    accepted_by_source_language = Counter()
    for row in sorted(raw_rows, key=lambda item: str(item["instance_id"])):
        source_language = str(row["lang"])
        if accepted_by_source_language[source_language] >= MULTISWE_TRAIN_FETCH_TARGETS[source_language]:
            continue
        repo = f"{row['org']}/{row['repo']}"
        if repo in MULTISWE_HELD_OUT_REPOSITORIES:
            excluded_held_out_repositories[repo] += 1
            continue
        base = row.get("base") if isinstance(row.get("base"), Mapping) else {}
        base_commit = str(base.get("sha", ""))
        ok, reason = _eligible_swe_row(
            {
                "instance_id": row.get("instance_id"),
                "repo": repo,
                "base_commit": base_commit,
                "patch": row.get("fix_patch"),
                "problem_statement": "\n\n".join(
                    value for value in (str(row.get("title", "")), str(row.get("body", ""))) if value
                ),
                "test_patch": row.get("test_patch"),
                "PASS_TO_PASS": list((row.get("p2p_tests") or {}).keys())
                if isinstance(row.get("p2p_tests"), Mapping)
                else [],
                "FAIL_TO_PASS": list((row.get("f2p_tests") or {}).keys())
                if isinstance(row.get("f2p_tests"), Mapping)
                else [],
            }
        )
        if not ok:
            raise RuntimeError(f"{row.get('instance_id')}: verified source schema rejected: {reason}")
        if (
            row.get("validation_reward_p1") != 1.0
            or row.get("validation_reward_p2") != 1.0
            or row.get("validation_reason_p1") != "pass"
            or row.get("validation_reason_p2") != "pass"
        ):
            raise RuntimeError(f"{row.get('instance_id')}: validation evidence is not two-pass clean")
        task_type, classification = derive_repository_task_type(row)
        source_record_hash = sha256_bytes(canonical_json(row).encode())
        language = "javascript_typescript" if row["lang"] in {"js", "ts"} else str(row["lang"])
        candidates.append(
            {
                "schema_version": "general_coding_replay_candidate_source_v2",
                "case_id": f"gc-replay-multi_swe_rl_verified-{row['instance_id']}",
                "source_id": "multi_swe_rl_verified",
                "dataset_id": MULTISWE_VERIFIED_DATASET_ID,
                "dataset_revision": MULTISWE_VERIFIED_REVISION,
                "dataset_row_id": row["instance_id"],
                "source_record_sha256": source_record_hash,
                "source_lineage_id": (
                    f"{MULTISWE_VERIFIED_DATASET_ID}@{MULTISWE_VERIFIED_REVISION}:"
                    f"{row['instance_id']}"
                ),
                "upstream_repository": f"https://github.com/{repo}.git",
                "base_commit": base_commit,
                "problem_id": row["instance_id"],
                "problem_statement": "\n\n".join(
                    value for value in (str(row.get("title", "")), str(row.get("body", ""))) if value
                ),
                "target_patch": row["fix_patch"],
                "target_patch_sha256": sha256_bytes(str(row["fix_patch"]).encode()),
                "test_patch": row["test_patch"],
                "test_patch_sha256": sha256_bytes(str(row["test_patch"]).encode()),
                "test_results_sha256": sha256_bytes(
                    canonical_json(
                        {
                            "run_result": row.get("run_result"),
                            "test_patch_result": row.get("test_patch_result"),
                            "fix_patch_result": row.get("fix_patch_result"),
                        }
                    ).encode()
                ),
                "validation_evidence": {
                    "pass_1": {
                        "reward": row["validation_reward_p1"],
                        "reason": row["validation_reason_p1"],
                        "elapsed_seconds": row["validation_elapsed_s_p1"],
                    },
                    "pass_2": {
                        "reward": row["validation_reward_p2"],
                        "reason": row["validation_reason_p2"],
                        "elapsed_seconds": row["validation_elapsed_s_p2"],
                    },
                },
                "dataset_license": "CC0 with ByteDance notice; repository licenses control code",
                "upstream_adapter_repository": MULTISWE_UPSTREAM_REPOSITORY,
                "upstream_adapter_revision": MULTISWE_UPSTREAM_REVISION,
                "upstream_adapter_license": MULTISWE_UPSTREAM_LICENSE,
                "primary_language": language,
                "primary_task_type": task_type,
                "task_type_evidence": classification,
                "status": "candidate_gold_patch_validated_upstream_not_locally_verified",
                "verified": False,
                "tests_executed_locally": False,
            }
        )
        accepted_by_source_language[source_language] += 1
    if accepted_by_source_language != Counter(MULTISWE_TRAIN_FETCH_TARGETS):
        raise RuntimeError(
            f"selective source could not replace held-out repositories: "
            f"{dict(accepted_by_source_language)}"
        )
    license_dir = control_root / "license-evidence" / "multi-swe-upstream"
    license_dir.mkdir(parents=True, exist_ok=True)
    representative_commits = {}
    for candidate in candidates:
        repo = _repository_slug(str(candidate["upstream_repository"]))
        commit = str(candidate["base_commit"])
        representative_commits[repo] = min(commit, representative_commits.get(repo, commit))
    repository_licenses = {
        repo: _license_evidence(repo, commit, license_dir)
        for repo, commit in sorted(representative_commits.items())
    }
    adapter_license = _license_evidence(
        "multi-swe-bench/multi-swe-bench",
        MULTISWE_UPSTREAM_REVISION,
        license_dir,
    )
    for evidence in [*repository_licenses.values(), adapter_license]:
        if evidence["status"] != "accepted":
            raise RuntimeError(
                f"frozen upstream license rejected for {evidence['repository']}: "
                f"{evidence.get('reason')}"
            )
    for candidate in candidates:
        repo = _repository_slug(str(candidate["upstream_repository"]))
        evidence = repository_licenses[repo]
        candidate["license_spdx"] = evidence["spdx"]
        candidate["license_evidence_path"] = (
            Path("license-evidence")
            / "multi-swe-upstream"
            / str(evidence["license_text_path"])
        ).as_posix()
    image_identities = _freeze_multiswe_image_identities(
        candidates, cache / "container-image-identities.json"
    )
    candidate_path = control_root / "multi-swe-verified-candidate-source-manifest.jsonl"
    write_jsonl(candidate_path, candidates)
    write_json(
        control_root / "multi-swe-upstream-license-report.json",
        {
            "repositories": [repository_licenses[key] for key in sorted(repository_licenses)],
            "adapter_repository": adapter_license,
        },
    )
    counts = Counter(row["primary_language"] for row in candidates)
    task_counts = Counter(row["primary_task_type"] for row in candidates)
    source_shortfall = 250 - min(
        250,
        min(PHASE1_LANGUAGE_TARGETS["go"], counts["go"])
        + min(
            PHASE1_LANGUAGE_TARGETS["javascript_typescript"],
            counts["javascript_typescript"],
        ),
    )
    report = {
        "schema_version": "general_coding_replay_multiswe_source_freeze_v1",
        "dataset_id": MULTISWE_VERIFIED_DATASET_ID,
        "dataset_revision": MULTISWE_VERIFIED_REVISION,
        "dataset_rows": MULTISWE_VERIFIED_ROWS,
        "dataset_license": "CC0 with ByteDance notice; upstream repository licenses remain controlling",
        "upstream_adapter_repository": {
            "url": MULTISWE_UPSTREAM_REPOSITORY,
            "revision": MULTISWE_UPSTREAM_REVISION,
            "license": MULTISWE_UPSTREAM_LICENSE,
            "license_evidence_sha256": sha256_file(
                license_dir / str(adapter_license["license_text_path"])
            ),
        },
        "repository_license_count": len(repository_licenses),
        "repository_licenses": dict(
            sorted((repo, evidence["spdx"]) for repo, evidence in repository_licenses.items())
        ),
        "container_images": {
            "digest_frozen": len(image_identities),
            "downloaded": 0,
            "identity_manifest_sha256": sha256_file(
                cache / "container-image-identities.json"
            ),
        },
        "retrieval": {
            "method": "datasets-server selective filter API",
            "full_corpus_downloaded": False,
            "parquet_shards_downloaded": 0,
            "responses": response_receipts,
        },
        "validation_evidence": validation_files,
        "readme": readme,
        "requested_rows": sum(MULTISWE_TRAIN_FETCH_TARGETS.values()),
        "candidate_rows": len(candidates),
        "excluded_for_repository_held_out": dict(sorted(excluded_held_out_repositories.items())),
        "candidate_language_counts": dict(sorted(counts.items())),
        "candidate_task_type_counts": dict(sorted(task_counts.items())),
        "authoritative_validation_language_counts": MULTISWE_VALIDATION_LANGUAGE_COUNTS,
        "quota_feasibility": {
            "status": "infeasible",
            "reason": "the frozen verified source has zero C++ and zero Rust rows",
            "language_deficits": {"cpp": 75, "rust": 50},
            "maximum_quota_admissible_rows": 125,
            "source_target_deficit_at_language_quotas": source_shortfall,
            "no_random_labels": True,
        },
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "tests_executed_locally": 0,
    }
    write_json(control_root / "multi-swe-verified-source-freeze-report.json", report)
    return report


def audit_frozen_candidate_quota_feasibility(
    control_root: Path,
    swe_candidates: Sequence[Mapping[str, Any]],
    multiswe_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the production selector over frozen candidates without claiming verification."""
    registry_path = control_root / "exclusion-registry.json"
    registry = _load_document(registry_path) if registry_path.is_file() else {}
    rows = []
    excluded = Counter()
    for source_rows in (swe_candidates, multiswe_candidates):
        for raw in source_rows:
            patch = str(raw.get("target_patch", ""))
            normalized_patch_hash = sha256_bytes(_normalized_patch(patch).encode())
            row = {
                "case_id": raw["case_id"],
                "source_id": raw["source_id"],
                "coding_task_type": raw["primary_task_type"],
                "primary_language": raw["primary_language"],
                "upstream_repository": raw["upstream_repository"],
                "problem_id": raw["problem_id"],
                "source_lineage_id": raw.get("source_lineage_id")
                or _candidate_lineage(raw),
                "source_hash": raw.get("source_record_sha256")
                or sha256_bytes(canonical_json(raw).encode()),
                "normalized_patch_hash": normalized_patch_hash,
                "diff_hunk_hash": normalized_patch_hash,
                "test_set_hash": raw.get("test_patch_sha256", ""),
                "verified_runtime_seconds": 0,
                "status": "candidate_only_not_verified",
            }
            if benchmark_overlap(row, registry):
                excluded[str(raw["source_id"])] += 1
                continue
            rows.append(row)
    quotas = {
        "source_targets": PHASE1_SOURCE_TARGETS,
        "task_type_targets": PHASE1_TASK_TYPE_TARGETS,
        "language_targets": PHASE1_LANGUAGE_TARGETS,
    }
    _selected, selector_report = select_replay(rows, quotas, 500)
    report = {
        "schema_version": "general_coding_replay_candidate_quota_feasibility_v1",
        "status": selector_report["status"],
        "scope": "frozen_candidates_only_not_verified",
        "candidate_rows": len(rows),
        "excluded_by_registry": dict(sorted(excluded.items())),
        "deficits": selector_report["deficits"],
        "counts_at_stall": selector_report["counts"],
        "candidate_cross_distribution": selector_report["candidate_cross_distribution"],
        "selected_at_stall": selector_report["selected"],
        "selector_constraints": {
            "repository_cap": 15,
            "lineage_cap": 2,
            "source_cap_fraction": 0.60,
        },
        "verified_rows": 0,
    }
    write_json(control_root / "candidate-quota-feasibility.json", report)
    return report


def preflight_multiswe_candidate_seed(
    control_root: Path,
    candidate_path: Path,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Bounded metadata preflight; does not clone repositories or execute tests."""
    if not 1 <= limit <= 100:
        raise ValueError("Multi-SWE preflight limit must be in [1, 100]")
    rows = read_jsonl(candidate_path)
    checked = []
    for row in rows[:limit]:
        if row.get("dataset_revision") != MULTISWE_VERIFIED_REVISION:
            raise RuntimeError("Multi-SWE preflight found dataset revision drift")
        if row.get("source_id") != "multi_swe_rl_verified":
            raise RuntimeError("Multi-SWE preflight found source identity drift")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("base_commit", ""))):
            raise RuntimeError(f"{row.get('case_id')}: non-immutable base commit")
        target_patch = str(row.get("target_patch", ""))
        test_patch = str(row.get("test_patch", ""))
        if sha256_bytes(target_patch.encode()) != row.get("target_patch_sha256"):
            raise RuntimeError(f"{row.get('case_id')}: fix patch checksum mismatch")
        if sha256_bytes(test_patch.encode()) != row.get("test_patch_sha256"):
            raise RuntimeError(f"{row.get('case_id')}: test patch checksum mismatch")
        license_path = control_root / str(row.get("license_evidence_path", ""))
        if not license_path.is_file():
            raise RuntimeError(f"{row.get('case_id')}: frozen license evidence missing")
        image = row.get("container_image") or {}
        if not re.fullmatch(
            r".+@sha256:[0-9a-f]{64}", str(image.get("repo_digest", ""))
        ):
            raise RuntimeError(f"{row.get('case_id')}: immutable container digest missing")
        validation = row.get("validation_evidence") or {}
        if any(
            validation.get(key, {}).get("reward") != 1.0
            or validation.get(key, {}).get("reason") != "pass"
            for key in ("pass_1", "pass_2")
        ):
            raise RuntimeError(f"{row.get('case_id')}: two-pass validation evidence drift")
        checked.append(
            {
                "case_id": row["case_id"],
                "base_commit": row["base_commit"],
                "target_patch_sha256": row["target_patch_sha256"],
                "test_patch_sha256": row["test_patch_sha256"],
                "license_evidence_sha256": sha256_file(license_path),
                "container_repo_digest": image["repo_digest"],
            }
        )
    report = {
        "schema_version": "general_coding_replay_multiswe_preflight_v1",
        "status": "metadata_ready_local_verification_not_run",
        "seed_rows": len(rows),
        "bounded_rows_checked": len(checked),
        "limit": limit,
        "checks": checked,
        "source_manifest_sha256": sha256_file(candidate_path),
        "repositories_cloned": 0,
        "container_images_downloaded": 0,
        "untrusted_tests_executed": 0,
        "network_policy_for_future_verification": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "multi-swe-preflight-report.json", report)
    return report


def freeze_real_sources(control_root: Path, *, swe_candidate_count: int = 1000) -> dict[str, Any]:
    """Freeze pinned source bytes and emit an explicitly unverified candidate seed."""
    if not 900 <= swe_candidate_count <= SWE_GYM_ROWS:
        raise ValueError(
            f"SWE-Gym candidate count must be in the frozen 900-{SWE_GYM_ROWS} range"
        )
    control_root.mkdir(parents=True, exist_ok=True)
    cache = control_root / "source-cache"
    swe_cache = cache / "swe-gym" / SWE_GYM_REVISION
    license_dir = control_root / "license-evidence" / "swe-gym-upstream"

    swe_api = _fetch_json_frozen(
        f"https://huggingface.co/api/datasets/{SWE_GYM_DATASET_ID}/revision/{SWE_GYM_REVISION}",
        swe_cache / "dataset-api.json",
    )
    if swe_api.get("sha") != SWE_GYM_REVISION:
        raise RuntimeError("SWE-Gym API did not resolve to the required immutable revision")
    swe_readme = _download_frozen(
        f"https://huggingface.co/datasets/{SWE_GYM_DATASET_ID}/resolve/{SWE_GYM_REVISION}/README.md",
        swe_cache / "README.md",
        expected_size=1141,
    )
    swe_parquet = _download_frozen(
        f"https://huggingface.co/datasets/{SWE_GYM_DATASET_ID}/resolve/{SWE_GYM_REVISION}/data/train-00000-of-00001.parquet",
        swe_cache / "train-00000-of-00001.parquet",
        expected_sha256=SWE_GYM_PARQUET_SHA256,
        expected_size=43_644_473,
    )

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("source freezing requires pyarrow to read the pinned parquet") from exc
    table = parquet.read_table(swe_cache / "train-00000-of-00001.parquet")
    if table.num_rows != SWE_GYM_ROWS:
        raise RuntimeError(f"expected {SWE_GYM_ROWS} SWE-Gym rows, found {table.num_rows}")
    rows = table.to_pylist()
    eligible: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        ok, reason = _eligible_swe_row(row)
        if ok:
            eligible.append(row)
        else:
            rejection_counts[str(reason)] += 1

    representative_commits: dict[str, str] = {}
    for row in eligible:
        repo = str(row["repo"])
        commit = str(row["base_commit"])
        representative_commits[repo] = min(commit, representative_commits.get(repo, commit))
    license_dir.mkdir(parents=True, exist_ok=True)
    licenses = {
        repo: _license_evidence(repo, commit, license_dir)
        for repo, commit in sorted(representative_commits.items())
    }
    accepted_repos = {repo for repo, evidence in licenses.items() if evidence["status"] == "accepted"}
    for row in eligible:
        if str(row["repo"]) not in accepted_repos:
            rejection_counts[str(licenses[str(row["repo"])]["reason"])] += 1
    licensed = [row for row in eligible if str(row["repo"]) in accepted_repos]
    ranked = sorted(
        licensed,
        key=lambda row: (
            sha256_bytes(f"{SWE_GYM_REVISION}:{row['instance_id']}".encode()),
            str(row["instance_id"]),
        ),
    )
    selected = ranked[:swe_candidate_count]
    candidates = []
    for row in selected:
        repo = str(row["repo"])
        evidence = licenses[repo]
        task_type, classification = derive_repository_task_type(row)
        candidates.append(
            {
                "schema_version": "general_coding_replay_candidate_source_v1",
                "case_id": f"gc-replay-swe_gym-{row['instance_id']}",
                "source_id": "swe_gym",
                "dataset_id": SWE_GYM_DATASET_ID,
                "dataset_revision": SWE_GYM_REVISION,
                "dataset_row_id": row["instance_id"],
                "upstream_repository": f"https://github.com/{repo}.git",
                "base_commit": row["base_commit"],
                "problem_id": row["instance_id"],
                "problem_statement": row["problem_statement"],
                "target_patch": row["patch"],
                "target_patch_sha256": sha256_bytes(str(row["patch"]).encode()),
                "test_patch": row.get("test_patch", ""),
                "test_patch_sha256": sha256_bytes(str(row.get("test_patch", "")).encode()),
                "pass_to_pass": row.get("PASS_TO_PASS") or [],
                "fail_to_pass": row.get("FAIL_TO_PASS") or [],
                "license_spdx": evidence["spdx"],
                "license_evidence_path": (
                    Path("license-evidence") / "swe-gym-upstream" / str(evidence["license_text_path"])
                ).as_posix(),
                "primary_language": "python",
                "primary_task_type": task_type,
                "task_type_evidence": classification,
                "status": "candidate_unverified",
                "verified": False,
                "tests_executed": False,
            }
        )
    candidate_path = control_root / "swe-gym-candidate-source-manifest.jsonl"
    write_jsonl(candidate_path, candidates)
    write_json(control_root / "swe-gym-upstream-license-report.json", {"repositories": list(licenses.values())})
    multiswe_report = freeze_multiswe_verified_source(control_root)
    multiswe_candidate_path = control_root / "multi-swe-verified-candidate-source-manifest.jsonl"
    multiswe_preflight = preflight_multiswe_candidate_seed(
        control_root, multiswe_candidate_path
    )
    quota_feasibility = audit_frozen_candidate_quota_feasibility(
        control_root,
        candidates,
        read_jsonl(multiswe_candidate_path),
    )

    source_manifest = {
        "schema_version": "general_coding_replay_source_manifest_v1",
        "status": "candidate_seed_frozen_unverified",
        "network_policy": "disabled_for_verification",
        "frozen_sources": [
            {
                "dataset_id": SWE_GYM_DATASET_ID,
                "revision": SWE_GYM_REVISION,
                "license": "MIT (dataset); per-repository evidence required and frozen separately",
                "candidate_seed": candidate_path.name,
            },
            {
                "dataset_id": MULTISWE_VERIFIED_DATASET_ID,
                "revision": MULTISWE_VERIFIED_REVISION,
                "license": (
                    "CC0 with ByteDance notice; per-repository licenses required "
                    "before local verification"
                ),
                "candidate_seed": multiswe_candidate_path.name,
                "candidate_rows": multiswe_report["candidate_rows"],
                "validation_evidence": "multi-swe-verified-source-freeze-report.json",
            },
        ],
        "sources": [],
        "note": "No rows are inventory-verified; full source entries require materialized parents and commands.",
    }
    atomic_write(control_root / "source-manifest.yaml", yaml.safe_dump(source_manifest, sort_keys=True))

    report = {
        "schema_version": "general_coding_replay_source_freeze_v1",
        "swe_gym": {
            "dataset_rows": table.num_rows,
            "metadata_eligible_rows": len(eligible),
            "license_admissible_rows": len(licensed),
            "candidate_seed_rows": len(candidates),
            "verified_rows": 0,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "license_rejected_rows_by_repository": dict(
                sorted(
                    Counter(
                        str(row["repo"])
                        for row in eligible
                        if str(row["repo"]) not in accepted_repos
                    ).items()
                )
            ),
            "candidate_rows_by_repository": dict(
                sorted(Counter(str(row["repo"]) for row in selected).items())
            ),
            "repository_count": len(representative_commits),
            "license_accepted_repositories": len(accepted_repos),
            "license_rejected_repositories": len(representative_commits) - len(accepted_repos),
            "parquet": swe_parquet,
            "readme": swe_readme,
            "candidate_manifest_sha256": sha256_file(candidate_path),
        },
        "multi_swe_rl_verified": multiswe_report,
        "multi_swe_preflight": multiswe_preflight,
        "phase1_targets": {
            "sources": PHASE1_SOURCE_TARGETS,
            "languages": PHASE1_LANGUAGE_TARGETS,
            "task_types": PHASE1_TASK_TYPE_TARGETS,
        },
        "candidate_quota_feasibility": quota_feasibility,
        "untrusted_tests_executed": 0,
    }
    write_json(control_root / "source-freeze-report.json", report)
    checksum_path = control_root / "source-freeze-checksums.sha256"
    checksum_targets = [
        candidate_path,
        multiswe_candidate_path,
        control_root / "source-manifest.yaml",
        control_root / "source-freeze-report.json",
        control_root / "multi-swe-verified-source-freeze-report.json",
        control_root / "multi-swe-upstream-license-report.json",
        control_root / "multi-swe-preflight-report.json",
        control_root / "candidate-quota-feasibility.json",
        control_root / "swe-gym-upstream-license-report.json",
        swe_cache / "dataset-api.json",
        swe_cache / "README.md",
        swe_cache / "train-00000-of-00001.parquet",
        *sorted(
            (
                control_root
                / "source-cache"
                / "multi-swe-rl-verified"
                / MULTISWE_VERIFIED_REVISION
            ).glob("*")
        ),
        *sorted(license_dir.glob("*")),
        *sorted((control_root / "license-evidence" / "multi-swe-upstream").glob("*")),
    ]
    atomic_write(
        checksum_path,
        "".join(
            f"{sha256_file(path)}  {path.relative_to(control_root).as_posix()}\n"
            for path in sorted(checksum_targets)
        ),
    )
    return report


def _normalized_benchmark_text(value: object) -> str:
    return re.sub(r"[ \t]+\n", "\n", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def _python_signatures(source: object) -> list[str]:
    text = str(source or "")
    if not text.strip():
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            signatures.append(f"{prefix}{node.name}({ast.unparse(node.args)})")
    return sorted(set(signatures))


def _benchmark_entry(
    *,
    dataset_key: str,
    dataset_id: str,
    revision: str,
    config: str,
    split: str,
    problem_id: object,
    prompt: object,
    text: object,
    signature_source: object = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_prompt = _normalized_benchmark_text(prompt)
    normalized_text = _normalized_benchmark_text(text)
    signatures = _python_signatures(signature_source)
    entry: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "config": config,
        "split": split,
        "problem_id": str(problem_id),
        "prompt_sha256": sha256_bytes(normalized_prompt.encode()),
        "text_sha256": sha256_bytes(normalized_text.encode()),
        "signatures": signatures,
        "signature_fingerprints": [
            sha256_bytes(signature.encode()) for signature in signatures
        ],
    }
    if extra:
        entry.update(extra)
    return entry


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("benchmark registry freezing requires pyarrow") from exc
    return parquet.read_table(path).to_pylist()


def _validate_hf_revision(dataset_id: str, revision: str, cache_path: Path) -> dict[str, Any]:
    payload = _fetch_json_frozen(
        f"https://huggingface.co/api/datasets/{dataset_id}/revision/{revision}",
        cache_path,
    )
    if payload.get("sha") != revision:
        raise RuntimeError(f"{dataset_id}: immutable revision did not resolve exactly")
    return payload


def _freeze_livecodebench_entries(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream the smallest official release delta and retain no hidden tests."""
    _validate_hf_revision(
        LIVECODEBENCH_DATASET_ID,
        LIVECODEBENCH_REVISION,
        cache_root / "dataset-api.json",
    )
    url = (
        f"https://huggingface.co/datasets/{LIVECODEBENCH_DATASET_ID}/resolve/"
        f"{LIVECODEBENCH_REVISION}/{LIVECODEBENCH_SOURCE_FILE}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=cache_root, delete=False) as handle:
        temporary = Path(handle.name)
        with requests.get(
            url,
            stream=True,
            timeout=(30, 600),
            headers={"User-Agent": "general-coding-replay-freezer/1"},
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    try:
        receipt = validate_frozen_file(
            temporary,
            LIVECODEBENCH_SOURCE_SHA256,
            LIVECODEBENCH_SOURCE_SIZE,
        )
        entries = []
        observed_dates: list[str] = []
        with temporary.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                row = json.loads(raw)
                contest_date = str(row["contest_date"])
                if not LIVECODEBENCH_WINDOW[0] <= contest_date <= LIVECODEBENCH_WINDOW[1]:
                    continue
                observed_dates.append(contest_date)
                prompt = canonical_json(
                    {
                        "question_title": row["question_title"],
                        "question_content": row["question_content"],
                        "starter_code": row.get("starter_code", ""),
                    }
                )
                entries.append(
                    _benchmark_entry(
                        dataset_key="livecodebench",
                        dataset_id=LIVECODEBENCH_DATASET_ID,
                        revision=LIVECODEBENCH_REVISION,
                        config="v6",
                        split="test",
                        problem_id=row["question_id"],
                        prompt=prompt,
                        text=row["question_content"],
                        signature_source=row.get("starter_code", ""),
                        extra={
                            "contest_date": contest_date,
                            "platform": str(row["platform"]),
                            "source_file": LIVECODEBENCH_SOURCE_FILE,
                            "source_line_number": line_number,
                        },
                    )
                )
        entries.sort(key=lambda item: (item["contest_date"], item["problem_id"]))
        if len(entries) != len({item["problem_id"] for item in entries}):
            raise RuntimeError("LiveCodeBench window contains duplicate question IDs")
        source = {
            "url": url,
            "path": LIVECODEBENCH_SOURCE_FILE,
            "bytes": receipt["bytes"],
            "sha256": receipt["sha256"],
            "retained": False,
            "selection": {
                "release_component": "v6",
                "inclusive_start": LIVECODEBENCH_WINDOW[0],
                "inclusive_end": LIVECODEBENCH_WINDOW[1],
                "observed_min": min(observed_dates) if observed_dates else None,
                "observed_max": max(observed_dates) if observed_dates else None,
                "policy": (
                    "Freeze every problem in the official v6 delta (test6.jsonl) whose "
                    "contest_date is within the inclusive 2025-01-01 through 2025-04-30 window. "
                    "Earlier release components are excluded from this window."
                ),
            },
        }
        return entries, source
    finally:
        temporary.unlink(missing_ok=True)


def _pending_registry_blockers() -> list[dict[str, Any]]:
    return [
        {
            "id": "general_coding_dev_problem_ids",
            "domain": "General Coding Dev",
            "expected_count": 80,
            "status": "pending_unavailable",
            "reason": "authoritative problem IDs were not provided",
        },
        {
            "id": "general_coding_repository_held_out",
            "domain": "General Coding Held-out",
            "status": "pending_unavailable",
            "reason": "repository-level hidden-test held-out identities were not provided",
        },
        {
            "id": "known_benchmark_mirrors_rewrites",
            "domain": "known mirrors and rewrites",
            "status": "pending_unavailable",
            "reason": "no authoritative mirror/rewrite mapping was provided",
        },
        {
            "id": "kernel_dev_held_out",
            "domain": "Kernel Dev/Held-out",
            "status": "pending_unavailable",
            "reason": "authoritative Kernel Dev/Held-out identities were not provided",
        },
    ]


def _normalized_alias(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _alias_fingerprint(kind: str, value: object) -> str:
    return sha256_bytes(f"{kind}\0{_normalized_alias(value)}".encode())


def _benchmark_alias_records(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build exact normalized aliases from frozen metadata; never import benchmark code."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        evidence = {
            "dataset_key": str(entry["dataset_key"]),
            "dataset_revision": str(entry["dataset_revision"]),
            "problem_id": str(entry["problem_id"]),
        }
        problem_aliases = {
            str(entry["problem_id"]),
            f"{entry['dataset_key']}/{entry['problem_id']}",
            f"{entry['dataset_id']}/{entry['config']}/{entry['problem_id']}",
        }
        candidates: list[tuple[str, str, str]] = [
            ("normalized_problem_alias", _alias_fingerprint("problem", alias), alias)
            for alias in problem_aliases
        ]
        candidates.extend(
            (
                "normalized_signature_alias",
                _alias_fingerprint("signature", signature),
                signature,
            )
            for signature in entry.get("signatures", [])
        )
        candidates.extend(
            (
                "normalized_prompt_sha256",
                str(entry[field]),
                field,
            )
            for field in ("prompt_sha256", "text_sha256")
            if entry.get(field)
        )
        for kind, fingerprint, source_value in candidates:
            key = (kind, fingerprint)
            record = records.setdefault(
                key,
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "normalization": (
                        "unicode_casefold_remove_non_alphanumeric_sha256"
                        if kind != "normalized_prompt_sha256"
                        else "crlf_to_lf_trim_trailing_line_whitespace_sha256"
                    ),
                    "evidence": [],
                },
            )
            item = {**evidence, "source_value": source_value}
            if item not in record["evidence"]:
                record["evidence"].append(item)
    for record in records.values():
        record["evidence"].sort(
            key=lambda item: (
                item["dataset_key"],
                item["problem_id"],
                item["source_value"],
            )
        )
    return [records[key] for key in sorted(records)]


def _kernel_split_terms(path: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    lineages: set[str] = set()
    candidates: set[str] = set()
    contracts: set[str] = set()
    symbols: set[str] = set()
    implementations: set[str] = set()
    for row in rows:
        if row.get("split") != split:
            raise RuntimeError(f"{path}: row split is not {split!r}")
        lineage = str(row.get("source_lineage_id", "")).strip()
        if not lineage:
            raise RuntimeError(f"{path}: source_lineage_id is required")
        lineages.add(lineage)
        candidates.update(str(value) for value in row.get("candidate_ids", []) if value)
        contracts.update(str(value) for value in row.get("contract_family_ids", []) if value)
        implementations.update(
            str(value) for value in row.get("implementation_family_ids", []) if value
        )
        symbols.add(lineage)
        symbols.add(lineage.rsplit(":", 1)[-1])
        symbols.update(str(value) for value in row.get("top10_families", []) if value)
    return {
        "split": split,
        "source": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(rows),
        },
        "source_lineages": sorted(lineages),
        "candidate_ids": sorted(candidates),
        "contract_family_ids": sorted(contracts),
        "implementation_family_ids": sorted(implementations),
        "symbols": sorted(symbols),
    }


def _assert_append_source(existing: Mapping[str, Any] | None, current: Mapping[str, Any]) -> None:
    if not existing:
        return
    previous = existing.get("source", {})
    source = current.get("source", {})
    if previous.get("sha256") != source.get("sha256"):
        raise RuntimeError(
            f"append-safe reservation source drift: {previous.get('path')} "
            f"was {previous.get('sha256')}, now {source.get('sha256')}"
        )


_DEV_REFACTOR_TITLE = re.compile(
    r"(?i)^(?:(?:\[[^]]+\]|\w+):\s*)*"
    r"(?:refactor\b|rename\b|depr?\b|deprecate\b)"
    r"|\b(?:api migration|migration to|backward compatibility)\b"
)
_DEV_IMPLEMENTATION_TITLE = re.compile(
    r"(?i)^(?:(?:\[[^]]+\]|\w+):\s*)*"
    r"(?:feature request\b|implement\b|add support\b|add .*\b(?:method|function|api)\b)"
    r"|\b(?:missing implementation|not implemented|NotImplementedError|add support for)\b"
)
_DEV_TARGETS = {
    "repository_bug_fix_or_test_repair": 40,
    "function_implementation": 20,
    "refactor_or_api_adaptation": 20,
}


def _classify_dev_candidate(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    title = str(row.get("problem_statement", "")).splitlines()[0].strip()
    refactor = _DEV_REFACTOR_TITLE.search(title)
    implementation = _DEV_IMPLEMENTATION_TITLE.search(title)
    if refactor:
        category = "refactor_or_api_adaptation"
        evidence = {"field": "problem_statement_title", "match": refactor.group(0)}
    elif implementation:
        category = "function_implementation"
        evidence = {"field": "problem_statement_title", "match": implementation.group(0)}
    elif (
        row.get("primary_task_type") == "repository_bug_fix_or_test_repair"
        and row.get("fail_to_pass")
        and row.get("target_patch_sha256")
    ):
        category = "repository_bug_fix_or_test_repair"
        evidence = {
            "field": "frozen_source_metadata",
            "primary_task_type": row["primary_task_type"],
            "fail_to_pass_count": len(row["fail_to_pass"]),
            "target_patch_sha256": row["target_patch_sha256"],
        }
    else:
        category = "unclassified"
        evidence = {"field": "none", "reason": "insufficient evidence"}
    return category, {"title": title, **evidence}


def _candidate_lineage(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('dataset_id')}@{row.get('dataset_revision')}:"
        f"{row.get('problem_id')}"
    )


def _reserve_general_coding_dev(
    candidate_seed: Path,
    control_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = read_jsonl(candidate_seed)
    prepared_path = control_root / "prepared-source-manifest.jsonl"
    prepared = read_jsonl(prepared_path) if prepared_path.is_file() else []
    train_problem_ids = {str(row.get("problem_id", "")) for row in prepared}
    train_repositories = {str(row.get("upstream_repository", "")) for row in prepared}
    train_lineages = {_candidate_lineage(row) for row in prepared}
    train_patch_hashes: set[str] = set()
    for row in prepared:
        patch_path = Path(str(row.get("target_patch_path", "")))
        if patch_path.is_file():
            train_patch_hashes.add(
                sha256_bytes(_normalized_patch(patch_path.read_text(encoding="utf-8")).encode())
            )

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    for row in candidates:
        patch = str(row.get("target_patch", ""))
        patch_hash = sha256_bytes(_normalized_patch(patch).encode()) if patch else ""
        lineage = _candidate_lineage(row)
        repository = str(row.get("upstream_repository", ""))
        if str(row.get("problem_id", "")) in train_problem_ids:
            rejected["replay_train_problem_id"] += 1
            continue
        if repository in train_repositories:
            rejected["replay_train_repository"] += 1
            continue
        if lineage in train_lineages:
            rejected["replay_train_lineage"] += 1
            continue
        if patch_hash and patch_hash in train_patch_hashes:
            rejected["replay_train_normalized_patch"] += 1
            continue
        category, evidence = _classify_dev_candidate(row)
        if category == "unclassified":
            rejected["unclassified"] += 1
            continue
        stable_key = sha256_bytes(
            f"{row.get('dataset_revision')}\0{row.get('problem_id')}\0{patch_hash}".encode()
        )
        eligible[category].append(
            {
                "problem_id": str(row["problem_id"]),
                "case_id": str(row["case_id"]),
                "dataset_id": str(row["dataset_id"]),
                "dataset_revision": str(row["dataset_revision"]),
                "upstream_repository": repository,
                "source_lineage_id": lineage,
                "normalized_patch_hash": patch_hash,
                "task_type": category,
                "classification_evidence": evidence,
                "stable_order_key": stable_key,
            }
        )
    selected: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    available: dict[str, int] = {}
    for category, target in _DEV_TARGETS.items():
        pool = sorted(
            eligible.get(category, []),
            key=lambda item: (item["stable_order_key"], item["problem_id"]),
        )
        available[category] = len(pool)
        selected.extend(pool[:target])
        if len(pool) < target:
            deficits[category] = target - len(pool)
    selected.sort(key=lambda item: (item["task_type"], item["stable_order_key"]))
    report = {
        "schema_version": "general_coding_dev_reservation_v1",
        "status": "complete" if not deficits else "quota_deficits",
        "source": {
            "path": str(candidate_seed.resolve()),
            "sha256": sha256_file(candidate_seed),
            "rows": len(candidates),
        },
        "replay_train": {
            "path": str(prepared_path.resolve()),
            "sha256": sha256_file(prepared_path) if prepared_path.is_file() else None,
            "rows": len(prepared),
        },
        "targets": dict(_DEV_TARGETS),
        "available_by_evidence": available,
        "reserved_by_task_type": dict(Counter(item["task_type"] for item in selected)),
        "reserved": len(selected),
        "deficits": deficits,
        "isolation": {
            "problem_id": True,
            "source_lineage": True,
            "repository": True,
            "normalized_patch": True,
        },
        "rejected_counts": dict(sorted(rejected.items())),
        "tests_executed": 0,
    }
    return selected, report


def freeze_available_exclusion_components(
    control_root: Path,
    kernel_dev: Path,
    kernel_held_out: Path,
    candidate_seed: Path,
) -> dict[str, Any]:
    """Append deterministic reservations while retaining unavailable-domain blockers."""
    registry_path = control_root / "exclusion-registry.json"
    registry = _load_document(registry_path)
    if not isinstance(registry, dict):
        raise ValueError("exclusion registry must be an object")
    if registry.get("kernel_train", {}).get("rows") != 2000:
        raise RuntimeError("Kernel Train 2,000-row reservation is missing")

    kernel = {
        "dev": _kernel_split_terms(kernel_dev, "dev"),
        "held_out": _kernel_split_terms(kernel_held_out, "held_out"),
    }
    existing_kernel = registry.get("kernel_dev_held_out", {})
    for split in ("dev", "held_out"):
        _assert_append_source(existing_kernel.get(split), kernel[split])
    registry["kernel_dev_held_out"] = kernel
    registry["source_lineages"] = sorted(
        set(registry.get("source_lineages", []))
        | {value for item in kernel.values() for value in item["source_lineages"]}
    )
    registry["problem_ids"] = sorted(
        set(registry.get("problem_ids", []))
        | {value for item in kernel.values() for value in item["candidate_ids"]}
    )
    registry["contract_terms"] = sorted(
        set(registry.get("contract_terms", []))
        | {value for item in kernel.values() for value in item["contract_family_ids"]}
    )
    registry["symbols"] = sorted(
        set(registry.get("symbols", []))
        | {
            value
            for item in kernel.values()
            for field in ("symbols", "implementation_family_ids")
            for value in item[field]
        }
    )

    aliases = _benchmark_alias_records(registry.get("entries", []))
    registry["mirror_rewrite_fingerprints"] = aliases
    registry["mirror_rewrite_evidence"] = {
        "source_registry_datasets": [
            {
                "dataset_key": item["dataset_key"],
                "revision": item["revision"],
                "rows": item["rows"],
                "source_sha256": item["source"]["sha256"],
            }
            for item in registry.get("datasets", [])
        ],
        "entries": len(registry.get("entries", [])),
        "fingerprints": len(aliases),
        "untrusted_code_executed": 0,
    }

    dev_rows, dev_report = _reserve_general_coding_dev(candidate_seed, control_root)
    existing_dev = registry.get("general_coding_dev", {})
    _assert_append_source(existing_dev, dev_report)
    write_jsonl(control_root / "general-coding-dev-reservations.jsonl", dev_rows)
    write_json(control_root / "general-coding-dev-reservation-report.json", dev_report)
    registry["general_coding_dev"] = {
        **dev_report,
        "reservation_path": "general-coding-dev-reservations.jsonl",
        "reservation_sha256": sha256_file(
            control_root / "general-coding-dev-reservations.jsonl"
        ),
    }
    registry["problem_ids"] = sorted(
        set(registry.get("problem_ids", [])) | {item["problem_id"] for item in dev_rows}
    )
    registry["repositories"] = sorted(
        set(registry.get("repositories", []))
        | {item["upstream_repository"] for item in dev_rows}
    )
    registry["source_lineages"] = sorted(
        set(registry.get("source_lineages", []))
        | {item["source_lineage_id"] for item in dev_rows}
    )
    registry["normalized_patch_hashes"] = sorted(
        set(registry.get("normalized_patch_hashes", []))
        | {item["normalized_patch_hash"] for item in dev_rows}
    )

    completed = {"kernel_dev_held_out", "known_benchmark_mirrors_rewrites"}
    if dev_report["status"] == "complete":
        completed.add("general_coding_dev_problem_ids")
    blockers = [
        item for item in registry.get("blockers", _pending_registry_blockers())
        if item.get("id") not in completed
    ]
    for item in blockers:
        if item.get("id") == "general_coding_repository_held_out":
            item.update(
                {
                    "status": "pending_unavailable",
                    "reason": (
                        "authoritative repository hidden-test Held-out inventory was not "
                        "provided; public SWE-Gym patches are not a substitute"
                    ),
                    "required_inventory": [
                        "opaque held-out task_id",
                        "immutable repository URL and base commit",
                        "source lineage ID",
                        "hidden-test bundle checksum and access receipt",
                        "evaluation command contract",
                    ],
                }
            )
    registry["blockers"] = blockers
    registry["pending"] = [item["id"] for item in blockers]
    registry["status"] = (
        "frozen_pending_repository_hidden_test_held_out"
        if [item["id"] for item in blockers] == ["general_coding_repository_held_out"]
        else "frozen_with_pending_domains"
    )
    registry["schema_version"] = "general_coding_exclusion_registry_v3"
    write_json(registry_path, registry)
    audit = _audit_prepared_benchmark_overlap(control_root, registry)
    public_report_path = control_root / "public-benchmark-freeze-report.json"
    if public_report_path.is_file():
        public_report = _load_document(public_report_path)
        if isinstance(public_report, dict):
            public_report["registry_sha256"] = sha256_file(registry_path)
            public_report["overlap_audit"] = audit
            public_report["supplemental_components"] = "exclusion-component-freeze-report.json"
            write_json(public_report_path, public_report)
    report = {
        "schema_version": "general_coding_exclusion_component_freeze_v1",
        "kernel": {
            split: {
                "rows": value["source"]["rows"],
                "source_lineages": len(value["source_lineages"]),
                "candidate_ids": len(value["candidate_ids"]),
                "contract_family_ids": len(value["contract_family_ids"]),
                "implementation_family_ids": len(value["implementation_family_ids"]),
                "symbols": len(value["symbols"]),
                "source_sha256": value["source"]["sha256"],
            }
            for split, value in kernel.items()
        },
        "mirror_rewrite_fingerprints": len(aliases),
        "general_coding_dev": dev_report,
        "overlap_audit": audit,
        "remaining_blockers": blockers,
        "registry_sha256": sha256_file(registry_path),
        "untrusted_tests_executed": 0,
    }
    write_json(control_root / "exclusion-component-freeze-report.json", report)
    return report


def _audit_prepared_benchmark_overlap(
    control_root: Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = control_root / "prepared-source-manifest.jsonl"
    rows = read_jsonl(manifest) if manifest.is_file() else []
    fingerprints = _registry_fingerprints(registry)
    alias_fingerprints = {
        str(item["fingerprint"])
        for item in registry.get("mirror_rewrite_fingerprints", [])
        if isinstance(item, Mapping) and item.get("fingerprint")
    }
    excluded = []
    for row in rows:
        statement_path = Path(str(row.get("problem_statement_path", "")))
        statement = statement_path.read_text(encoding="utf-8") if statement_path.is_file() else ""
        prompt_hash = (
            sha256_bytes(_normalized_benchmark_text(statement).encode())
            if statement
            else ""
        )
        patch_path = Path(str(row.get("target_patch_path", "")))
        patch_hash = (
            sha256_bytes(_normalized_patch(patch_path.read_text(encoding="utf-8")).encode())
            if patch_path.is_file()
            else ""
        )
        normalized_aliases = {
            _alias_fingerprint("problem", row.get("problem_id", "")),
            *(
                _alias_fingerprint("signature", signature)
                for signature in _python_signatures(statement)
            ),
        }
        alias_matches = sorted(normalized_aliases & alias_fingerprints)
        matches = sorted(
            {
                value
                for value in (
                    str(row.get("problem_id", "")),
                    str(row.get("upstream_repository", "")),
                    prompt_hash,
                    patch_hash,
                )
                if value and value in fingerprints
            }
            | set(alias_matches)
        )
        if matches:
            excluded.append(
                {
                    "case_id": row.get("case_id"),
                    "problem_id": row.get("problem_id"),
                    "matches": matches,
                }
            )
    blockers = list(registry.get("blockers", []))
    gate_ready = not blockers and not excluded
    if gate_ready:
        for row in rows:
            row["benchmark_exclusion_gate"] = "ready"
        write_jsonl(manifest, rows)
    excluded_path = control_root / "benchmark-excluded-candidates.jsonl"
    write_jsonl(excluded_path, excluded)
    report = {
        "schema_version": "general_coding_benchmark_overlap_audit_v1",
        "prepared_candidates": len(rows),
        "overlap_count": len(excluded),
        "excluded_candidates": [item["case_id"] for item in excluded],
        "all_applicable_registries_frozen": not blockers,
        "benchmark_exclusion_gate": "ready" if gate_ready else "blocked_pending_registry",
        "remaining_blocker_ids": [item["id"] for item in blockers],
        "prepared_manifest_updated": gate_ready,
    }
    write_json(control_root / "benchmark-overlap-audit.json", report)
    return report


def freeze_public_benchmark_registry(control_root: Path) -> dict[str, Any]:
    """Freeze public benchmark identities and merge them into Kernel exclusions."""
    registry_path = control_root / "exclusion-registry.json"
    registry = _load_document(registry_path)
    if not isinstance(registry, dict):
        raise ValueError("exclusion registry must be an object")
    if registry.get("kernel_train", {}).get("rows") != 2000:
        raise RuntimeError("refusing to merge without the 2,000-row Kernel Train registry")
    kernel_snapshot = {
        key: registry.get(key)
        for key in ("kernel_train", "source_lineages", "contract_hashes", "symbols")
    }
    cache = control_root / "source-cache" / "public-benchmarks"

    humaneval_root = cache / "humaneval" / HUMANEVAL_REVISION
    _validate_hf_revision(
        HUMANEVAL_DATASET_ID,
        HUMANEVAL_REVISION,
        humaneval_root / "dataset-api.json",
    )
    humaneval_file = humaneval_root / "test-00000-of-00001.parquet"
    humaneval_source = _download_frozen(
        (
            f"https://huggingface.co/datasets/{HUMANEVAL_DATASET_ID}/resolve/"
            f"{HUMANEVAL_REVISION}/openai_humaneval/test-00000-of-00001.parquet"
        ),
        humaneval_file,
        expected_sha256=HUMANEVAL_TEST_SHA256,
        expected_size=83_920,
    )
    humaneval_rows = _read_parquet_rows(humaneval_file)
    if len(humaneval_rows) != HUMANEVAL_TEST_ROWS:
        raise RuntimeError(f"HumanEval row-count drift: {len(humaneval_rows)}")
    humaneval_entries = [
        _benchmark_entry(
            dataset_key="humaneval",
            dataset_id=HUMANEVAL_DATASET_ID,
            revision=HUMANEVAL_REVISION,
            config="openai_humaneval",
            split="test",
            problem_id=row["task_id"],
            prompt=row["prompt"],
            text=row["prompt"],
            signature_source=row["prompt"],
            extra={"entry_point": row["entry_point"]},
        )
        for row in humaneval_rows
    ]

    mbpp_root = cache / "mbpp" / MBPP_REVISION
    _validate_hf_revision(MBPP_DATASET_ID, MBPP_REVISION, mbpp_root / "dataset-api.json")
    mbpp_file = mbpp_root / "test-00000-of-00001.parquet"
    mbpp_source = _download_frozen(
        (
            f"https://huggingface.co/datasets/{MBPP_DATASET_ID}/resolve/{MBPP_REVISION}/"
            "sanitized/test-00000-of-00001.parquet"
        ),
        mbpp_file,
        expected_sha256=MBPP_SANITIZED_TEST_SHA256,
        expected_size=60_864,
    )
    mbpp_rows = _read_parquet_rows(mbpp_file)
    if len(mbpp_rows) != MBPP_SANITIZED_TEST_ROWS:
        raise RuntimeError(f"MBPP Sanitized row-count drift: {len(mbpp_rows)}")
    mbpp_entries = [
        _benchmark_entry(
            dataset_key="mbpp_sanitized",
            dataset_id=MBPP_DATASET_ID,
            revision=MBPP_REVISION,
            config="sanitized",
            split="test",
            problem_id=row["task_id"],
            prompt=row["prompt"],
            text=row["prompt"],
            signature_source=row["code"],
            extra={"source_file": row["source_file"]},
        )
        for row in mbpp_rows
    ]
    lcb_entries, lcb_source = _freeze_livecodebench_entries(
        cache / "livecodebench" / LIVECODEBENCH_REVISION
    )
    entries = sorted(
        humaneval_entries + mbpp_entries + lcb_entries,
        key=lambda item: (item["dataset_key"], item["problem_id"]),
    )
    registry.update(
        {
            "schema_version": "general_coding_exclusion_registry_v2",
            "status": "public_benchmarks_frozen_pending_private_domains",
            "datasets": [
                {
                    "dataset_key": "humaneval",
                    "dataset_id": HUMANEVAL_DATASET_ID,
                    "revision": HUMANEVAL_REVISION,
                    "config": "openai_humaneval",
                    "split": "test",
                    "rows": len(humaneval_entries),
                    "source": {**humaneval_source, "path": humaneval_file.relative_to(control_root).as_posix()},
                },
                {
                    "dataset_key": "mbpp_sanitized",
                    "dataset_id": MBPP_DATASET_ID,
                    "revision": MBPP_REVISION,
                    "config": "sanitized",
                    "split": "test",
                    "rows": len(mbpp_entries),
                    "source": {**mbpp_source, "path": mbpp_file.relative_to(control_root).as_posix()},
                },
                {
                    "dataset_key": "livecodebench",
                    "dataset_id": LIVECODEBENCH_DATASET_ID,
                    "revision": LIVECODEBENCH_REVISION,
                    "config": "v6",
                    "split": "test",
                    "rows": len(lcb_entries),
                    "source": lcb_source,
                },
            ],
            "entries": entries,
            "problem_ids": sorted(set(registry.get("problem_ids", [])) | {item["problem_id"] for item in entries}),
            "prompt_hashes": sorted(set(registry.get("prompt_hashes", [])) | {item["prompt_sha256"] for item in entries}),
            "text_hashes": sorted(set(registry.get("text_hashes", [])) | {item["text_sha256"] for item in entries}),
            "signature_fingerprints": sorted(
                set(registry.get("signature_fingerprints", []))
                | {
                    fingerprint
                    for item in entries
                    for fingerprint in item["signature_fingerprints"]
                }
            ),
            "blockers": _pending_registry_blockers(),
            "pending": [item["id"] for item in _pending_registry_blockers()],
        }
    )
    for key, value in kernel_snapshot.items():
        if registry.get(key) != value:
            raise RuntimeError(f"Kernel Train exclusion field changed during merge: {key}")
    write_json(registry_path, registry)
    audit = _audit_prepared_benchmark_overlap(control_root, registry)
    report = {
        "schema_version": "general_coding_public_benchmark_freeze_v1",
        "registry_path": registry_path.name,
        "registry_sha256": sha256_file(registry_path),
        "kernel_train_rows_preserved": registry["kernel_train"]["rows"],
        "dataset_counts": {
            "humaneval": len(humaneval_entries),
            "mbpp_sanitized": len(mbpp_entries),
            "livecodebench": len(lcb_entries),
        },
        "total_public_entries": len(entries),
        "overlap_audit": audit,
        "untrusted_benchmark_code_executed": 0,
        "large_livecodebench_source_retained": False,
    }
    write_json(control_root / "public-benchmark-freeze-report.json", report)
    return report


def bootstrap_kernel_exclusions(train_jsonl: Path, output: Path) -> dict[str, Any]:
    lineages: set[str] = set()
    contracts: set[str] = set()
    symbols: set[str] = set()
    rows = 0
    for row in read_jsonl(train_jsonl):
        rows += 1
        provenance = row.get("provenance", {})
        seed = row.get("input", {}).get("contract", {}).get("provenance", {}).get("case_seed", {})
        lineage = provenance.get("source_lineage_id") or seed.get("source_lineage_id")
        contract_hash = provenance.get("contract_hash") or row.get("input", {}).get("contract", {}).get("contract_hash")
        if lineage:
            lineages.add(str(lineage))
        if contract_hash:
            contracts.add(str(contract_hash))
        for value in (
            provenance.get("operator"),
            row.get("input", {}).get("contract", {}).get("operator"),
            seed.get("source_test_id"),
        ):
            if value:
                symbols.add(str(value))
        for source in row.get("input", {}).get("parent_source", {}).values():
            if isinstance(source, str):
                symbols.update(re.findall(r"(?m)^\s*(?:def|class)\s+([A-Za-z_]\w*)", source))
    if rows != 2000:
        raise RuntimeError(f"expected frozen Kernel Train to contain 2,000 rows, found {rows}")
    registry = {
        "schema_version": "general_coding_exclusion_registry_v1",
        "status": "bootstrap_pending_external_benchmarks",
        "kernel_train": {
            "path": str(train_jsonl.resolve()),
            "sha256": sha256_file(train_jsonl),
            "rows": rows,
        },
        "source_lineages": sorted(lineages),
        "contract_hashes": sorted(contracts),
        "symbols": sorted(symbols),
        "problem_ids": [],
        "repositories": [],
        "prompt_hashes": [],
        "text_hashes": [],
        "ast_fingerprints": [],
        "signature_fingerprints": [],
        "pending": [
            "HumanEval official test split",
            "MBPP Sanitized official test split",
            "LiveCodeBench frozen date window",
            "General Coding Dev 80 problem IDs",
            "repository-level hidden-test Held-out",
            "known mirrors and rewrites",
            "Kernel Dev/Held-out",
        ],
    }
    write_json(output, registry)
    return registry


LICENSE_FILENAMES = re.compile(
    r"^(?:LICENSE|LICENCE|COPYING)(?:\.[A-Za-z0-9._-]+)?$",
    re.IGNORECASE,
)


def _repository_slug(url: str) -> str:
    match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?", url)
    if not match:
        raise ValueError(f"unsupported upstream repository URL: {url}")
    return match.group(1)


def _mirror_has_commit(mirror: Path, commit: str) -> bool:
    result = _git(mirror, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
    if result.returncode:
        return False
    return _git(mirror, "rev-parse", f"{commit}^{{commit}}").stdout.strip() == commit


def _ensure_mirror(url: str, mirror: Path, *, allow_network: bool) -> dict[str, Any]:
    if mirror.exists():
        if _git(mirror, "rev-parse", "--is-bare-repository", check=False).stdout.strip() != "true":
            raise RuntimeError(f"shared mirror is not bare: {mirror}")
        action = "reused"
        if allow_network:
            _git(mirror, "remote", "update", "--prune")
            action = "updated"
    else:
        if not allow_network:
            return {"status": "blocked", "reason": "mirror_missing_network_disabled"}
        mirror.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--mirror", url, str(mirror)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30 * 60,
        )
        action = "cloned"
    return {"status": "available", "action": action}


def _patch_paths(patch: str) -> list[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = shlex.split(line)
        if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
            raise ReplayRejected("patch_apply_failed", f"malformed diff header: {line[:200]}")
        for raw in parts[2:]:
            paths.add(_safe_relative_path(raw[2:]))
    if not paths:
        raise ReplayRejected("target_patch_missing", "target patch has no git diff paths")
    return sorted(paths)


def _license_at_revision(
    mirror: Path,
    commit: str,
    expected_spdx: str,
    evidence_root: Path,
) -> dict[str, Any]:
    names = _git(mirror, "ls-tree", "-r", "--name-only", commit).stdout.splitlines()
    candidates = sorted(
        name
        for name in names
        if "/" not in name and LICENSE_FILENAMES.fullmatch(name)
    )
    if not candidates:
        raise ReplayRejected("license_unknown", f"{commit}: no root license file")
    path = candidates[0]
    blob = _git(mirror, "rev-parse", f"{commit}:{path}").stdout.strip()
    content = subprocess.run(
        ["git", "-C", str(mirror), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    evidence_root.mkdir(parents=True, exist_ok=True)
    text_path = evidence_root / f"{blob}.{Path(path).name}"
    if text_path.exists() and text_path.read_bytes() != content:
        raise RuntimeError(f"license blob collision: {text_path}")
    if not text_path.exists():
        atomic_write(text_path, content)
    metadata_path = evidence_root / "revisions" / f"{commit}.json"
    metadata = {
        "schema_version": "general_coding_replay_license_evidence_v1",
        "base_commit": commit,
        "license_path_at_revision": path,
        "license_blob": blob,
        "license_text_path": text_path.relative_to(evidence_root.parent.parent).as_posix(),
        "license_text_sha256": sha256_bytes(content),
        "spdx": expected_spdx,
        "source": "git_object_at_pinned_base_commit",
    }
    write_json(metadata_path, metadata)
    return {"metadata_path": metadata_path, **metadata}


def _pytest_command(values: Sequence[Any], *, disable_snail: bool = False) -> str:
    # Pytest renders control characters in parameterized node IDs as escapes.
    # Dataset JSON decoding turns those escapes back into literal characters,
    # so restore pytest's node-ID representation before shell quoting.
    tests = [
        str(value)
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        for value in values
        if str(value).strip()
    ]
    base = "python3 -m pytest"
    if disable_snail:
        base += " -p no:snail"
    return base + " -q " + " ".join(shlex.quote(value) for value in tests)


def _candidate_image_tag(repo: str, problem_id: str) -> str:
    instance_number = problem_id.rsplit("-", 1)[-1]
    if not instance_number.isdigit():
        raise ValueError(f"problem_id has no numeric SWE instance suffix: {problem_id}")
    image_repo = repo.replace("/", "_s_")
    return f"xingyaoww/sweb.eval.x86_64.{image_repo}-{instance_number}:latest"


def _local_docker_tags() -> set[str]:
    docker = shutil.which("docker")
    if not docker:
        return set()
    result = subprocess.run(
        [docker, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return set(result.stdout.splitlines()) if result.returncode == 0 else set()


def _environment_preflight(repo: str, mirror: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    images: list[str] = []
    docker_error = None
    if docker:
        result = subprocess.run(
            [docker, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            images = sorted(line for line in result.stdout.splitlines() if line)
        else:
            docker_error = result.stderr.strip()
    swe_images = [
        image for image in images
        if re.search(r"(?:swe[-_]?bench|swe[-_]?gym)", image, re.IGNORECASE)
    ]
    wheelhouses = [
        path for path in (
            Path.home() / ".cache" / "general-coding-replay" / "wheelhouse" / repo.replace("/", "__"),
            Path.home() / ".cache" / "swebench",
        )
        if path.is_dir() and any(path.iterdir())
    ]
    blockers = []
    if not swe_images:
        blockers.append("swe_gym_or_swe_bench_image_missing")
    if not wheelhouses:
        blockers.append("offline_dependency_bundle_missing")
    return {
        "repository": repo,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "matching_container_images": swe_images,
        "offline_dependency_bundles": [str(path) for path in wheelhouses],
        "docker_available": bool(docker),
        "docker_error": docker_error,
        "network_policy_for_verification": "disabled",
        "gpu_policy": "cpu_only",
    }


def prepare_swe_gym_candidates(
    candidate_seed: Path,
    control_root: Path,
    *,
    max_repositories: int | None = None,
    repositories: Sequence[str] = (),
    allow_network_for_mirrors: bool = False,
) -> dict[str, Any]:
    """Materialize immutable parents and metadata without executing target code."""
    rows = read_jsonl(candidate_seed)
    if any(row.get("dataset_revision") != SWE_GYM_REVISION for row in rows):
        raise RuntimeError("candidate seed contains a non-frozen SWE-Gym revision")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_repository_slug(str(row["upstream_repository"]))].append(row)
    selected_repositories = sorted(repositories) if repositories else sorted(grouped)
    unknown = sorted(set(selected_repositories) - set(grouped))
    if unknown:
        raise ValueError(f"repositories absent from candidate seed: {unknown}")
    if max_repositories is not None:
        selected_repositories = selected_repositories[:max_repositories]

    license_report = _load_document(control_root / "swe-gym-upstream-license-report.json")
    frozen_licenses = {
        str(item["repository"]): item
        for item in license_report.get("repositories", [])
        if item.get("status") == "accepted"
    }
    exclusion = _load_document(control_root / "exclusion-registry.json")
    benchmark_gate_ready = not bool(exclusion.get("pending"))
    mirrors_root = control_root / "mirrors"
    artifact_root = control_root / "prepared-artifacts"
    prepared_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    repository_reports: list[dict[str, Any]] = []
    untrusted_tests_executed = 0
    environment_reports = {
        repo: _environment_preflight(
            repo, mirrors_root / f"{repo.replace('/', '__')}.git"
        )
        for repo in sorted(grouped)
    }
    available_images: dict[str, dict[str, Any]] = {}
    local_docker_tags = _local_docker_tags()
    for repo, source_rows in grouped.items():
        for row in source_rows:
            repo_tag = _candidate_image_tag(repo, str(row["problem_id"]))
            identity = _docker_image_identity(repo_tag) if repo_tag in local_docker_tags else None
            if identity is not None:
                available_images[str(row["case_id"])] = identity
        available_count = sum(
            str(row["case_id"]) in available_images for row in source_rows
        )
        environment_reports[repo]["candidate_images_available"] = available_count
        environment_reports[repo]["candidate_images_missing"] = len(source_rows) - available_count
        if available_count and available_count < len(source_rows):
            environment_reports[repo]["status"] = "partially_available"
            environment_reports[repo]["blockers"] = [
                f"candidate_images_missing:{len(source_rows) - available_count}"
            ]
        environment_reports[repo]["available_candidate_images"] = [
            {"case_id": str(row["case_id"]), **available_images[str(row["case_id"])]}
            for row in source_rows
            if str(row["case_id"]) in available_images
        ]
    write_json(
        control_root / "environment-blockers.json",
        {
            "schema_version": "general_coding_replay_environment_preflight_v1",
            "network_policy_for_verification": "disabled",
            "gpu_policy": "cpu_only",
            "repositories": [environment_reports[repo] for repo in sorted(environment_reports)],
        },
    )

    for repo in selected_repositories:
        source_rows = sorted(grouped[repo], key=lambda row: str(row["case_id"]))
        mirror = mirrors_root / f"{repo.replace('/', '__')}.git"
        report: dict[str, Any] = {
            "repository": repo,
            "candidate_count": len(source_rows),
            "unique_base_commits": len({str(row["base_commit"]) for row in source_rows}),
            "mirror_path": str(mirror),
            "prepared": 0,
            "rejected": 0,
        }
        try:
            report["mirror"] = _ensure_mirror(
                str(source_rows[0]["upstream_repository"]),
                mirror,
                allow_network=allow_network_for_mirrors,
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            report["mirror"] = {"status": "blocked", "reason": str(exc)}
        report["environment"] = environment_reports[repo]
        if report["mirror"]["status"] != "available":
            for row in source_rows:
                rejections.append(
                    {
                        "case_id": row["case_id"],
                        "status": "blocked",
                        "rejection_reason": "parent_revision_missing",
                        "detail": report["mirror"]["reason"],
                    }
                )
            report["rejected"] = len(source_rows)
            repository_reports.append(report)
            continue
        if repo not in frozen_licenses:
            raise ReplayRejected("license_unknown", f"no accepted frozen license for {repo}")
        expected_spdx = str(frozen_licenses[repo]["spdx"])
        evidence_root = control_root / "license-evidence" / "prepared" / repo.replace("/", "__")
        for row in source_rows:
            case_id = str(row["case_id"])
            commit = str(row["base_commit"])
            try:
                if not re.fullmatch(r"[0-9a-f]{40}", commit) or not _mirror_has_commit(mirror, commit):
                    raise ReplayRejected("parent_revision_missing", f"{repo}@{commit}")
                license_evidence = _license_at_revision(
                    mirror, commit, expected_spdx, evidence_root
                )
                target_patch = str(row["target_patch"])
                test_patch = str(row.get("test_patch", ""))
                allowed_paths = _patch_paths(target_patch)
                _check_changed_paths(allowed_paths, allowed_paths, target_patch)
                case_root = artifact_root / case_id
                statement_path = case_root / "problem.md"
                target_path = case_root / "target.patch"
                test_patch_path = case_root / "test.patch"
                metadata_path = case_root / "test-metadata.json"
                atomic_write(statement_path, str(row["problem_statement"]))
                atomic_write(target_path, target_patch)
                atomic_write(test_patch_path, test_patch)
                test_metadata = {
                    "schema_version": "general_coding_replay_test_metadata_v1",
                    "pass_to_pass": list(row.get("pass_to_pass") or []),
                    "fail_to_pass": list(row.get("fail_to_pass") or []),
                    "test_patch_path": test_patch_path.relative_to(control_root).as_posix(),
                    "test_patch_sha256": sha256_bytes(test_patch.encode()),
                    "tests_executed": False,
                    "network_policy_for_verification": "disabled",
                }
                write_json(metadata_path, test_metadata)
                if sha256_file(target_path) != str(row["target_patch_sha256"]):
                    raise RuntimeError(f"{case_id}: target patch bytes changed")
                if sha256_file(test_patch_path) != str(row["test_patch_sha256"]):
                    raise RuntimeError(f"{case_id}: test patch bytes changed")
                targeted = _pytest_command(
                    test_metadata["fail_to_pass"],
                    disable_snail=repo == "facebookresearch/hydra",
                )
                regression = _pytest_command(
                    test_metadata["pass_to_pass"],
                    disable_snail=repo == "facebookresearch/hydra",
                )
                image_identity = available_images.get(case_id)
                entry = {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": case_id,
                    "source_id": "swe_gym",
                    "dataset_id": SWE_GYM_DATASET_ID,
                    "dataset_revision": SWE_GYM_REVISION,
                    "upstream_repository": str(row["upstream_repository"]),
                    "local_repository": str(mirror),
                    "base_commit": commit,
                    "problem_id": str(row["problem_id"]),
                    "license_spdx": expected_spdx,
                    "license_evidence_path": str(license_evidence["metadata_path"]),
                    "primary_language": "python",
                    "primary_task_type": "repository_bug_fix_or_test_repair",
                    "problem_statement_path": str(statement_path),
                    "target_patch_path": str(target_path),
                    "test_patch_path": str(test_patch_path),
                    "test_metadata_path": str(metadata_path),
                    "allowed_paths": allowed_paths,
                    "install_command": (
                        "true"
                        if image_identity
                        else (
                            "python3 -m pip install --no-index "
                            f"--find-links \"$GENERAL_CODING_WHEELHOUSE/{repo.replace('/', '__')}\" -e ."
                        )
                    ),
                    "compile_or_typecheck_command": "python3 -m compileall -q .",
                    "targeted_test_command": targeted,
                    # SWE-Gym's frozen PASS_TO_PASS set is the auditable
                    # regression contract. Running the repository's entire
                    # suite introduces unrelated/environment-specific failures.
                    "full_regression_command": regression,
                    "network_policy": "disabled",
                    "gpu_required": False,
                    "timeouts": dict(DEFAULT_TIMEOUTS),
                    "status": (
                        "prepared_benchmark_blocked"
                        if image_identity and not benchmark_gate_ready
                        else "prepared"
                        if image_identity and benchmark_gate_ready
                        else "prepared_environment_blocked"
                    ),
                    "parent_reproduced": True,
                    "tests_executed": False,
                    "benchmark_exclusion_gate": (
                        "ready" if benchmark_gate_ready else "blocked_pending_registry"
                    ),
                }
                if image_identity:
                    entry.update(
                        {
                            "verification_backend": "docker",
                            "container_image": image_identity,
                            "container_workdir": "/testbed",
                            "container_network": "none",
                            "container_gpu_devices": [],
                        }
                    )
                validate_source_entry(entry, control_root)
                prepared_rows.append(entry)
                report["prepared"] += 1
            except (ReplayRejected, RuntimeError, subprocess.SubprocessError) as exc:
                reason = exc.reason if isinstance(exc, ReplayRejected) else "parent_not_reproducible"
                rejections.append(
                    {
                        "case_id": case_id,
                        "status": "rejected",
                        "rejection_reason": reason,
                        "detail": str(exc),
                    }
                )
                report["rejected"] += 1
        repository_reports.append(report)

    prepared_rows.sort(key=lambda row: str(row["case_id"]))
    write_jsonl(control_root / "prepared-source-manifest.jsonl", prepared_rows)
    write_jsonl(
        control_root / "prepared-image-available-source-manifest.jsonl",
        [row for row in prepared_rows if row.get("verification_backend") == "docker"],
    )
    write_jsonl(control_root / "preflight-rejections.jsonl", rejections)
    frozen_manifest_path = control_root / "source-manifest.yaml"
    source_manifest = _load_document(frozen_manifest_path)
    source_manifest.update(
        {
            "status": "preflight_prepared_verification_not_run",
            "network_policy": "disabled_for_verification",
            "sources": prepared_rows,
            "note": "Parents and immutable artifacts prepared; untrusted tests have not run.",
        }
    )
    atomic_write(
        control_root / "prepared-source-manifest.yaml",
        yaml.safe_dump(source_manifest, sort_keys=True),
    )
    report = {
        "schema_version": "general_coding_replay_preflight_v1",
        "dataset_revision": SWE_GYM_REVISION,
        "seed_candidates": len(rows),
        "selected_repositories": selected_repositories,
        "selected_candidates": sum(len(grouped[repo]) for repo in selected_repositories),
        "prepared": len(prepared_rows),
        "rejected_or_blocked": len(rejections),
        "repository_reports": repository_reports,
        "environment_blockers_path": "environment-blockers.json",
        "benchmark_exclusion_gate": (
            "ready" if benchmark_gate_ready else "blocked_pending_registry"
        ),
        "untrusted_tests_executed": untrusted_tests_executed,
        "network_policy_for_verification": "disabled",
        "gpu_policy": "cpu_only",
        "candidate_seed_sha256": sha256_file(candidate_seed),
        "prepared_manifest_sha256": sha256_file(control_root / "prepared-source-manifest.jsonl"),
    }
    write_json(control_root / "preflight-report.json", report)
    return report


def inventory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen General Coding Replay candidate inventory.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(canonical_json(build_inventory(args.source_manifest, args.exclusion_registry, args.output)))
    return 0


def prepare_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare immutable SWE-Gym replay parents and metadata without running tests."
    )
    parser.add_argument("--candidate-seed", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--repository", action="append", default=[])
    parser.add_argument("--allow-network-for-mirrors", action="store_true")
    args = parser.parse_args(argv)
    report = prepare_swe_gym_candidates(
        args.candidate_seed,
        args.control_root,
        max_repositories=args.max_repositories,
        repositories=args.repository,
        allow_network_for_mirrors=args.allow_network_for_mirrors,
    )
    print(canonical_json(report))
    return 0


def freeze_sources_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze and import pinned General Coding Replay sources.")
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--swe-candidates", type=int, default=1000)
    args = parser.parse_args(argv)
    print(canonical_json(freeze_real_sources(args.control_root, swe_candidate_count=args.swe_candidates)))
    return 0


def freeze_benchmarks_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze authoritative public benchmark exclusions and audit prepared candidates."
    )
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(canonical_json(freeze_public_benchmark_registry(args.control_root)))
    return 0


def freeze_exclusion_components_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze available Kernel, benchmark-alias, and General Coding Dev exclusions."
    )
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--kernel-dev", type=Path, required=True)
    parser.add_argument("--kernel-held-out", type=Path, required=True)
    parser.add_argument("--candidate-seed", type=Path, required=True)
    args = parser.parse_args(argv)
    report = freeze_available_exclusion_components(
        args.control_root,
        args.kernel_dev,
        args.kernel_held_out,
        args.candidate_seed,
    )
    print(canonical_json(report))
    return 0 if report["general_coding_dev"]["status"] == "complete" else 2


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify General Coding Replay cases in fresh offline workspaces.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--network", choices=("disabled",), required=True)
    parser.add_argument("--gpus", choices=("disabled",), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-id")
    args = parser.parse_args(argv)
    print(
        canonical_json(
            verify_candidates(
                args.candidates,
                args.output_dir,
                resume=args.resume,
                limit=args.limit,
                workers=args.workers,
                batch_id=args.batch_id,
            )
        )
    )
    return 0


def select_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deduplicate and quota-select General Coding Replay.")
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--quota-file", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = select_verified(args.verified_dir, args.quota_file, args.target, args.output)
    print(canonical_json(report))
    return 0 if report["status"] == "exact" else 2


def package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package accepted General Coding Replay as relative-path HF artifacts.")
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(canonical_json(package_replay(args.accepted, args.hf_root)))
    return 0
