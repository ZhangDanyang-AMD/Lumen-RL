"""Append-only, provenance-preserving artifacts for Kernel SFT construction."""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "auth_token",
        "password",
        "secret",
        "token_value",
    }
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).strip().lower() in _SENSITIVE_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "argv": list(argv), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "argv": list(argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


class SFTCollector:
    """Write dataset sidecars without mutating normal trajectory semantics."""

    schema_version = "geak_sft_event_v1"

    def __init__(self, run_dir: Path, config: Any) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_id = self.run_dir.name
        self.dataset_root = Path(config.sft_dataset_root).expanduser().resolve()
        self.blob_root = self.dataset_root / "blobs" / "sha256"
        self.events_path = self.run_dir / "sft_events.jsonl"
        self.manifest_path = self.run_dir / "sft_manifest.json"
        self.environment_path = self.run_dir / "environment.json"
        self.task_type = str(config.sft_task_type)
        self._lock = threading.RLock()
        self._errors: list[str] = []
        self._candidate_count = 0
        self._accepted_candidate_count = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.blob_root.mkdir(parents=True, exist_ok=True)

        config_value = {
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
        }
        self.config_snapshot = _redact(config_value)
        self.config_hash = self.store_blob(_json_bytes(self.config_snapshot))
        environment = self._environment(config)
        if (environment.get("lumen_git") or {}).get("dirty"):
            self.mark_ineligible("lumen_worktree_dirty")
        if (environment.get("geak_git") or {}).get("dirty"):
            self.mark_ineligible("geak_worktree_dirty")
        self.environment_hash = self.store_blob(_json_bytes(environment))
        self._write_json(self.environment_path, environment)
        self.append(
            {
                "event": "collector_start",
                "timestamp": time.time(),
                "role": "orchestrator",
                "phase": "setup",
                "round": 0,
                "payload": {
                    "task_type": self.task_type,
                    "config_hash": self.config_hash,
                    "environment_hash": self.environment_hash,
                },
            }
        )

    def append(self, record: Mapping[str, Any]) -> None:
        event = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            **_redact(dict(record)),
        }
        line = json.dumps(
            event, sort_keys=True, default=str, ensure_ascii=False
        ) + "\n"
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def mark_ineligible(self, reason: str) -> None:
        value = str(reason).strip()
        if not value:
            return
        with self._lock:
            if value not in self._errors:
                self._errors.append(value)

    def store_blob(self, data: bytes) -> str:
        digest = _sha256(data)
        path = self.blob_root / digest[:2] / digest
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != data:
                    raise RuntimeError("content-addressed blob collision: %s" % digest)
            else:
                temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
                temporary.write_bytes(data)
                temporary.replace(path)
        return digest

    def record_plan(
        self,
        round_index: int,
        plan: Mapping[str, Any],
        directions: Sequence[Mapping[str, Any]],
        *,
        user_request: str,
    ) -> Path:
        created_at = time.time()
        payload = {
            "schema_version": "geak_sft_plan_v1",
            "run_id": self.run_id,
            "round": int(round_index),
            "task_type": self.task_type,
            "created_at": created_at,
            "user_request": user_request,
            "plan": _redact(dict(plan)),
            "directions": _redact(list(directions)),
        }
        path = self.run_dir / ("round_%d" % round_index) / "plan.json"
        self._write_json(path, payload)
        self.append(
            {
                "event": "sft_plan_frozen",
                "timestamp": created_at,
                "role": "tech_lead",
                "phase": "plan_round",
                "round": round_index,
                "payload": {
                    "plan_path": str(path),
                    "plan_hash": self.store_blob(_json_bytes(payload)),
                },
            }
        )
        return path

    def record_candidate(
        self,
        environment: Any,
        parent_session_id: str,
        candidate_session_id: str,
        candidate: Mapping[str, Any],
        verify_result: Mapping[str, Any],
        *,
        round_index: int,
        role: str = "engineer",
    ) -> dict[str, Any]:
        parent = self._source_snapshot(environment, parent_session_id)
        child = self._source_snapshot(environment, candidate_session_id)
        patch = self._unified_patch(parent, child)
        patch_bytes = patch.encode("utf-8")
        patch_hash = self.store_blob(patch_bytes)
        evaluation = dict(verify_result.get("evaluation") or {})
        independently_verified = (
            verify_result.get("verify_source") == "multitune_independent"
            and bool(verify_result.get("verify_session_id"))
        )
        if not independently_verified:
            self.mark_ineligible(
                "candidate_without_independent_verify:%s"
                % candidate.get("candidate_id")
            )
        record = {
            "schema_version": "geak_sft_candidate_v1",
            "run_id": self.run_id,
            "round": int(round_index),
            "role": role,
            "task_type": self.task_type,
            "recorded_at": time.time(),
            "candidate": _redact(dict(candidate)),
            "parent_session_id": parent_session_id,
            "candidate_session_id": candidate_session_id,
            "parent_sources": parent,
            "candidate_sources": child,
            "patch_hash": patch_hash,
            "patch_bytes": len(patch_bytes),
            "patch_applies": bool(patch),
            "verify_result": _redact(dict(verify_result)),
            "independent_verify": independently_verified,
            "compile_pass": bool(evaluation.get("compiled")),
            "correctness_pass": bool(evaluation.get("correct")),
            "benchmark_valid": bool(
                evaluation.get("compiled")
                and evaluation.get("correct")
                and evaluation.get("candidate_ms")
                and float(evaluation.get("speedup_geomean") or 0.0) > 0.0
            ),
        }
        accepted = bool(
            record["patch_applies"]
            and record["independent_verify"]
            and record["compile_pass"]
            and record["correctness_pass"]
            and record["benchmark_valid"]
            and candidate.get("accepted")
        )
        record["sft_positive_eligible"] = accepted and role == "engineer"
        path = self.run_dir / ("round_%d" % round_index) / "candidates.jsonl"
        line = json.dumps(
            record, sort_keys=True, default=str, ensure_ascii=False
        ) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._candidate_count += 1
            if record["sft_positive_eligible"]:
                self._accepted_candidate_count += 1
        self.append(
            {
                "event": "sft_candidate_recorded",
                "timestamp": record["recorded_at"],
                "role": role,
                "phase": "independent_verify",
                "round": round_index,
                "payload": {
                    "candidate_id": (candidate.get("candidate_id")),
                    "candidate_path": str(path),
                    "patch_hash": patch_hash,
                    "sft_positive_eligible": record["sft_positive_eligible"],
                },
            }
        )
        return record

    def finalize(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        try:
            events_hash = _sha256(self.events_path.read_bytes())
        except OSError as exc:
            self.mark_ineligible("sft_events_unreadable: %s" % exc)
            events_hash = ""
        manifest = {
            "schema_version": "geak_sft_manifest_v1",
            "run_id": self.run_id,
            "task_type": self.task_type,
            "created_at": time.time(),
            "collector_complete": not self._errors,
            "dataset_eligible": not self._errors,
            "ineligible_reasons": list(self._errors),
            "candidate_count": self._candidate_count,
            "positive_candidate_count": self._accepted_candidate_count,
            "events_path": str(self.events_path),
            "events_hash": events_hash,
            "environment_path": str(self.environment_path),
            "environment_hash": self.environment_hash,
            "config_hash": self.config_hash,
            "summary": _redact(dict(summary)),
        }
        self._write_json(self.manifest_path, manifest)
        return manifest

    def _source_snapshot(
        self, environment: Any, session_id: str
    ) -> dict[str, dict[str, Any]]:
        state = environment.get(session_id)
        workspace = Path(state.workspace).resolve()
        result: dict[str, dict[str, Any]] = {}
        for relative in state.sandbox.allowed_write_paths:
            path = (workspace / relative).resolve()
            try:
                path.relative_to(workspace)
            except ValueError as exc:
                self.mark_ineligible("source_path_escape:%s" % relative)
                raise RuntimeError("source path escapes workspace: %s" % relative) from exc
            data = path.read_bytes()
            result[str(relative)] = {
                "sha256": self.store_blob(data),
                "size": len(data),
            }
        return result

    def _unified_patch(
        self,
        parent: Mapping[str, Mapping[str, Any]],
        child: Mapping[str, Mapping[str, Any]],
    ) -> str:
        chunks: list[str] = []
        for relative in sorted(set(parent) | set(child)):
            before = self._blob_bytes(parent.get(relative)).decode(
                "utf-8", errors="surrogateescape"
            )
            after = self._blob_bytes(child.get(relative)).decode(
                "utf-8", errors="surrogateescape"
            )
            if before == after:
                continue
            chunks.extend(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile="a/" + relative,
                    tofile="b/" + relative,
                )
            )
        return "".join(chunks)

    def _blob_bytes(self, metadata: Mapping[str, Any] | None) -> bytes:
        if not metadata:
            return b""
        digest = str(metadata["sha256"])
        return (self.blob_root / digest[:2] / digest).read_bytes()

    def _write_json(self, path: Path, value: Any) -> None:
        data = _json_bytes(_redact(value))
        temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(data)
            temporary.replace(path)

    def _environment(self, config: Any) -> dict[str, Any]:
        lumen_root = Path(__file__).resolve().parents[4]
        geak_root = Path(config.geak_root).resolve()
        container = os.environ.get("GEAK_CONTAINER_NAME", "geak-phase1-vllm")
        return {
            "schema_version": "geak_sft_environment_v1",
            "captured_at": time.time(),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
            },
            "lumen_git": self._git_state(lumen_root),
            "geak_git": self._git_state(geak_root),
            "container": {
                "name": container,
                "identity": _run(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.Image}}",
                        container,
                    ]
                ),
            },
            "gpu_inventory": _run(
                [
                    "docker",
                    "exec",
                    container,
                    "/opt/rocm/bin/rocm-smi",
                    "--showproductname",
                    "--showuniqueid",
                ]
            ),
            "gpu_architecture": _run(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-lc",
                    "/opt/rocm/bin/rocminfo | "
                    "awk '/Name: +gfx/{print $2}' | sort -u",
                ]
            ),
            "gpu_ids": str(config.gpu_ids),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "GEAK_CONTAINER_NAME",
                    "GEAK_GPU_ALLOWED",
                    "GEAK_GPU_REQUIRE_IDLE",
                    "HIP_VISIBLE_DEVICES",
                    "PYTORCH_ROCM_ARCH",
                )
            },
        }

    @staticmethod
    def _git_state(root: Path) -> dict[str, Any]:
        head = _run(["git", "rev-parse", "HEAD"], cwd=root)
        branch = _run(["git", "branch", "--show-current"], cwd=root)
        remote = _run(["git", "remote", "get-url", "origin"], cwd=root)
        status = _run(["git", "status", "--porcelain"], cwd=root)
        diff = _run(["git", "diff", "--binary"], cwd=root)
        diff_bytes = str(diff.get("stdout") or "").encode("utf-8")
        return {
            "root": str(root),
            "head": str(head.get("stdout") or "").strip(),
            "branch": str(branch.get("stdout") or "").strip(),
            "remote": str(remote.get("stdout") or "").strip(),
            "dirty": bool(str(status.get("stdout") or "").strip()),
            "status": str(status.get("stdout") or "").splitlines(),
            "working_diff_sha256": _sha256(diff_bytes),
        }
