"""Deterministic ETL, validation, leakage, and coverage helpers for Kernel SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


TOOL_VERSION = "geak_sft_tools_v1"
INPUT_MANIFEST_SCHEMA = "geak_sft_input_manifest_v1"
DATASET_MANIFEST_SCHEMA = "geak_sft_dataset_manifest_v1"
SAMPLE_SCHEMA = "geak_kernel_sft_v1"
REJECTION_SCHEMA = "geak_sft_rejection_v1"
KNOWN_TASK_TYPES = {
    "cold_start",
    "profile_guided",
    "direction_conditioned",
    "error_recovery",
    "regression_balance",
}
KNOWN_SPLITS = {"train", "dev", "held_out"}
KNOWN_LANGUAGES = {"triton", "hip", "flydsl", "gluon"}
KNOWN_ARCHITECTURES = {"gfx942", "gfx950"}
VERIFY_SOURCES = {"multitune_independent", "verify_engineer"}
PROTECTED_PATH_PARTS = {
    "config.yaml",
    "metadata.json",
    "task_runner.py",
    "unittest.py",
    "oracle.py",
    "test.py",
    "tests",
    "hidden",
    "harness",
}
HEX256 = re.compile(r"^[0-9a-f]{64}$")
HUNK = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new>\d+)(?:,(?P<new_count>\d+))? @@"
)


class DatasetError(ValueError):
    """Raised for fatal manifest or dataset errors."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def compact_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require_map(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetError(f"{where}: expected JSON object")
    return value


def _required(record: Mapping[str, Any], key: str, where: str) -> Any:
    value = record.get(key)
    if value is None or value == "":
        raise DatasetError(f"{where}: missing required field {key}")
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path}: invalid JSON: {exc}") from exc
    return _require_map(value, str(path))


def read_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    records: list[Mapping[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [{"line": None, "error": str(exc)}]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            errors.append({"line": number, "error": "blank JSONL line"})
            continue
        try:
            value = json.loads(line)
            records.append(_require_map(value, f"{path}:{number}"))
        except (json.JSONDecodeError, DatasetError) as exc:
            errors.append({"line": number, "error": str(exc)})
    return records, errors


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> str:
    data = canonical_bytes(value)
    atomic_write(path, data)
    return sha256_bytes(data)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> str:
    data = b"".join(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for value in values
    )
    atomic_write(path, data)
    return sha256_bytes(data)


def _safe_relative(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or not candidate.parts
        or str(candidate) in {".", ""}
    ):
        raise DatasetError(f"unsafe patch path: {path!r}")
    return str(candidate)


def _diff_path(header: str) -> str | None:
    value = header[4:].rstrip("\r\n").split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return _safe_relative(value)


def apply_unified_patch(
    parent: Mapping[str, bytes], patch: bytes
) -> tuple[dict[str, bytes], set[str]]:
    """Apply a standard unified diff without touching a workspace."""
    try:
        text = patch.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"patch is not UTF-8: {exc}") from exc
    lines = text.splitlines(keepends=True)
    output = dict(parent)
    changed: set[str] = set()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            raise DatasetError(f"unexpected patch line {index + 1}")
        old_path = _diff_path(lines[index])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise DatasetError("missing +++ patch header")
        new_path = _diff_path(lines[index])
        index += 1
        target = new_path or old_path
        if target is None:
            raise DatasetError("patch has no source or destination path")
        before = output.get(old_path or "", b"").decode(
            "utf-8", errors="surrogateescape"
        )
        source_lines = before.splitlines(keepends=True)
        result: list[str] = []
        cursor = 0
        saw_hunk = False
        while index < len(lines) and not lines[index].startswith("--- "):
            match = HUNK.match(lines[index])
            if not match:
                raise DatasetError(f"invalid hunk header at patch line {index + 1}")
            saw_hunk = True
            old_start = int(match.group("old"))
            expected = max(0, old_start - 1)
            if expected < cursor or expected > len(source_lines):
                raise DatasetError("hunk source range is outside parent source")
            result.extend(source_lines[cursor:expected])
            cursor = expected
            old_seen = new_seen = 0
            old_count = int(match.group("old_count") or "1")
            new_count = int(match.group("new_count") or "1")
            index += 1
            while index < len(lines):
                line = lines[index]
                if line.startswith(("@@ ", "--- ")):
                    break
                if line.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not line or line[0] not in " +-":
                    raise DatasetError(f"invalid hunk body at patch line {index + 1}")
                body = line[1:]
                if line[0] == " ":
                    if cursor >= len(source_lines) or source_lines[cursor] != body:
                        raise DatasetError("patch context does not match parent source")
                    result.append(source_lines[cursor])
                    cursor += 1
                    old_seen += 1
                    new_seen += 1
                elif line[0] == "-":
                    if cursor >= len(source_lines) or source_lines[cursor] != body:
                        # difflib concatenates changed final lines when both inputs
                        # omit a trailing newline: ``-old+new``. Recover that
                        # deterministic representation without weakening replay.
                        source = source_lines[cursor] if cursor < len(source_lines) else ""
                        combined_prefix = source + "+"
                        if (
                            source
                            and not source.endswith(("\n", "\r"))
                            and body.startswith(combined_prefix)
                        ):
                            result.append(body[len(combined_prefix) :])
                            cursor += 1
                            old_seen += 1
                            new_seen += 1
                            index += 1
                            continue
                        raise DatasetError("patch deletion does not match parent source")
                    cursor += 1
                    old_seen += 1
                else:
                    result.append(body)
                    new_seen += 1
                index += 1
            if old_seen != old_count or new_seen != new_count:
                raise DatasetError("patch hunk count mismatch")
        if not saw_hunk:
            raise DatasetError("file patch contains no hunks")
        result.extend(source_lines[cursor:])
        if new_path is None:
            output.pop(target, None)
        else:
            output[target] = "".join(result).encode(
                "utf-8", errors="surrogateescape"
            )
        changed.add(target)
    if not changed:
        raise DatasetError("empty patch")
    return output, changed


def normalized_text_hash(patch: bytes) -> str:
    text = patch.decode("utf-8", errors="surrogateescape")
    normalized = "\n".join(
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    )
    return sha256_bytes(normalized.encode("utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _blob_path(blob_root: Path, digest: str) -> Path:
    if not HEX256.fullmatch(digest):
        raise DatasetError(f"invalid SHA256 digest: {digest!r}")
    return blob_root / digest[:2] / digest


def _nested(record: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if value is not None and value != "":
                return value
    return None


def _environment_fields(environment: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _nested(
        environment,
        ("gpu", "architecture"),
        ("gpu_architecture", "architecture"),
    )
    if architecture is None:
        arch_receipt = environment.get("gpu_architecture")
        if isinstance(arch_receipt, Mapping) and arch_receipt.get("ok"):
            values = str(arch_receipt.get("stdout") or "").split()
            if len(set(values)) == 1:
                architecture = values[0]
    inventory = environment.get("gpu_inventory")
    inventory_stdout = (
        str(inventory.get("stdout") or "")
        if isinstance(inventory, Mapping) and inventory.get("ok")
        else ""
    )
    sku_match = re.search(r"Card SKU:\s*(\S+)", inventory_stdout)
    return {
        "architecture": architecture,
        "gpu_sku": _nested(
            environment, ("gpu", "sku"), ("gpu_sku",), ("hardware", "gpu_sku")
        )
        or (sku_match.group(1) if sku_match else None),
        "rocm_version": _nested(
            environment,
            ("software", "rocm_version"),
            ("rocm_version",),
        ),
        "compiler_version": _nested(
            environment,
            ("software", "compiler_version"),
            ("compiler_version",),
        ),
        "container_digest": _nested(
            environment,
            ("container", "digest"),
            ("container", "image_digest"),
            ("container", "identity", "stdout"),
        ),
        "lumen_git_sha": _nested(environment, ("lumen_git", "head")),
        "geak_git_sha": _nested(environment, ("geak_git", "head")),
        "lumen_working_state_sha256": _nested(
            environment,
            ("lumen_git", "working_state_sha256"),
            ("lumen_git", "working_diff_sha256"),
        ),
        "geak_working_state_sha256": _nested(
            environment,
            ("geak_git", "working_state_sha256"),
            ("geak_git", "working_diff_sha256"),
        ),
    }


def _environment_with_override(
    environment: Mapping[str, Any], input_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    fields = _environment_fields(environment)
    if fields["rocm_version"] and fields["compiler_version"]:
        return environment
    digest = str(fields.get("container_digest") or "").strip()
    overrides = input_manifest.get("environment_overrides")
    if not digest or not isinstance(overrides, Mapping):
        return environment
    override = overrides.get(digest)
    if not isinstance(override, Mapping):
        return environment
    software = override.get("software")
    evidence = override.get("evidence")
    if not isinstance(software, Mapping) or not isinstance(evidence, Mapping):
        raise DatasetError(f"environment override for {digest} lacks software/evidence")
    for key in ("rocm_version", "compiler_version"):
        receipt = evidence.get(key)
        value = software.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or not isinstance(receipt, Mapping)
            or receipt.get("ok") is not True
            or receipt.get("returncode") != 0
            or not str(receipt.get("command") or "").strip()
            or str(receipt.get("stdout") or "").strip() != value.strip()
        ):
            raise DatasetError(
                f"environment override for {digest} has invalid {key} evidence"
            )
    merged = dict(environment)
    merged["software"] = dict(software)
    merged["software_override"] = dict(override)
    return merged


def _contract_fields(frozen: Mapping[str, Any]) -> dict[str, Any]:
    case = _require_map(frozen.get("contract"), "frozen input contract")
    contract = case.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    provenance = case.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    seed = provenance.get("case_seed")
    seed = seed if isinstance(seed, Mapping) else {}
    lane = seed.get("target_lane") or case.get("target_lane")
    language = case.get("language") or seed.get("language")
    lane_architecture = None
    if isinstance(lane, str):
        parts = lane.lower().rsplit("_", 1)
        if len(parts) == 2:
            lane_language, lane_architecture = parts
            if language is None:
                language = lane_language
    dtype = _nested(
        contract,
        ("dtype",),
        ("input", "dtype"),
    )
    data_format = _nested(
        contract,
        ("format",),
        ("input", "format"),
        ("weight", "format"),
    )
    shape_regime = (
        case.get("shape_regime")
        or contract.get("shape_regime")
        or seed.get("shape_regime")
    )
    if not shape_regime:
        shapes = case.get("shapes") or contract.get("shapes") or contract.get("shape")
        if shapes:
            values = [
                int(value)
                for value in re.findall(r"\d+", json.dumps(shapes, sort_keys=True))
            ]
            if values:
                maximum = max(values)
                shape_regime = (
                    "small" if maximum <= 256 else "medium" if maximum <= 2048 else "large"
                )
    return {
        "case": case,
        "contract": contract,
        "task_id": case.get("case_id") or case.get("task_id"),
        "operator": case.get("operator") or contract.get("operator"),
        "top10_family": case.get("top10_family")
        or provenance.get("top10_family")
        or seed.get("top10_family")
        or contract.get("top10_family"),
        "language": str(language).lower() if language else None,
        "architecture": case.get("architecture") or contract.get("architecture"),
        "lane": lane,
        "lane_architecture": lane_architecture,
        "dtype_format": "/".join(
            str(item) for item in (dtype, data_format) if item not in (None, "")
        ),
        "shape_regime": shape_regime,
        "source_lineage_id": frozen.get("source_lineage_id")
        or seed.get("source_lineage_id"),
        "implementation_family_id": seed.get("implementation_family_id")
        or provenance.get("implementation_family_id"),
        "split": frozen.get("split_group"),
        "split_version": frozen.get("split_version"),
        "contract_hash": case.get("contract_hash") or provenance.get("contract_hash"),
    }


def _native_quantization_unsupported(contract: Mapping[str, Any], arch: str) -> bool:
    text = json.dumps(contract, sort_keys=True).lower()
    native_mxfp = any(value in text for value in ("mxfp4", "mxfp6", "mxfp8"))
    return native_mxfp and arch != "gfx950"


def _receipt_ok(receipt: Any, mode: str) -> bool:
    return (
        isinstance(receipt, Mapping)
        and receipt.get("mode") == mode
        and receipt.get("ok") is True
        and receipt.get("returncode") == 0
        and isinstance(receipt.get("command"), str)
        and bool(receipt["command"].strip())
    )


def _reason(code: str, detail: str, field: str | None = None) -> dict[str, Any]:
    result = {"code": code, "detail": detail}
    if field:
        result["field"] = field
    return result


def rejection(
    run_id: str,
    candidate_id: str | None,
    source: str,
    reasons: Sequence[Mapping[str, Any]],
    *,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    identity = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "source": source,
        "reasons": list(reasons),
        "duplicate_of": duplicate_of,
    }
    return {
        "schema_version": REJECTION_SCHEMA,
        "rejection_id": compact_hash(identity),
        **identity,
    }


def _initial_reasons(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if candidate.get("role") != "engineer":
        reasons.append(_reason("not_engineer_candidate", "candidate role is not engineer"))
    if candidate.get("patch_applies") is not True:
        reasons.append(_reason("patch_does_not_apply", "collector marked patch invalid"))
    if candidate.get("independent_verify") is not True:
        reasons.append(_reason("verify_missing", "independent verification is absent"))
    if candidate.get("compile_pass") is not True:
        reasons.append(_reason("compile_failed", "compile gate did not pass"))
    if candidate.get("correctness_pass") is not True:
        reasons.append(_reason("correctness_failed", "correctness gate did not pass"))
    if (
        candidate.get("task_type") != "error_recovery"
        and candidate.get("benchmark_valid") is not True
    ):
        reasons.append(_reason("benchmark_invalid", "benchmark gate did not pass"))
    if candidate.get("sft_positive_eligible") is not True:
        reasons.append(
            _reason(
                "not_positive_eligible",
                "authoritative candidate record is not positive eligible",
            )
        )
    return reasons


def _validate_mode_input(
    task_type: str, frozen: Mapping[str, Any], reasons: list[dict[str, Any]]
) -> None:
    if task_type == "cold_start":
        forbidden = {
            "profile",
            "direction",
            "error_feedback",
            "per_case_benchmark",
            "regression_constraints",
        }
        present = sorted(key for key in forbidden if frozen.get(key) is not None)
        if present:
            reasons.append(
                _reason("invalid_frozen_input", f"cold_start contains {present}")
            )
    elif task_type == "profile_guided":
        profile = frozen.get("profile")
        if not isinstance(profile, Mapping):
            reasons.append(_reason("missing_profile", "profile_guided input has no profile"))
        elif profile.get("source_hash") != frozen.get("parent_source_hash"):
            reasons.append(
                _reason("profile_source_mismatch", "profile source hash differs")
            )
    elif task_type == "direction_conditioned":
        # Older collectors persist the direction in the candidate record.
        pass
    elif task_type == "error_recovery" and not isinstance(
        frozen.get("error_feedback"), Mapping
    ):
        reasons.append(
            _reason("missing_error_feedback", "error_recovery input has no exact error")
        )
    elif task_type == "regression_balance":
        if not isinstance(frozen.get("per_case_benchmark"), Mapping) or not isinstance(
            frozen.get("regression_constraints"), Mapping
        ):
            reasons.append(
                _reason(
                    "missing_regression_context",
                    "regression input or constraints are absent",
                )
            )


def _extract_tokens(candidate: Mapping[str, Any]) -> int | None:
    value = _nested(
        candidate,
        ("valid_assistant_tokens",),
        ("candidate", "valid_assistant_tokens"),
        ("candidate", "usage", "completion_tokens"),
    )
    return value if isinstance(value, int) and value >= 0 else None


def _build_one(
    candidate: Mapping[str, Any],
    frozen: Mapping[str, Any],
    environment: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    blob_root: Path,
    source: str,
    plan: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reasons = _initial_reasons(candidate)
    run_id = str(candidate.get("run_id") or run_manifest.get("run_id") or "")
    candidate_body = candidate.get("candidate")
    candidate_body = candidate_body if isinstance(candidate_body, Mapping) else {}
    candidate_id = candidate_body.get("candidate_id")
    if candidate.get("schema_version") != "geak_sft_candidate_v1":
        reasons.append(_reason("schema_error", "unexpected candidate schema_version"))
    if frozen.get("schema_version") != "geak_sft_frozen_input_v1":
        reasons.append(_reason("schema_error", "unexpected frozen input schema_version"))
    if run_manifest.get("schema_version") != "geak_sft_manifest_v1":
        reasons.append(_reason("schema_error", "unexpected run manifest schema_version"))
    if not run_manifest.get("collector_complete") or not run_manifest.get(
        "dataset_eligible"
    ):
        reasons.append(_reason("collector_incomplete", "run is not dataset eligible"))
    frozen_body = (
        frozen.get("input")
        if isinstance(frozen.get("input"), Mapping)
        else frozen
    )
    task_type = candidate.get("task_type")
    if task_type not in KNOWN_TASK_TYPES or frozen.get("task_type") != task_type:
        reasons.append(_reason("schema_error", "invalid or inconsistent task_type"))
    else:
        _validate_mode_input(str(task_type), frozen_body, reasons)
    persisted_direction = None
    if task_type == "direction_conditioned":
        candidate_direction = candidate_body.get("direction")
        direction_id = (
            candidate_direction.get("direction_id")
            if isinstance(candidate_direction, Mapping)
            else None
        )
        if (
            not isinstance(plan, Mapping)
            or plan.get("schema_version") != "geak_sft_plan_v1"
            or not isinstance(plan.get("directions"), list)
        ):
            reasons.append(
                _reason("missing_direction_plan", "persisted pre-engineer plan is absent")
            )
        else:
            persisted_direction = next(
                (
                    item
                    for item in plan["directions"]
                    if isinstance(item, Mapping)
                    and (item.get("direction_id") or item.get("id")) == direction_id
                ),
                None,
            )
            if persisted_direction is None:
                reasons.append(
                    _reason(
                        "direction_mismatch",
                        "candidate direction is not in the persisted plan",
                    )
                )
            if (
                isinstance(plan.get("created_at"), (int, float))
                and isinstance(frozen.get("created_at"), (int, float))
                and plan["created_at"] > frozen["created_at"]
            ):
                reasons.append(
                    _reason("direction_timing_invalid", "plan was saved after frozen input")
                )

    frozen_hash = candidate.get("frozen_input_hash")
    if not isinstance(frozen_hash, str) or sha256_bytes(canonical_bytes(frozen)) != frozen_hash:
        reasons.append(_reason("frozen_input_hash_mismatch", "frozen input hash differs"))
    elif not _blob_path(blob_root, frozen_hash).is_file():
        reasons.append(_reason("missing_blob", "frozen input blob is absent"))
    elif sha256_file(_blob_path(blob_root, frozen_hash)) != frozen_hash:
        reasons.append(_reason("blob_hash_mismatch", "frozen input blob is corrupt"))

    parent_text = frozen.get("input", frozen).get("parent_source") if isinstance(
        frozen.get("input", frozen), Mapping
    ) else None
    parent_text = frozen_body.get("parent_source")
    parent_meta = candidate.get("parent_sources")
    child_meta = candidate.get("candidate_sources")
    parent_bytes: dict[str, bytes] = {}
    authoritative_parent_text: dict[str, str] = {}
    expected_child: dict[str, bytes] = {}
    if not isinstance(parent_text, Mapping) or not parent_text:
        reasons.append(_reason("missing_parent_source", "frozen parent source is absent"))
    if not isinstance(parent_meta, Mapping) or not isinstance(child_meta, Mapping):
        reasons.append(_reason("missing_source_metadata", "source metadata is absent"))
    if isinstance(parent_text, Mapping) and isinstance(parent_meta, Mapping):
        for path, text in parent_text.items():
            if not isinstance(path, str) or not isinstance(text, str):
                reasons.append(_reason("schema_error", "parent source must map paths to text"))
                continue
            try:
                safe = _safe_relative(path)
                metadata = _require_map(parent_meta.get(path), f"parent source {path}")
                digest = str(_required(metadata, "sha256", f"parent source {path}"))
                blob = _blob_path(blob_root, digest)
                data = text.encode("utf-8")
                if not blob.is_file() or sha256_file(blob) != digest:
                    reasons.append(_reason("missing_blob", f"parent blob {digest} unavailable"))
                else:
                    blob_data = blob.read_bytes()
                    if blob_data.rstrip(b"\r\n") != data.rstrip(b"\r\n"):
                        reasons.append(
                            _reason(
                                "source_hash_mismatch",
                                f"parent bytes differ beyond trailing newline for {path}",
                            )
                        )
                    parent_bytes[safe] = blob_data
                    authoritative_parent_text[safe] = blob_data.decode(
                        "utf-8", errors="surrogateescape"
                    )
            except DatasetError as exc:
                reasons.append(_reason("source_hash_mismatch", str(exc)))
    if isinstance(child_meta, Mapping):
        for path, metadata_value in child_meta.items():
            try:
                safe = _safe_relative(str(path))
                metadata = _require_map(metadata_value, f"candidate source {path}")
                digest = str(_required(metadata, "sha256", f"candidate source {path}"))
                blob = _blob_path(blob_root, digest)
                if not blob.is_file() or sha256_file(blob) != digest:
                    reasons.append(_reason("missing_blob", f"candidate blob {digest} unavailable"))
                else:
                    expected_child[safe] = blob.read_bytes()
            except DatasetError as exc:
                reasons.append(_reason("source_hash_mismatch", str(exc)))

    frozen_source_hash = frozen_body.get("parent_source_hash")
    actual_source_hash = sha256_bytes(
        json.dumps(parent_text, sort_keys=True, default=str).encode("utf-8")
    ) if isinstance(parent_text, Mapping) else None
    if frozen_source_hash != actual_source_hash:
        reasons.append(_reason("source_hash_mismatch", "combined parent source hash differs"))
    source_hash = sha256_bytes(
        json.dumps(
            authoritative_parent_text, sort_keys=True, default=str
        ).encode("utf-8")
    )

    patch_hash = candidate.get("patch_hash")
    patch = b""
    if not isinstance(patch_hash, str):
        reasons.append(_reason("missing_patch", "patch hash is absent"))
    else:
        try:
            patch_path = _blob_path(blob_root, patch_hash)
            if not patch_path.is_file() or sha256_file(patch_path) != patch_hash:
                reasons.append(_reason("missing_blob", "patch blob is absent or corrupt"))
            else:
                patch = patch_path.read_bytes()
        except DatasetError as exc:
            reasons.append(_reason("missing_patch", str(exc)))
    if patch:
        try:
            applied, changed = apply_unified_patch(parent_bytes, patch)
            allowed = set(parent_meta or {}) | set(child_meta or {})
            if not changed <= allowed:
                reasons.append(
                    _reason("harness_modified", f"patch changed undeclared paths {sorted(changed - allowed)}")
                )
            protected = [
                path
                for path in changed
                if any(part.lower() in PROTECTED_PATH_PARTS for part in PurePosixPath(path).parts)
            ]
            if protected:
                reasons.append(
                    _reason("harness_modified", f"protected paths changed: {protected}")
                )
            if applied != expected_child:
                reasons.append(
                    _reason("patch_result_mismatch", "applied patch differs from candidate blobs")
                )
        except DatasetError as exc:
            reasons.append(_reason("patch_does_not_apply", str(exc)))

    verify = candidate.get("verify_result")
    verify = verify if isinstance(verify, Mapping) else {}
    evaluation = verify.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    if (
        verify.get("verify_source") not in VERIFY_SOURCES
        or not verify.get("verify_session_id")
        or not verify.get("verify_workspace")
    ):
        reasons.append(_reason("verify_missing", "verify provenance is incomplete"))
    if not _receipt_ok(evaluation.get("compile"), "compile"):
        reasons.append(_reason("compile_failed", "compile receipt is incomplete"))
    if not _receipt_ok(evaluation.get("correctness"), "correctness"):
        reasons.append(_reason("correctness_failed", "correctness receipt is incomplete"))
    if task_type != "error_recovery" and not _receipt_ok(
        evaluation.get("performance"), "performance"
    ):
        reasons.append(_reason("benchmark_invalid", "performance receipt is incomplete"))

    fields = _contract_fields(frozen_body)
    env = _environment_fields(environment)
    for key in (
        "task_id",
        "operator",
        "language",
        "architecture",
        "lane",
        "dtype_format",
        "shape_regime",
        "source_lineage_id",
        "implementation_family_id",
        "split",
        "split_version",
        "contract_hash",
    ):
        if fields.get(key) in (None, ""):
            reasons.append(_reason("missing_provenance", f"missing {key}", key))
    for key in (
        "gpu_sku",
        "rocm_version",
        "compiler_version",
        "container_digest",
        "lumen_git_sha",
        "geak_git_sha",
        "lumen_working_state_sha256",
        "geak_working_state_sha256",
    ):
        if env.get(key) in (None, ""):
            reasons.append(_reason("missing_provenance", f"missing {key}", key))
    if fields["language"] not in KNOWN_LANGUAGES:
        reasons.append(_reason("unsupported_contract", "unknown language"))
    if fields["architecture"] not in KNOWN_ARCHITECTURES:
        reasons.append(_reason("unsupported_contract", "unknown architecture"))
    if env["architecture"] != fields["architecture"]:
        reasons.append(_reason("architecture_mismatch", "verify GPU architecture differs"))
    if fields["lane_architecture"] != fields["architecture"]:
        reasons.append(_reason("lane_mismatch", "target lane architecture differs"))
    if fields["split"] not in KNOWN_SPLITS:
        reasons.append(_reason("split_leakage", "unknown split"))
    if _native_quantization_unsupported(fields["contract"], str(fields["architecture"])):
        reasons.append(
            _reason("unsupported_contract", "native MXFP requires gfx950")
        )

    if reasons:
        return None, reasons
    speedup = evaluation.get("speedup_geomean")
    if task_type != "error_recovery" and (
        not isinstance(speedup, (int, float)) or speedup <= 0
    ):
        return None, [_reason("benchmark_invalid", "verified speedup is invalid")]
    verified_speedup = (
        float(speedup)
        if isinstance(speedup, (int, float)) and speedup > 0
        else None
    )
    input_value = {
        "contract": fields["case"],
        "parent_source": authoritative_parent_text,
        "baseline": frozen_body.get("baseline"),
        "profile": frozen_body.get("profile"),
        "direction": persisted_direction
        if task_type == "direction_conditioned"
        else frozen_body.get("direction"),
        "error_feedback": frozen_body.get("error_feedback"),
        "per_case_benchmark": frozen_body.get("per_case_benchmark"),
        "regression_constraints": frozen_body.get("regression_constraints"),
    }
    identity = {
        "run_id": run_id,
        "round": candidate.get("round"),
        "candidate_id": candidate_id,
        "patch_hash": patch_hash,
        "frozen_input_hash": frozen_hash,
    }
    sample = {
        "schema_version": SAMPLE_SCHEMA,
        "sample_id": compact_hash(identity),
        "task_type": task_type,
        "split": fields["split"],
        "input": input_value,
        "output": {"patch": patch.decode("utf-8")},
        "labels": {
            "patch_applies": True,
            "compile_pass": True,
            "correctness_pass": True,
            "benchmark_valid": candidate.get("benchmark_valid") is True,
            "verified_speedup": verified_speedup,
        },
        "provenance": {
            **identity,
            "source_hash": source_hash,
            "frozen_parent_source_hash": frozen_source_hash,
            "parent_source_blobs": parent_meta,
            "candidate_source_blobs": child_meta,
            "patch_hash": patch_hash,
            "normalized_text_hash": normalized_text_hash(patch),
            "lumen_git_sha": env["lumen_git_sha"],
            "geak_git_sha": env["geak_git_sha"],
            "lumen_working_state_sha256": env["lumen_working_state_sha256"],
            "geak_working_state_sha256": env["geak_working_state_sha256"],
            "gpu": fields["architecture"],
            "gpu_sku": env["gpu_sku"],
            "rocm_version": env["rocm_version"],
            "compiler_version": env["compiler_version"],
            "container_digest": str(env["container_digest"]).strip(),
            "language": fields["language"],
            "lane": fields["lane"],
            "operator": fields["operator"],
            **(
                {"top10_family": fields["top10_family"]}
                if fields.get("top10_family")
                else {}
            ),
            "dtype_format": fields["dtype_format"],
            "shape_regime": fields["shape_regime"],
            "task_id": fields["task_id"],
            "contract_hash": fields["contract_hash"],
            "source_lineage_id": fields["source_lineage_id"],
            "implementation_family_id": fields["implementation_family_id"],
            "split_version": fields["split_version"],
            "verify_source": verify["verify_source"],
            "verify_session_id": verify["verify_session_id"],
            "verify_workspace": verify["verify_workspace"],
            "frozen_input_hash": frozen_hash,
            "raw_candidate_path": source,
            "valid_assistant_tokens": _extract_tokens(candidate),
        },
        "verification": verify,
    }
    return sample, []


def _run_entries(manifest: Mapping[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    runs = _required(manifest, "runs", str(manifest_path))
    if not isinstance(runs, list) or not runs:
        raise DatasetError(f"{manifest_path}: runs must be a non-empty list")
    entries: list[dict[str, Any]] = []
    for index, value in enumerate(runs):
        entry = {"path": value} if isinstance(value, str) else dict(
            _require_map(value, f"runs[{index}]")
        )
        _required(entry, "path", f"runs[{index}]")
        entries.append(entry)
    return entries


def _verify_pinned_artifacts(run_dir: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    artifacts = entry.get("artifacts")
    if artifacts is None:
        return reasons
    if not isinstance(artifacts, Mapping):
        return [_reason("input_manifest_error", "artifacts must be path-to-SHA map")]
    for relative, expected in artifacts.items():
        try:
            safe = _safe_relative(str(relative))
            path = (run_dir / safe).resolve()
            path.relative_to(run_dir)
            if not path.is_file():
                reasons.append(_reason("missing_raw_artifact", f"{safe} is absent"))
            elif sha256_file(path) != expected:
                reasons.append(_reason("raw_artifact_hash_mismatch", safe))
        except (DatasetError, ValueError) as exc:
            reasons.append(_reason("input_manifest_error", str(exc)))
    return reasons


def _candidate_files(run_dir: Path, entry: Mapping[str, Any]) -> list[Path]:
    specified = entry.get("candidate_files")
    if specified is not None:
        if not isinstance(specified, list):
            raise DatasetError("candidate_files must be a list")
        return [(run_dir / _safe_relative(str(item))).resolve() for item in specified]
    return sorted(run_dir.glob("round_*/candidates.jsonl"))


def _candidate_selected(
    record: Mapping[str, Any], entry: Mapping[str, Any]
) -> bool:
    """Apply an optional immutable candidate allow-list from the input manifest."""
    selected = entry.get("candidate_ids")
    if selected is None:
        return True
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        raise DatasetError("candidate_ids must be a list of non-empty strings")
    body = record.get("candidate")
    body = body if isinstance(body, Mapping) else {}
    return body.get("candidate_id") in set(selected)


def _deduplicate(
    samples: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    pinned_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    # Stable reliability ordering: complete token accounting, then speedup, then ID.
    pinned = pinned_keys or set()
    samples.sort(
        key=lambda item: (
            (
                str(item["provenance"].get("run_id") or ""),
                str(item["provenance"].get("candidate_id") or ""),
            )
            in pinned,
            item["provenance"].get("valid_assistant_tokens") is not None,
            item["labels"]["verified_speedup"] or 0.0,
            item["sample_id"],
        ),
        reverse=True,
    )
    patch_seen: dict[str, str] = {}
    normalized_seen: dict[str, str] = {}
    accepted: list[dict[str, Any]] = []
    for sample in samples:
        provenance = sample["provenance"]
        patch_hash = provenance["patch_hash"]
        normalized = provenance["normalized_text_hash"]
        duplicate_of = patch_seen.get(patch_hash)
        code = "duplicate_exact_patch"
        if duplicate_of is None:
            duplicate_of = normalized_seen.get(normalized)
            code = "duplicate_normalized_text"
        if duplicate_of is not None:
            rejections.append(
                rejection(
                    str(provenance["run_id"]),
                    str(provenance.get("candidate_id") or ""),
                    str(provenance["raw_candidate_path"]),
                    [_reason(code, "duplicate sample does not count toward quota")],
                    duplicate_of=duplicate_of,
                )
            )
            continue
        patch_seen[patch_hash] = sample["sample_id"]
        normalized_seen[normalized] = sample["sample_id"]
        accepted.append(sample)
    return sorted(accepted, key=lambda item: item["sample_id"])


def build_dataset(input_manifest_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    input_manifest_path = input_manifest_path.expanduser().resolve()
    input_manifest = load_json(input_manifest_path)
    if input_manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        raise DatasetError(
            f"{input_manifest_path}: schema_version must be {INPUT_MANIFEST_SCHEMA}"
        )
    dataset_version = str(
        _required(input_manifest, "dataset_version", str(input_manifest_path))
    )
    base = input_manifest_path.parent
    if output_root is None:
        root_value = _required(input_manifest, "output_root", str(input_manifest_path))
        output_root = _resolve(base, str(root_value))
    else:
        output_root = output_root.expanduser().resolve()
    all_samples: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    raw_files: set[Path] = set()
    candidates_considered = 0
    for entry in _run_entries(input_manifest, input_manifest_path):
        run_dir = _resolve(base, str(entry["path"]))
        run_id = run_dir.name
        pin_reasons = _verify_pinned_artifacts(run_dir, entry)
        try:
            run_manifest_path = run_dir / "sft_manifest.json"
            environment_path = run_dir / "environment.json"
            run_manifest = load_json(run_manifest_path)
            environment = load_json(environment_path)
            environment = _environment_with_override(environment, input_manifest)
            raw_files.update((run_manifest_path, environment_path))
        except DatasetError as exc:
            rejections.append(
                rejection(run_id, None, str(run_dir), [_reason("invalid_json", str(exc))])
            )
            continue
        if pin_reasons:
            rejections.append(rejection(run_id, None, str(run_dir), pin_reasons))
            continue
        try:
            candidate_files = _candidate_files(run_dir, entry)
        except DatasetError as exc:
            rejections.append(
                rejection(
                    run_id, None, str(run_dir), [_reason("input_manifest_error", str(exc))]
                )
            )
            continue
        if not candidate_files:
            rejections.append(
                rejection(
                    run_id,
                    None,
                    str(run_dir),
                    [_reason("missing_raw_artifact", "no candidate JSONL files")],
                )
            )
            continue
        blob_root = _resolve(
            base,
            str(
                entry.get("blob_root")
                or input_manifest.get("blob_root")
                or (run_dir.parent.parent / "blobs" / "sha256")
            ),
        )
        for candidates_path in candidate_files:
            raw_files.add(candidates_path)
            records, parse_errors = read_jsonl(candidates_path)
            for error in parse_errors:
                rejections.append(
                    rejection(
                        run_id,
                        None,
                        f"{candidates_path}:{error['line']}",
                        [_reason("invalid_jsonl", str(error["error"]))],
                    )
                )
            round_name = candidates_path.parent.name
            frozen_path = candidates_path.parent / "frozen_input.json"
            try:
                frozen = load_json(frozen_path)
                raw_files.add(frozen_path)
            except DatasetError as exc:
                for record in records or [{}]:
                    body = record.get("candidate")
                    body = body if isinstance(body, Mapping) else {}
                    rejections.append(
                        rejection(
                            run_id,
                            body.get("candidate_id"),
                            str(candidates_path),
                            [_reason("invalid_json", str(exc))],
                        )
                    )
                continue
            plan = None
            if frozen.get("task_type") == "direction_conditioned":
                plan_path = candidates_path.parent / "plan.json"
                try:
                    plan = load_json(plan_path)
                    raw_files.add(plan_path)
                except DatasetError:
                    # Each candidate receives a structured missing-plan reason below.
                    plan = None
            try:
                selected_records = [
                    record for record in records if _candidate_selected(record, entry)
                ]
            except DatasetError as exc:
                rejections.append(
                    rejection(
                        run_id,
                        None,
                        str(candidates_path),
                        [_reason("input_manifest_error", str(exc))],
                    )
                )
                continue
            for record in selected_records:
                candidates_considered += 1
                body = record.get("candidate")
                body = body if isinstance(body, Mapping) else {}
                sample, reasons = _build_one(
                    record,
                    frozen,
                    environment,
                    run_manifest,
                    blob_root,
                    f"{candidates_path}:{round_name}",
                    plan,
                )
                if sample is None:
                    rejections.append(
                        rejection(
                            run_id,
                            body.get("candidate_id"),
                            str(candidates_path),
                            reasons,
                        )
                    )
                else:
                    all_samples.append(sample)
    raw_pins = input_manifest.get("pinned_sample_keys", [])
    if not isinstance(raw_pins, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or not all(isinstance(value, str) and value for value in item)
        for item in raw_pins
    ):
        raise DatasetError("pinned_sample_keys must contain [run_id, candidate_id] pairs")
    pinned_keys = {(item[0], item[1]) for item in raw_pins}
    samples = _deduplicate(all_samples, rejections, pinned_keys)
    accepted_keys = {
        (
            str(sample["provenance"].get("run_id") or ""),
            str(sample["provenance"].get("candidate_id") or ""),
        )
        for sample in samples
    }
    missing_pins = sorted(pinned_keys - accepted_keys)
    if missing_pins:
        raise DatasetError(
            f"{len(missing_pins)} pinned samples are unavailable after deduplication"
        )
    processed = output_root / "processed"
    files: dict[str, dict[str, Any]] = {}
    for split in sorted(KNOWN_SPLITS):
        path = processed / f"{split}.jsonl"
        split_samples = [item for item in samples if item["split"] == split]
        digest = write_jsonl(path, split_samples)
        files[str(path.relative_to(output_root))] = {
            "sha256": digest,
            "records": len(split_samples),
            "bytes": path.stat().st_size,
            "schema_version": SAMPLE_SCHEMA,
        }
    rejected_path = processed / "rejected.jsonl"
    rejections.sort(key=lambda item: item["rejection_id"])
    digest = write_jsonl(rejected_path, rejections)
    files[str(rejected_path.relative_to(output_root))] = {
        "sha256": digest,
        "records": len(rejections),
        "bytes": rejected_path.stat().st_size,
        "schema_version": REJECTION_SCHEMA,
    }
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "dataset_version": dataset_version,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "output_root": str(output_root),
        "files": files,
        "raw_inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(raw_files)
            if path.is_file()
        ],
        "counts": {
            "candidates_considered": candidates_considered,
            "accepted": len(samples),
            "rejected": len(rejections),
            "duplicates_rejected": sum(
                any(reason["code"].startswith("duplicate_") for reason in item["reasons"])
                for item in rejections
            ),
        },
    }
    manifest["content_sha256"] = compact_hash(manifest)
    manifest_path = output_root / "dataset_manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    return {
        "tool_version": TOOL_VERSION,
        "dataset_version": dataset_version,
        "manifest": str(manifest_path),
        "sha256": manifest_sha,
        "accepted": len(samples),
        "rejected": len(rejections),
    }


def _dataset_files(manifest_path: Path) -> tuple[Mapping[str, Any], Path]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise DatasetError(f"{manifest_path}: invalid dataset manifest schema")
    content = dict(manifest)
    expected_content_hash = content.pop("content_sha256", None)
    if expected_content_hash != compact_hash(content):
        raise DatasetError(f"{manifest_path}: manifest content SHA256 mismatch")
    root = _resolve(manifest_path.parent, str(_required(manifest, "output_root", "manifest")))
    files = _required(manifest, "files", "manifest")
    if not isinstance(files, Mapping):
        raise DatasetError("manifest files must be an object")
    return manifest, root


def validate_sample(sample: Mapping[str, Any], where: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for key in (
        "schema_version",
        "sample_id",
        "task_type",
        "split",
        "input",
        "output",
        "labels",
        "provenance",
        "verification",
    ):
        if key not in sample:
            errors.append(_reason("schema_error", f"{where}: missing {key}", key))
    if sample.get("schema_version") != SAMPLE_SCHEMA:
        errors.append(_reason("schema_error", f"{where}: invalid schema_version"))
    if sample.get("task_type") not in KNOWN_TASK_TYPES:
        errors.append(_reason("schema_error", f"{where}: invalid task_type"))
    if sample.get("split") not in KNOWN_SPLITS:
        errors.append(_reason("schema_error", f"{where}: invalid split"))
    output = sample.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("patch"), str):
        errors.append(_reason("schema_error", f"{where}: output.patch missing"))
    labels = sample.get("labels")
    for key in ("patch_applies", "compile_pass", "correctness_pass"):
        if not isinstance(labels, Mapping) or labels.get(key) is not True:
            errors.append(_reason("quality_gate_failed", f"{where}: label {key} is not true"))
    if (
        sample.get("task_type") != "error_recovery"
        and (
            not isinstance(labels, Mapping)
            or labels.get("benchmark_valid") is not True
        )
    ):
        errors.append(
            _reason("quality_gate_failed", f"{where}: label benchmark_valid is not true")
        )
    provenance = sample.get("provenance")
    required_provenance = (
        "run_id",
        "candidate_id",
        "source_hash",
        "patch_hash",
        "lumen_git_sha",
        "geak_git_sha",
        "lumen_working_state_sha256",
        "geak_working_state_sha256",
        "gpu",
        "gpu_sku",
        "rocm_version",
        "compiler_version",
        "container_digest",
        "implementation_family_id",
        "source_lineage_id",
        "verify_source",
        "verify_session_id",
        "frozen_input_hash",
        "language",
        "lane",
        "task_id",
        "operator",
        "shape_regime",
        "dtype_format",
    )
    for key in required_provenance:
        if not isinstance(provenance, Mapping) or provenance.get(key) in (None, ""):
            errors.append(_reason("missing_provenance", f"{where}: missing {key}", key))
    if isinstance(provenance, Mapping) and isinstance(output, Mapping):
        patch = output.get("patch")
        if isinstance(patch, str) and sha256_bytes(patch.encode("utf-8")) != provenance.get(
            "patch_hash"
        ):
            errors.append(_reason("hash_mismatch", f"{where}: patch hash mismatch"))
        parent_source = (
            sample.get("input", {}).get("parent_source")
            if isinstance(sample.get("input"), Mapping)
            else None
        )
        if isinstance(parent_source, Mapping) and all(
            isinstance(path, str) and isinstance(text, str)
            for path, text in parent_source.items()
        ):
            actual_source_hash = sha256_bytes(
                json.dumps(parent_source, sort_keys=True, default=str).encode("utf-8")
            )
            if actual_source_hash != provenance.get("source_hash"):
                errors.append(
                    _reason("source_hash_mismatch", f"{where}: parent source hash mismatch")
                )
            if isinstance(patch, str):
                try:
                    applied, changed = apply_unified_patch(
                        {
                            str(path): str(text).encode("utf-8")
                            for path, text in parent_source.items()
                        },
                        patch.encode("utf-8"),
                    )
                    child_meta = provenance.get("candidate_source_blobs")
                    if not isinstance(child_meta, Mapping):
                        errors.append(
                            _reason(
                                "missing_provenance",
                                f"{where}: candidate source metadata missing",
                            )
                        )
                    else:
                        actual_hashes = {
                            path: sha256_bytes(data) for path, data in applied.items()
                        }
                        expected_hashes = {
                            str(path): metadata.get("sha256")
                            for path, metadata in child_meta.items()
                            if isinstance(metadata, Mapping)
                        }
                        if actual_hashes != expected_hashes:
                            errors.append(
                                _reason(
                                    "patch_result_mismatch",
                                    f"{where}: replayed source differs",
                                )
                            )
                    if any(
                        part.lower() in PROTECTED_PATH_PARTS
                        for path in changed
                        for part in PurePosixPath(path).parts
                    ):
                        errors.append(
                            _reason("harness_modified", f"{where}: protected path changed")
                        )
                except DatasetError as exc:
                    errors.append(
                        _reason("patch_does_not_apply", f"{where}: {exc}")
                    )
        else:
            errors.append(
                _reason("missing_parent_source", f"{where}: parent source missing")
            )
    return errors


def validate_dataset(manifest_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest, root = _dataset_files(manifest_path)
    errors: list[dict[str, Any]] = []
    files = manifest["files"]
    sample_count = rejection_count = 0
    for relative, metadata_value in sorted(files.items()):
        metadata = _require_map(metadata_value, f"files.{relative}")
        path = (root / _safe_relative(str(relative))).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(_reason("manifest_path_escape", str(relative)))
            continue
        if not path.is_file():
            errors.append(_reason("missing_output", str(relative)))
            continue
        if sha256_file(path) != metadata.get("sha256"):
            errors.append(_reason("hash_mismatch", str(relative)))
        records, parse_errors = read_jsonl(path)
        errors.extend(
            _reason("invalid_jsonl", f"{relative}:{item['line']}: {item['error']}")
            for item in parse_errors
        )
        if metadata.get("records") != len(records):
            errors.append(_reason("record_count_mismatch", str(relative)))
        if metadata.get("schema_version") == SAMPLE_SCHEMA:
            sample_count += len(records)
            for index, sample in enumerate(records, 1):
                errors.extend(validate_sample(sample, f"{relative}:{index}"))
                if sample.get("split") != Path(relative).stem:
                    errors.append(_reason("split_mismatch", f"{relative}:{index}"))
        elif metadata.get("schema_version") == REJECTION_SCHEMA:
            rejection_count += len(records)
            for index, item in enumerate(records, 1):
                if (
                    item.get("schema_version") != REJECTION_SCHEMA
                    or not isinstance(item.get("reasons"), list)
                    or not item["reasons"]
                ):
                    errors.append(_reason("schema_error", f"{relative}:{index}"))
    report = {
        "schema_version": "geak_sft_quality_report_v1",
        "tool_version": TOOL_VERSION,
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": "pass" if not errors else "fail",
        "sample_count": sample_count,
        "rejection_count": rejection_count,
        "error_count": len(errors),
        "errors": errors,
    }
    report["content_sha256"] = compact_hash(report)
    target = report_path or root / "reports" / "quality_report.json"
    output_sha = write_json(target, report)
    return {**report, "report": str(target), "sha256": output_sha}


def _load_samples(manifest: Mapping[str, Any], root: Path) -> list[Mapping[str, Any]]:
    samples: list[Mapping[str, Any]] = []
    for relative, metadata in sorted(manifest["files"].items()):
        if isinstance(metadata, Mapping) and metadata.get("schema_version") == SAMPLE_SCHEMA:
            records, _ = read_jsonl(root / relative)
            samples.extend(records)
    return samples


def audit_leakage(manifest_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest, root = _dataset_files(manifest_path)
    samples = _load_samples(manifest, root)
    dimensions = (
        "source_lineage_id",
        "implementation_family_id",
        "contract_hash",
        "task_id",
    )
    memberships: dict[str, dict[str, set[str]]] = {
        dimension: defaultdict(set) for dimension in dimensions
    }
    duplicate_patches: dict[str, list[str]] = defaultdict(list)
    duplicate_normalized: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        provenance = sample.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        split = str(sample.get("split"))
        for dimension in dimensions:
            value = provenance.get(dimension)
            if value:
                memberships[dimension][str(value)].add(split)
        duplicate_patches[str(provenance.get("patch_hash"))].append(sample["sample_id"])
        duplicate_normalized[str(provenance.get("normalized_text_hash"))].append(
            sample["sample_id"]
        )
    overlaps = []
    for dimension in dimensions:
        for value, splits in sorted(memberships[dimension].items()):
            if len(splits) > 1:
                overlaps.append(
                    {"dimension": dimension, "value": value, "splits": sorted(splits)}
                )
    duplicates = [
        {"kind": "exact_patch", "hash": digest, "sample_ids": ids}
        for digest, ids in sorted(duplicate_patches.items())
        if len(ids) > 1
    ] + [
        {"kind": "normalized_text", "hash": digest, "sample_ids": ids}
        for digest, ids in sorted(duplicate_normalized.items())
        if len(ids) > 1
    ]
    protected_hits: list[dict[str, Any]] = []
    input_manifest = load_json(Path(str(manifest["input_manifest"])))
    terms = input_manifest.get("protected_held_out_terms") or []
    if not isinstance(terms, list):
        terms = []
    for sample in samples:
        if sample.get("split") == "held_out":
            continue
        prompt = json.dumps(sample.get("input"), sort_keys=True, ensure_ascii=False)
        for term in terms:
            if isinstance(term, str) and term and term in prompt:
                protected_hits.append({"sample_id": sample["sample_id"], "term": term})
    report = {
        "schema_version": "geak_sft_leakage_report_v1",
        "tool_version": TOOL_VERSION,
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": "pass" if not overlaps and not duplicates and not protected_hits else "fail",
        "source_lineage_split_overlap_count": sum(
            item["dimension"] == "source_lineage_id" for item in overlaps
        ),
        "overlap_count": len(overlaps),
        "duplicate_count": len(duplicates),
        "protected_prompt_hit_count": len(protected_hits),
        "overlaps": overlaps,
        "duplicates": duplicates,
        "protected_prompt_hits": protected_hits,
    }
    report["content_sha256"] = compact_hash(report)
    target = report_path or root / "reports" / "leakage_report.json"
    output_sha = write_json(target, report)
    return {**report, "report": str(target), "sha256": output_sha}


def coverage_report(manifest_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest, root = _dataset_files(manifest_path)
    samples = _load_samples(manifest, root)
    input_manifest = load_json(Path(str(manifest["input_manifest"])))
    task_families = input_manifest.get("task_families") or {}
    if not isinstance(task_families, Mapping):
        task_families = {}
    raw_pinned_keys = input_manifest.get("pinned_sample_keys") or []
    pinned_keys = {
        (str(value[0]), str(value[1]))
        for value in raw_pinned_keys
        if isinstance(value, list) and len(value) == 2
    }
    keys = {
        "language": ("provenance", "language"),
        "lane": ("provenance", "lane"),
        "architecture": ("provenance", "gpu"),
        "gpu_sku": ("provenance", "gpu_sku"),
        "task_type": ("task_type",),
        "operator": ("provenance", "operator"),
        "top10_family": ("provenance", "top10_family"),
        "dtype_format": ("provenance", "dtype_format"),
        "shape_regime": ("provenance", "shape_regime"),
        "source_lineage": ("provenance", "source_lineage_id"),
        "implementation_family": ("provenance", "implementation_family_id"),
        "split": ("split",),
    }
    coverage: dict[str, dict[str, int]] = {}
    for name, path in keys.items():
        counts: Counter[str] = Counter()
        for sample in samples:
            value: Any = sample
            for part in path:
                value = value.get(part) if isinstance(value, Mapping) else None
            if name == "top10_family" and value in (None, ""):
                task_id = _nested(sample, ("provenance", "task_id"))
                value = task_families.get(str(task_id))
            counts[str(value) if value not in (None, "") else "<missing>"] += 1
        coverage[name] = dict(sorted(counts.items()))
    post_checkpoint_families: Counter[str] = Counter()
    for sample in samples:
        sample_key = (
            str(_nested(sample, ("provenance", "run_id")) or ""),
            str(_nested(sample, ("provenance", "candidate_id")) or ""),
        )
        if sample_key in pinned_keys:
            continue
        family = _nested(sample, ("provenance", "top10_family"))
        if family in (None, ""):
            task_id = _nested(sample, ("provenance", "task_id"))
            family = task_families.get(str(task_id))
        post_checkpoint_families[
            str(family) if family not in (None, "") else "<missing>"
        ] += 1
    coverage["post_checkpoint_top10_family"] = dict(
        sorted(post_checkpoint_families.items())
    )
    targets = input_manifest.get("coverage_targets") or {}
    if not isinstance(targets, Mapping):
        targets = {}
    gaps: list[dict[str, Any]] = []
    minimum_samples = targets.get("sample_count", 1)
    if not isinstance(minimum_samples, int) or minimum_samples < 1:
        minimum_samples = 1
    if len(samples) < minimum_samples:
        gaps.append(
            {
                "dimension": "sample_count",
                "expected_minimum": minimum_samples,
                "actual": len(samples),
            }
        )
    for dimension, expected_value in sorted(targets.items()):
        if dimension == "sample_count" or not isinstance(expected_value, Mapping):
            continue
        actual_counts = coverage.get(str(dimension), {})
        for value, minimum in sorted(expected_value.items()):
            if isinstance(minimum, int) and actual_counts.get(str(value), 0) < minimum:
                gaps.append(
                    {
                        "dimension": str(dimension),
                        "value": str(value),
                        "expected_minimum": minimum,
                        "actual": actual_counts.get(str(value), 0),
                    }
                )
    token_by_split: Counter[str] = Counter()
    missing_tokens = 0
    for sample in samples:
        value = _nested(sample, ("provenance", "valid_assistant_tokens"))
        if isinstance(value, int):
            token_by_split[str(sample["split"])] += value
        else:
            missing_tokens += 1
    report = {
        "schema_version": "geak_sft_coverage_report_v1",
        "tool_version": TOOL_VERSION,
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": "pass" if not gaps else "fail",
        "sample_count": len(samples),
        "coverage": coverage,
        "coverage_targets": targets,
        "gap_count": len(gaps),
        "gaps": gaps,
        "valid_assistant_tokens": {
            "total": sum(token_by_split.values()),
            "by_split": dict(sorted(token_by_split.items())),
            "samples_missing_exact_count": missing_tokens,
        },
    }
    report["content_sha256"] = compact_hash(report)
    target = report_path or root / "reports" / "coverage_report.json"
    output_sha = write_json(target, report)
    return {**report, "report": str(target), "sha256": output_sha}


def _emit(result: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


def build_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic GEAK Kernel SFT JSONL")
    parser.add_argument("--input-manifest", "--manifest", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_dataset(args.input_manifest, args.output_root)
    except DatasetError as exc:
        parser.error(str(exc))
    _emit(result)
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GEAK Kernel SFT artifacts")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_dataset(args.manifest, args.report)
    except DatasetError as exc:
        parser.error(str(exc))
    _emit(result)
    return 0 if result["status"] == "pass" else 1


def leakage_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit GEAK Kernel SFT split leakage")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_leakage(args.manifest, args.report)
    except DatasetError as exc:
        parser.error(str(exc))
    _emit(result)
    return 0 if result["status"] == "pass" else 1


def coverage_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report GEAK Kernel SFT coverage")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        result = coverage_report(args.manifest, args.report)
    except DatasetError as exc:
        parser.error(str(exc))
    _emit(result)
    return 0 if result["status"] == "pass" else 1
