"""Eval-only Kernel held-out inventory, materialization, and audit helpers.

This module deliberately does not import the training flow or SFT collector.
Incomplete inventories produce a deficit report and never a partial task set.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "geak_kernel_held_out_eval_v1"
INVENTORY_SCHEMA_VERSION = "geak_kernel_held_out_inventory_v1"
TASK_TYPES = (
    "cold_start",
    "profile_guided",
    "direction_conditioned",
    "error_recovery",
    "regression_balance",
)
TASK_TYPE_QUOTAS = dict(zip(TASK_TYPES, (18, 18, 54, 18, 12)))
LANE_QUOTAS = {"hip_gfx942": 60, "triton_gfx942": 60}
SOURCE_SUITE_QUOTAS = {
    "geak_native": 40,
    "aiter_derived": 40,
    "adversarial_boundary": 40,
}
TARGET_COUNT = 120
FORBIDDEN_KEYS = {
    "candidate_sources",
    "output.patch",
    "patch_target",
    "reference_patch",
    "sft_events",
    "target_patch",
    "trajectory",
}
REQUIRED_TASK_KEYS = {
    "task_id",
    "source_lineage_id",
    "contract_family_id",
    "implementation_family_id",
    "lane",
    "source_suite",
    "contract",
    "initial_source",
    "protected",
    "environment_hash",
}
SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HeldOutError(ValueError):
    """Raised when held-out safety or schema requirements are violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree including normalized relative names."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise HeldOutError(f"path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HeldOutError(f"{path}:{line_number}: expected JSON object")
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _walk_keys(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            yield str(key)
            yield dotted
            yield from _walk_keys(child, dotted)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child, prefix)


def validate_eval_task(task: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TASK_KEYS - set(task))
    if missing:
        raise HeldOutError(f"{task.get('task_id', '<unknown>')}: missing {missing}")
    if task.get("schema_version") != SCHEMA_VERSION:
        raise HeldOutError(f"{task['task_id']}: invalid schema_version")
    if not SAFE_TASK_ID.fullmatch(str(task["task_id"])):
        raise HeldOutError(f"{task['task_id']!r}: unsafe task_id")
    if task.get("sft_enabled") is not False:
        raise HeldOutError(f"{task['task_id']}: sft_enabled must be false")
    if task.get("split") != "held_out" or task.get("eval_only") is not True:
        raise HeldOutError(f"{task['task_id']}: must be held_out eval-only")
    forbidden = sorted(FORBIDDEN_KEYS.intersection(_walk_keys(task)))
    if forbidden:
        raise HeldOutError(f"{task['task_id']}: forbidden fields: {forbidden}")
    if task["lane"] not in LANE_QUOTAS:
        raise HeldOutError(f"{task['task_id']}: unsupported lane {task['lane']!r}")
    if task["source_suite"] not in SOURCE_SUITE_QUOTAS:
        raise HeldOutError(f"{task['task_id']}: unsupported source_suite")
    if task.get("task_type") not in TASK_TYPE_QUOTAS:
        raise HeldOutError(f"{task['task_id']}: invalid task_type")
    initial = task["initial_source"]
    if not isinstance(initial, Mapping) or not initial.get("path") or not initial.get("sha256"):
        raise HeldOutError(f"{task['task_id']}: initial_source needs path and sha256")
    if not SHA256.fullmatch(str(initial["sha256"])):
        raise HeldOutError(f"{task['task_id']}: invalid initial source hash")
    protected = task["protected"]
    if not isinstance(protected, Mapping):
        raise HeldOutError(f"{task['task_id']}: invalid protected refs")
    for name in ("harness", "oracle"):
        ref = protected.get(name)
        if not isinstance(ref, Mapping) or not ref.get("sha256"):
            raise HeldOutError(f"{task['task_id']}: protected {name} hash missing")
        if not SHA256.fullmatch(str(ref["sha256"])):
            raise HeldOutError(f"{task['task_id']}: invalid protected {name} hash")
        if "content" in ref:
            raise HeldOutError(f"{task['task_id']}: protected {name} content is forbidden")
    if not SHA256.fullmatch(str(task["environment_hash"])):
        raise HeldOutError(f"{task['task_id']}: invalid environment hash")


def v4_inventory(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_refs = 0
    candidate_lane_slots = 0
    units: set[tuple[str, str]] = set()
    lane_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    for group in groups:
        candidates = {str(item) for item in group.get("candidate_ids", [])}
        lanes = {str(item) for item in group.get("target_lanes", [])}
        lineage = str(group.get("source_lineage_id", ""))
        candidate_refs += len(candidates)
        candidate_lane_slots += len(candidates) * len(lanes)
        if lineage.startswith("aiter:"):
            suite = "aiter_derived"
        elif lineage.startswith("geak:"):
            suite = "geak_native"
        else:
            suite = "unclassified"
        for lane in lanes:
            units.add((lineage, lane))
            lane_counts[lane] += 1
            suite_counts[suite] += 1
    return {
        "lineage_count": len({lineage for lineage, _ in units}),
        "candidate_ref_count": candidate_refs,
        "candidate_lane_slot_count": candidate_lane_slots,
        "lineage_lane_unit_count": len(units),
        "lane_available": dict(sorted(lane_counts.items())),
        "source_suite_available": dict(sorted(suite_counts.items())),
    }


def deterministic_assign(
    records: Sequence[Mapping[str, Any]],
    *,
    target_count: int = TARGET_COUNT,
) -> list[dict[str, Any]]:
    """Select and assign a complete quota-balanced set using stable hashes."""
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = (str(record.get("source_lineage_id")), str(record.get("lane")))
        if key in deduped:
            raise HeldOutError(f"duplicate lineage/lane inventory unit: {key}")
        deduped[key] = record

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in deduped.values():
        cells[(str(record.get("lane")), str(record.get("source_suite")))].append(record)
    for values in cells.values():
        values.sort(key=lambda item: (stable_hash(item), str(item)))

    # With two lanes and three suites, enumerate the HIP row. The Triton row is
    # then fixed by suite quotas. This avoids an order-dependent greedy dead end.
    suites = tuple(SOURCE_SUITE_QUOTAS)
    allocations: list[tuple[int, int, int]] = []
    for first in range(SOURCE_SUITE_QUOTAS[suites[0]] + 1):
        for second in range(SOURCE_SUITE_QUOTAS[suites[1]] + 1):
            third = LANE_QUOTAS["hip_gfx942"] - first - second
            hip = (first, second, third)
            if not 0 <= third <= SOURCE_SUITE_QUOTAS[suites[2]]:
                continue
            triton = tuple(
                SOURCE_SUITE_QUOTAS[suite] - count
                for suite, count in zip(suites, hip)
            )
            if all(
                len(cells[("hip_gfx942", suite)]) >= count
                and len(cells[("triton_gfx942", suite)]) >= other
                for suite, count, other in zip(suites, hip, triton)
            ):
                allocations.append(hip)
    if not allocations:
        raise HeldOutError("inventory cannot satisfy lane and source-suite quotas")
    hip_allocation = min(allocations, key=lambda value: stable_hash(value))
    selected: list[dict[str, Any]] = []
    for suite, hip_count in zip(suites, hip_allocation):
        selected.extend(cells[("hip_gfx942", suite)][:hip_count])
        triton_count = SOURCE_SUITE_QUOTAS[suite] - hip_count
        selected.extend(cells[("triton_gfx942", suite)][:triton_count])
    if len(selected) != target_count:
        raise HeldOutError("selected task count does not match target")

    type_slots = [
        task_type
        for task_type in TASK_TYPES
        for _ in range(TASK_TYPE_QUOTAS[task_type])
    ]
    # A second fixed hash avoids coupling assignment to inventory append order.
    selected.sort(key=lambda item: stable_hash({"assignment": item["task_id"]}))
    materialized = []
    for record, task_type in zip(selected, type_slots):
        task = {
            **record,
            "schema_version": SCHEMA_VERSION,
            "split": "held_out",
            "eval_only": True,
            "sft_enabled": False,
            "task_type": task_type,
        }
        validate_eval_task(task)
        materialized.append(task)
    return sorted(materialized, key=lambda item: str(item["task_id"]))


def inventory_deficit(
    groups: Sequence[Mapping[str, Any]],
    complete_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    evidence = v4_inventory(groups)
    keys = {
        (str(item.get("source_lineage_id")), str(item.get("lane")))
        for item in complete_records
    }
    lane = Counter(str(item.get("lane")) for item in complete_records)
    suite = Counter(str(item.get("source_suite")) for item in complete_records)
    complete = len(keys)
    available = complete if complete_records else evidence["lineage_lane_unit_count"]
    lane_available = lane if complete_records else Counter(evidence["lane_available"])
    suite_available = (
        suite if complete_records else Counter(evidence["source_suite_available"])
    )
    return {
        "schema_version": "geak_kernel_held_out_inventory_deficit_v1",
        "status": "ready" if complete >= TARGET_COUNT else "not_materialized",
        "target_task_count": TARGET_COUNT,
        "complete_eval_task_units": complete,
        "reserved_independent_units": evidence["lineage_lane_unit_count"],
        "task_deficit": max(0, TARGET_COUNT - available),
        "materializable_task_deficit": max(0, TARGET_COUNT - complete),
        "reservation_evidence": evidence,
        "lane_deficit": {
            key: max(0, quota - lane_available[key])
            for key, quota in LANE_QUOTAS.items()
        },
        "materializable_lane_deficit": {
            key: max(0, quota - lane[key]) for key, quota in LANE_QUOTAS.items()
        },
        "source_suite_deficit": {
            key: max(0, quota - suite_available[key])
            for key, quota in SOURCE_SUITE_QUOTAS.items()
        },
        "materializable_source_suite_deficit": {
            key: max(0, quota - suite[key])
            for key, quota in SOURCE_SUITE_QUOTAS.items()
        },
        "required_task_type_quota": TASK_TYPE_QUOTAS,
        "reason": (
            "v4 groups reserve lineages but do not provide complete contracts, initial "
            "sources, and protected harness/oracle hash references"
            if not complete_records
            else (
                "complete inventory is ready for deterministic quota assignment"
                if complete >= TARGET_COUNT
                else "complete inventory does not satisfy all hard quotas"
            )
        ),
    }


def protected_terms(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    terms: set[str] = set()
    for task in tasks:
        for key in ("task_id", "source_lineage_id", "contract_family_id"):
            value = task.get(key)
            if value:
                terms.add(str(value))
        terms.update(str(value) for value in task.get("protected_symbols", []) if value)
    return sorted(terms)


def protected_terms_from_groups(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    terms: set[str] = set()
    for group in groups:
        for key in ("source_lineage_id",):
            if group.get(key):
                terms.add(str(group[key]))
        for key in ("candidate_ids", "contract_family_ids"):
            terms.update(str(value) for value in group.get(key, []) if value)
    return sorted(terms)


def _sample_values(sample: Mapping[str, Any]) -> tuple[set[str], set[str], str]:
    provenance = sample.get("provenance", {})
    seed = sample.get("input", {}).get("contract", {}).get("provenance", {}).get(
        "case_seed", {}
    )
    identities = {
        str(value)
        for value in (
            provenance.get("source_lineage_id"),
            provenance.get("contract_hash"),
            seed.get("source_lineage_id"),
            seed.get("contract_family_id"),
        )
        if value
    }
    prompt = json.dumps(sample.get("input", {}), sort_keys=True, ensure_ascii=False)
    return identities, set(SYMBOL.findall(prompt)), prompt


def leakage_audit(
    tasks: Sequence[Mapping[str, Any]],
    train_jsonl: Path,
    dev_jsonl: Path | None = None,
    *,
    expected_train_count: int | None = None,
) -> dict[str, Any]:
    held_identities: set[str] = set()
    held_symbols: set[str] = set()
    for task in tasks:
        held_identities.update(
            str(task[key])
            for key in ("source_lineage_id", "contract_family_id")
            if task.get(key)
        )
        held_identities.add(stable_hash(task["contract"]))
        held_symbols.update(str(value) for value in task.get("protected_symbols", []))

    hits: list[dict[str, Any]] = []
    checked = Counter()
    for split, path in (("train", train_jsonl), ("dev", dev_jsonl)):
        if path is None:
            continue
        for sample in iter_jsonl(path):
            checked[split] += 1
            identities, symbols, _ = _sample_values(sample)
            for value in sorted(held_identities & identities):
                hits.append({"split": split, "sample_id": sample.get("sample_id"), "kind": "identity", "value": value})
            for value in sorted(held_symbols & symbols):
                hits.append({"split": split, "sample_id": sample.get("sample_id"), "kind": "symbol", "value": value})
    inventory_ok = expected_train_count is None or checked["train"] == expected_train_count
    return {
        "schema_version": "geak_kernel_held_out_leakage_audit_v1",
        "status": "pass" if not hits and inventory_ok else "fail",
        "checked_records": dict(checked),
        "expected_train_records": expected_train_count,
        "train_inventory_valid": inventory_ok,
        "lineage_contract_symbol_hit_count": len(hits),
        "hits": hits,
    }


def reservation_leakage_audit(
    groups: Sequence[Mapping[str, Any]],
    train_jsonl: Path,
    dev_jsonl: Path | None = None,
    *,
    expected_train_count: int = 2000,
) -> dict[str, Any]:
    identities = {
        str(value)
        for group in groups
        for value in (
            [group.get("source_lineage_id")]
            + list(group.get("contract_family_ids", []))
        )
        if value
    }
    symbols = {
        str(value)
        for group in groups
        for value in group.get("candidate_ids", [])
        if value
    }
    checked = Counter()
    hits: list[dict[str, Any]] = []
    for split, path in (("train", train_jsonl), ("dev", dev_jsonl)):
        if path is None:
            continue
        for sample in iter_jsonl(path):
            checked[split] += 1
            sample_identities, _, prompt = _sample_values(sample)
            for value in sorted(identities & sample_identities):
                hits.append(
                    {
                        "split": split,
                        "sample_id": sample.get("sample_id"),
                        "kind": "lineage_or_contract",
                        "value": value,
                    }
                )
            for value in sorted(symbol for symbol in symbols if symbol in prompt):
                hits.append(
                    {
                        "split": split,
                        "sample_id": sample.get("sample_id"),
                        "kind": "symbol",
                        "value": value,
                    }
                )
    inventory_ok = checked["train"] == expected_train_count
    return {
        "schema_version": "geak_kernel_held_out_reservation_leakage_audit_v1",
        "status": "pass" if not hits and inventory_ok else "fail",
        "scope": "v4_reservations",
        "checked_records": dict(checked),
        "expected_train_records": expected_train_count,
        "train_inventory_valid": inventory_ok,
        "lineage_contract_symbol_hit_count": len(hits),
        "hits": hits,
    }


def build_control_root(
    groups_path: Path,
    output_root: Path,
    inventory_path: Path | None = None,
    train_jsonl: Path | None = None,
    dev_jsonl: Path | None = None,
) -> dict[str, Any]:
    groups = read_jsonl(groups_path)
    records = read_jsonl(inventory_path) if inventory_path and inventory_path.exists() else []
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_dir = output_root / "inventory"
    inventory_dir.mkdir(exist_ok=True)
    frozen_groups = inventory_dir / "held_out_groups.v4.jsonl"
    if groups_path.resolve() != frozen_groups.resolve():
        shutil.copyfile(groups_path, frozen_groups)
    append_path = inventory_dir / "task_candidates.jsonl"
    if inventory_path:
        if inventory_path.resolve() != append_path.resolve():
            shutil.copyfile(inventory_path, append_path)
    elif not append_path.exists():
        append_path.touch()

    deficit = inventory_deficit(groups, records)
    terms_path = output_root / "protected_held_out_terms.json"
    write_json(terms_path, protected_terms_from_groups(groups))
    deficit["inputs"] = {
        "frozen_v4_groups": {
            "path": str(frozen_groups.relative_to(output_root)),
            "sha256": sha256_file(frozen_groups),
        },
        "append_safe_candidates": str(append_path.relative_to(output_root)),
        "protected_held_out_terms": {
            "path": str(terms_path.relative_to(output_root)),
            "sha256": sha256_file(terms_path),
        },
    }
    write_json(output_root / "inventory_deficit.json", deficit)
    if deficit["status"] != "ready":
        if train_jsonl:
            report = reservation_leakage_audit(groups, train_jsonl, dev_jsonl)
            write_json(output_root / "reports" / "leakage_report.json", report)
            if report["status"] != "pass":
                raise HeldOutError("v4 reservation leakage audit failed")
        # Never leave a stale partial materialization after an inventory regression.
        tasks_path = output_root / "tasks" / "kernel.jsonl"
        if tasks_path.exists():
            tasks_path.unlink()
        write_checksums(output_root)
        return deficit

    tasks = deterministic_assign(records)
    tasks_path = output_root / "tasks" / "kernel.jsonl"
    write_jsonl(tasks_path, tasks)
    terms = sorted(
        set(protected_terms(tasks)).union(protected_terms_from_groups(groups))
    )
    write_json(output_root / "protected_held_out_terms.json", terms)
    deficit["inputs"]["protected_held_out_terms"]["sha256"] = sha256_file(
        output_root / "protected_held_out_terms.json"
    )
    write_json(output_root / "inventory_deficit.json", deficit)
    if train_jsonl:
        report = leakage_audit(
            tasks, train_jsonl, dev_jsonl, expected_train_count=2000
        )
        write_json(output_root / "reports" / "leakage_report.json", report)
        if report["status"] != "pass":
            tasks_path.unlink(missing_ok=True)
            raise HeldOutError("held-out leakage audit failed")
    manifest = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "status": "materialized",
        "task_count": len(tasks),
        "tasks": {
            "path": str(tasks_path.relative_to(output_root)),
            "sha256": sha256_file(tasks_path),
        },
        "protected_held_out_terms": {
            "path": "protected_held_out_terms.json",
            "sha256": sha256_file(output_root / "protected_held_out_terms.json"),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    write_checksums(output_root)
    return manifest


def run_evaluation(
    tasks_path: Path,
    output_root: Path,
    evaluator: Sequence[str],
) -> dict[str, Any]:
    """Run a frozen external evaluator once per task and store hash-only receipts."""
    tasks = read_jsonl(tasks_path)
    if len(tasks) != TARGET_COUNT:
        raise HeldOutError(f"expected exactly {TARGET_COUNT} tasks, found {len(tasks)}")
    output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for task in tasks:
        validate_eval_task(task)
        command = [part.format(task_id=task["task_id"]) for part in evaluator]
        result = subprocess.run(
            command,
            input=json.dumps(task, sort_keys=True),
            text=True,
            capture_output=True,
            check=False,
        )
        receipt = {
            "task_id": task["task_id"],
            "returncode": result.returncode,
            "stdout_sha256": sha256_bytes(result.stdout.encode()),
            "stderr_sha256": sha256_bytes(result.stderr.encode()),
        }
        receipts.append(receipt)
    path = output_root / "receipts" / "kernel.jsonl"
    write_jsonl(path, receipts)
    report = {
        "schema_version": "geak_kernel_held_out_run_v1",
        "eval_only": True,
        "sft_enabled": False,
        "task_count": len(tasks),
        "passed": sum(item["returncode"] == 0 for item in receipts),
        "receipts": {"path": "receipts/kernel.jsonl", "sha256": sha256_file(path)},
    }
    write_json(output_root / "report.json", report)
    write_checksums(output_root)
    return report


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != checksum_path:
                handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
