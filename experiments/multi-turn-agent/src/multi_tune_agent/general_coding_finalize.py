"""Fail-closed finalization audit for General Coding Replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .general_coding_replay import (
    canonical_json,
    deduplicate_verified,
    select_replay,
    sha256_file,
    write_json,
    write_jsonl,
)

DIMENSIONS = {
    "repository": ("upstream_repository", "source_repo", "repository"),
    "problem": ("problem_id", "task_id"),
    "lineage": ("source_lineage_id",),
    "base_commit": ("base_commit", "source_sha"),
    "normalized_patch": ("normalized_patch_hash", "patch_hash"),
    "test_patch": ("test_patch_sha256", "test_set_hash", "source_test_sha256"),
}


def _walk_values(value: Any, keys: set[str]) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, (str, int)) and str(item):
                yield str(item)
            yield from _walk_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, keys)


def _normalize_repository(value: str) -> str:
    value = value.strip().lower().removesuffix(".git").rstrip("/")
    value = value.replace("git@github.com:", "https://github.com/")
    return value


def _dimension_values(row: Mapping[str, Any], dimension: str) -> set[str]:
    values = set(_walk_values(row, set(DIMENSIONS[dimension])))
    if dimension == "repository":
        values = {_normalize_repository(value) for value in values}
    return {value for value in values if value}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_verified(control_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(control_root.glob("**/verified.json"))
    rows = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["coding_task_type"] = row.get("coding_task_type") or row.get(
            "primary_task_type"
        )
        row["source_lineage_id"] = row.get("source_lineage_id") or (
            f"{row.get('dataset_id')}@{row.get('dataset_revision')}:"
            f"{row.get('problem_id')}"
        )
        row["_verified_receipt_path"] = path.relative_to(control_root).as_posix()
        rows.append(row)
    return rows, paths


def _receipt_audit(
    control_root: Path, rows: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    failures = []
    audited = []
    receipt_paths: list[Path] = []
    for raw in rows:
        row = dict(raw)
        verified_path = control_root / str(row["_verified_receipt_path"])
        receipts = []
        errors = []
        for field in ("target_test_receipt", "fresh_verify_receipt"):
            name = row.get(field)
            path = verified_path.parent / str(name or "")
            if not name or not path.is_file():
                errors.append(f"missing_{field}")
                continue
            receipt_paths.append(path)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipts.append(receipt)
            if not receipt.get("targeted_tests", {}).get("ok"):
                errors.append(f"{field}_targeted_tests")
            if not receipt.get("regression_tests", {}).get("ok"):
                errors.append(f"{field}_regression_tests")
            if receipt.get("network") not in {"none", "disabled_linux_namespace"}:
                errors.append(f"{field}_network")
            visibility = receipt.get("gpu_visibility")
            gpu_disabled = (
                not any(visibility.values())
                if isinstance(visibility, Mapping)
                else visibility == "disabled"
            )
            if not gpu_disabled or receipt.get("gpu_devices_mounted") is True:
                errors.append(f"{field}_gpu")
        identities = [
            receipt.get("container_id")
            or receipt.get("workspace")
            or receipt.get("workspace_path")
            for receipt in receipts
        ]
        if len(receipts) != 2 or not all(identities) or identities[0] == identities[1]:
            errors.append("two_distinct_fresh_replays")
        for field in (
            "case_id",
            "source_id",
            "primary_language",
            "coding_task_type",
            "upstream_repository",
            "base_commit",
            "problem_id",
            "source_lineage_id",
            "normalized_patch_hash",
            "test_set_hash",
            "license_spdx",
        ):
            if not row.get(field):
                errors.append(f"missing_{field}")
        if row.get("status") != "verified":
            errors.append("status_not_verified")
        if row.get("network_policy") != "disabled" or row.get("gpu_required") is not False:
            errors.append("execution_policy")
        if errors:
            failures.append({"case_id": row.get("case_id"), "errors": sorted(set(errors))})
        else:
            audited.append(row)
    report = {
        "schema_version": "general_coding_replay_quality_v1",
        "status": "pass" if not failures else "fail",
        "verified_receipts_discovered": len(list(rows)) if isinstance(rows, list) else None,
        "two_fresh_replay_passed": len(audited),
        "failed": len(failures),
        "failures": failures,
        "receipt_evidence_policy": (
            "target-receipt.json and fresh-receipt.json must both pass in distinct "
            "isolated workspace/container identities"
        ),
    }
    return report, audited, receipt_paths


def _domain_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {
        dimension: set().union(
            *(_dimension_values(row, dimension) for row in rows)
        )
        if rows
        else set()
        for dimension in DIMENSIONS
    }


def _registry_domain(component: Mapping[str, Any]) -> dict[str, set[str]]:
    aliases = {
        "repository": ("repositories",),
        "problem": ("problem_ids", "task_ids"),
        "lineage": ("source_lineages",),
        "base_commit": ("base_commits",),
        "normalized_patch": ("normalized_patch_hashes",),
        "test_patch": ("test_patch_hashes", "test_set_hashes", "oracle_hashes"),
    }
    result = {}
    for dimension, keys in aliases.items():
        values = set()
        for key in keys:
            raw = component.get(key, [])
            if isinstance(raw, list):
                values.update(str(item) for item in raw if item)
        if dimension == "repository":
            values = {_normalize_repository(value) for value in values}
        result[dimension] = values
    return result


def _overlap_report(
    pool: list[dict[str, Any]],
    domains: Mapping[str, tuple[int, Mapping[str, set[str]]]],
) -> dict[str, Any]:
    results = {}
    total = 0
    for name, (rows, index) in domains.items():
        matches = []
        by_dimension = Counter()
        for row in pool:
            found = {}
            for dimension in DIMENSIONS:
                values = _dimension_values(row, dimension) & set(index[dimension])
                if values:
                    found[dimension] = sorted(values)
                    by_dimension[dimension] += 1
            if found:
                matches.append({"case_id": row["case_id"], "dimensions": found})
        total += len(matches)
        results[name] = {
            "rows": rows,
            "overlap_cases": len(matches),
            "overlap_by_dimension": dict(sorted(by_dimension.items())),
            "matches": matches,
        }
    return {
        "schema_version": "general_coding_replay_cross_domain_leakage_v1",
        "status": "pass" if total == 0 else "fail",
        "pool_rows_audited": len(pool),
        "dimensions": list(DIMENSIONS),
        "domains": results,
        "total_domain_case_matches": total,
    }


def _counter(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _resolve_training_artifact(
    control_root: Path, row: Mapping[str, Any], field: str
) -> Path:
    raw = str(row.get(field, ""))
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [control_root / path]
    receipt = row.get("_verified_receipt_path")
    if receipt and not path.is_absolute():
        candidates.append((control_root / str(receipt)).parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{row.get('case_id')}: missing training artifact {field}={raw}")


def _materialize_training_rows(
    control_root: Path, selected: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    token_counts_path = control_root / "token-counts.json"
    token_counts = (
        json.loads(token_counts_path.read_text(encoding="utf-8"))
        if token_counts_path.is_file()
        else {}
    )
    replay_counts = token_counts.get("replay_assistant_loss_tokens", {})
    kernel_tokens = token_counts.get("kernel_assistant_loss_tokens")
    materialized = []
    for raw in selected:
        row = dict(raw)
        problem = _resolve_training_artifact(
            control_root, row, "problem_statement_path"
        ).read_text(encoding="utf-8")
        patch = _resolve_training_artifact(
            control_root, row, "target_patch_path"
        ).read_text(encoding="utf-8")
        source_schema = row.get("schema_version")
        row.update(
            {
                "schema_version": "general_coding_replay_v1",
                "source_schema_version": source_schema,
                "sample_id": str(row["case_id"]),
                "sample_domain": "general_coding",
                "task_type": "general_coding_replay",
                "split": "train",
                "input": {"prompt": problem},
                "output": {"patch": patch},
                "provenance": {
                    "dataset_id": row["dataset_id"],
                    "dataset_revision": row["dataset_revision"],
                    "source_id": row["source_id"],
                    "source_lineage_id": row["source_lineage_id"],
                    "language": row["primary_language"],
                    "base_commit": row["base_commit"],
                    "verified_receipt": row["_verified_receipt_path"],
                },
            }
        )
        count = replay_counts.get(str(row["case_id"]))
        if isinstance(count, int) and count > 0:
            row["assistant_loss_tokens"] = count
        if isinstance(kernel_tokens, int) and kernel_tokens > 0:
            row["kernel_assistant_loss_tokens"] = kernel_tokens
        materialized.append(row)
    return materialized


def _derive_exact_quota_v2(
    rows: Iterable[Mapping[str, Any]],
    target: int,
    *,
    repository_cap: int = 70,
    lineage_cap: int = 2,
    source_fraction: float = 0.70,
) -> dict[str, Any]:
    """Freeze achievable exact quotas from a leakage-clean, deduplicated pool."""
    pool, _duplicates = deduplicate_verified(rows)
    ordered = sorted(pool, key=lambda row: str(row["case_id"]))
    if len(ordered) < target:
        raise RuntimeError(
            f"cannot derive quota v2: deduplicated pool {len(ordered)} < target {target}"
        )

    specs: list[tuple[list[int], int, int]] = [
        (list(range(len(ordered))), target, target)
    ]
    for field, cap in (
        ("upstream_repository", repository_cap),
        ("source_lineage_id", lineage_cap),
    ):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(ordered):
            groups[str(row.get(field))].append(index)
        specs.extend((indices, 0, cap) for indices in groups.values())

    source_cap = int(target * source_fraction)
    sources: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        sources[str(row["source_id"])].append(index)
    specs.extend((indices, 1, source_cap) for indices in sources.values())

    for field in ("primary_language", "coding_task_type"):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(ordered):
            groups[str(row[field])].append(index)
        specs.extend((indices, 1, target) for indices in groups.values())

    matrix = lil_matrix((len(specs), len(ordered)), dtype=float)
    lower = np.empty(len(specs), dtype=float)
    upper = np.empty(len(specs), dtype=float)
    for row_index, (indices, minimum, maximum) in enumerate(specs):
        matrix[row_index, indices] = 1.0
        lower[row_index] = minimum
        upper[row_index] = maximum
    objective = np.array(
        [
            float(row.get("verified_runtime_seconds", 0))
            + (index + 1) / max(1, len(ordered)) * 1e-6
            for index, row in enumerate(ordered)
        ]
    )
    solution = milp(
        c=objective,
        integrality=np.ones(len(ordered)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 300},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"cannot derive quota v2: {solution.message}")
    selected = [
        row
        for row, chosen in zip(ordered, solution.x, strict=True)
        if chosen >= 0.5
    ]
    if len(selected) != target:
        raise RuntimeError(
            f"quota v2 derivation selected {len(selected)} rows, expected {target}"
        )

    source_language: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        source_language[str(row["source_id"])][str(row["primary_language"])] += 1
    return {
        "schema_version": "general_coding_replay_quota_v2",
        "train_rows": target,
        "derivation": (
            "exact distributions frozen from the leakage-clean, deduplicated, "
            "locally double-replayed pool"
        ),
        "source_targets": _counter(selected, "source_id"),
        "source_language_targets": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_language.items())
        },
        "language_targets": _counter(selected, "primary_language"),
        "task_type_targets": _counter(selected, "coding_task_type"),
        "caps": {
            "repository": repository_cap,
            "lineage": lineage_cap,
            "source_fraction": source_fraction,
        },
        "assistant_loss_token_share": {"min": 0.15, "max": 0.20},
    }


def finalize(
    control_root: Path,
    kernel_train: Path,
    kernel_dev: Path,
    kernel_held_out: Path,
    *,
    target: int = 500,
    derive_quota_v2: bool = False,
) -> dict[str, Any]:
    rows, verified_paths = _load_verified(control_root)
    quality, quality_rows, receipt_paths = _receipt_audit(control_root, rows)
    write_jsonl(control_root / "verified-pool.jsonl", quality_rows)

    registry = json.loads((control_root / "exclusion-registry.json").read_text())
    dev_rows = _load_jsonl(control_root / "general-coding-dev-reservations.jsonl")
    domains = {
        "kernel_train_2000": (
            len(_load_jsonl(kernel_train)),
            _domain_index(_load_jsonl(kernel_train)),
        ),
        "kernel_dev_200": (
            len(_load_jsonl(kernel_dev)),
            _domain_index(_load_jsonl(kernel_dev)),
        ),
        "kernel_held_out_120": (
            len(_load_jsonl(kernel_held_out)),
            _domain_index(_load_jsonl(kernel_held_out)),
        ),
        "general_coding_dev_80": (len(dev_rows), _domain_index(dev_rows)),
        "coding_held_out_40": (
            len(registry["general_coding_repository_held_out"].get("problem_ids", [])),
            _registry_domain(registry["general_coding_repository_held_out"]),
        ),
    }
    candidate_leakage = _overlap_report(quality_rows, domains)
    overlapping = {
        match["case_id"]
        for domain in candidate_leakage["domains"].values()
        for match in domain["matches"]
    }
    eligible = [row for row in quality_rows if row["case_id"] not in overlapping]

    quota_path = control_root / "quotas.yaml"
    if derive_quota_v2:
        v1_path = control_root / "quotas-v1.yaml"
        if not v1_path.exists():
            v1_path.write_text(quota_path.read_text(encoding="utf-8"), encoding="utf-8")
        derived = _derive_exact_quota_v2(eligible, target)
        quota_path.write_text(
            yaml.safe_dump(derived, sort_keys=False), encoding="utf-8"
        )
        (control_root / "quota-overrides.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "general_coding_replay_quota_override_v2",
                    "status": "approved_pool_constrained_exact_quota",
                    "reason": (
                        "The user authorized a v2 quota after locally double-replayed "
                        "MEnvData rows made the legacy source cells obsolete. Targets "
                        "are frozen from the leakage-clean pool; no rejected or "
                        "overlapping row is admitted."
                    ),
                    "supersedes": "quotas-v1.yaml",
                    "effective_quotas": {
                        key: derived[key]
                        for key in (
                            "source_targets",
                            "source_language_targets",
                            "language_targets",
                            "task_type_targets",
                            "caps",
                        )
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        write_json(
            control_root / "quota-v2-derivation.json",
            {
                "schema_version": "general_coding_replay_quota_v2_derivation",
                "status": "frozen",
                "target": target,
                "quality_eligible": len(quality_rows),
                "leakage_excluded": len(overlapping),
                "cross_domain_eligible": len(eligible),
                "deduplicated_eligible": len(deduplicate_verified(eligible)[0]),
                "quotas_sha256": sha256_file(quota_path),
                "caps": derived["caps"],
            },
        )
    quotas = yaml.safe_load(quota_path.read_text())
    override_path = control_root / "quota-overrides.yaml"
    evidence_path = control_root / "quota-override-evidence.json"
    evidence = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.is_file()
        else {}
    )
    cpp_targets = {
        source: languages["cpp"]
        for source, languages in quotas["source_language_targets"].items()
        if "cpp" in languages
    }
    cpp_verified = Counter(
        str(row["source_id"])
        for row in quality_rows
        if row["primary_language"] == "cpp"
    )
    evidence.update(
        {
            "schema_version": "general_coding_replay_quota_override_evidence_v1",
            "status": "verified_quota_deficits",
            "effective_quotas": {
                key: quotas[key]
                for key in (
                    "source_targets",
                    "source_language_targets",
                    "language_targets",
                    "task_type_targets",
                )
            },
            "admitted_cpp_sources": {
                source: {
                    "target": target_count,
                    "verified_receipts": cpp_verified[source],
                    "deficit": target_count - cpp_verified[source],
                }
                for source, target_count in cpp_targets.items()
            },
            "fallback": {
                "dataset_id": "ByteDance-Seed/Multi-SWE-RL",
                "revision": "9777648932daa214ba18c70c81e85821b5836f32",
                "effective_language_targets": {"cpp": 43, "rust": 50},
                "status": "locally_verified_quota_deficits",
                "verified_rows": sum(
                    1
                    for row in quality_rows
                    if row["source_id"] == "bytedance_multi_swe_rl_fallback"
                ),
            },
            "verified_pool_receipts": len(quality_rows),
            "override_sha256": sha256_file(override_path),
            "note": (
                "Targets admit only locally double-replayed rows; deficits are "
                "reported and are not backfilled with rejected or candidate-only rows."
            ),
        }
    )
    write_json(evidence_path, evidence)
    selected, selection = select_replay(eligible, quotas, target)
    selection.update(
        {
            "verified_json_receipts_discovered": len(rows),
            "quality_eligible": len(quality_rows),
            "cross_domain_eligible": len(eligible),
            "accepted_written": len(selected),
            "exact_target_gate": target,
            "blockers": []
            if selected
            else [
                f"verified eligible pool is {len(eligible)}, below exact target {target}",
                "exact source/language/task quotas are infeasible from current receipts",
            ],
        }
    )
    selected = _materialize_training_rows(control_root, selected)
    write_jsonl(control_root / "accepted.jsonl", selected)
    leakage = _overlap_report(selected, domains)
    write_json(control_root / "selection-report.json", selection)
    write_json(control_root / "quota-deficits.json", selection)
    write_json(control_root / "quality-report.json", quality)
    write_json(control_root / "candidate-leakage-exclusions.json", candidate_leakage)
    write_json(control_root / "leakage-report.json", leakage)
    write_json(control_root / "source-lineage-audit.json", leakage)

    deduplicated, duplicate_rejections = deduplicate_verified(eligible)
    write_jsonl(control_root / "dedup-rejections.jsonl", duplicate_rejections)
    coverage = {
        "schema_version": "general_coding_replay_coverage_v1",
        "status": "blocked_below_target" if not selected else "complete",
        "verified_pool": len(rows),
        "quality_eligible": len(quality_rows),
        "cross_domain_eligible": len(eligible),
        "deduplicated_pool": len(deduplicated),
        "duplicates": len(duplicate_rejections),
        "selected": len(selected),
        "target": target,
        "verified_distributions": {
            "source": _counter(quality_rows, "source_id"),
            "language": _counter(quality_rows, "primary_language"),
            "task_type": _counter(quality_rows, "coding_task_type"),
            "repository": _counter(quality_rows, "upstream_repository"),
        },
        "selected_distributions": {
            "source": _counter(selected, "source_id"),
            "language": _counter(selected, "primary_language"),
            "task_type": _counter(selected, "coding_task_type"),
        },
        "effective_targets": {
            key: quotas[key]
            for key in ("source_targets", "source_language_targets", "language_targets", "task_type_targets")
        },
    }
    write_json(control_root / "coverage-report.json", coverage)

    selected_receipt_paths = [
        control_root / str(row["_verified_receipt_path"]) for row in selected
    ]
    trust = {
        "schema_version": "general_coding_replay_trust_v1",
        "status": "pass" if len(selected) == target and quality["status"] == "pass" else "fail",
        "immutable_verified_receipts": len(selected),
        "offline_cpu_only_double_replay_passed": len(selected),
        "quality_failures": 0 if quality["status"] == "pass" else quality["failed"],
        "verified_receipt_sha256": {
            path.relative_to(control_root).as_posix(): sha256_file(path)
            for path in selected_receipt_paths
        },
    }
    write_json(control_root / "trust-report.json", trust)

    assistant_tokens = [
        row.get("assistant_loss_tokens")
        for row in selected
        if isinstance(row.get("assistant_loss_tokens"), int)
    ]
    kernel_tokens = {
        row.get("kernel_assistant_loss_tokens")
        for row in selected
        if isinstance(row.get("kernel_assistant_loss_tokens"), int)
    }
    package_blockers = []
    if quality["status"] != "pass":
        package_blockers.append("verified receipt quality audit did not pass")
    if leakage["status"] != "pass":
        package_blockers.append("selected rows overlap a protected split")
    if len(selected) != target:
        package_blockers.append(f"accepted rows {len(selected)} != {target}")
    if len(assistant_tokens) != target:
        package_blockers.append(
            f"assistant_loss_tokens present on {len(assistant_tokens)} of {target} rows"
        )
    if len(kernel_tokens) != 1:
        package_blockers.append(
            "one frozen kernel_assistant_loss_tokens value is not present on all accepted rows"
        )
    token_share = None
    if len(assistant_tokens) == target and len(kernel_tokens) == 1:
        general_tokens = sum(assistant_tokens)
        kernel_token_count = next(iter(kernel_tokens))
        token_share = general_tokens / (general_tokens + kernel_token_count)
        if not 0.15 <= token_share <= 0.20:
            package_blockers.append(
                f"General Coding assistant loss token share {token_share:.6f} "
                "is outside 15%-20%"
            )
    package_gate = {
        "schema_version": "general_coding_replay_package_gate_v1",
        "status": "blocked" if package_blockers else "ready",
        "hf_artifact_created": not package_blockers,
        "blockers": package_blockers,
    }
    write_json(control_root / "package-gate-report.json", package_gate)
    write_json(
        control_root / "token-mix-report.json",
        {
            **package_gate,
            "assistant_loss_token_rows": len(assistant_tokens),
            "kernel_token_values": sorted(kernel_tokens),
            "general_coding_assistant_loss_token_share": token_share,
        },
    )
    manifest = {
        "schema_version": "general_coding_replay_package_v1",
        "status": package_gate["status"],
        "sample_count": len(selected),
        "accepted": {
            "path": "accepted.jsonl",
            "sha256": sha256_file(control_root / "accepted.jsonl"),
        },
        "relative_paths_only": True,
        "token_counts": {
            "general_coding_assistant_loss_tokens": sum(assistant_tokens),
            "kernel_assistant_loss_tokens": next(iter(kernel_tokens))
            if len(kernel_tokens) == 1
            else None,
        },
        "blockers": package_blockers,
    }
    write_json(control_root / "manifest.json", manifest)

    outputs = [
        control_root / name
        for name in (
            "accepted.jsonl",
            "verified-pool.jsonl",
            "selection-report.json",
            "quota-deficits.json",
            "quality-report.json",
            "candidate-leakage-exclusions.json",
            "leakage-report.json",
            "source-lineage-audit.json",
            "coverage-report.json",
            "trust-report.json",
            "package-gate-report.json",
            "manifest.json",
            "token-mix-report.json",
            "dedup-rejections.jsonl",
            "quotas.yaml",
            "quota-overrides.yaml",
            "quota-override-evidence.json",
            "quota-v2-derivation.json",
            "token-counts.json",
        )
        if (control_root / name).is_file()
    ]
    all_inputs = sorted(set(verified_paths + receipt_paths + outputs))
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(control_root).as_posix()}"
        for path in all_inputs
    ]
    (control_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "status": selection["status"],
        "verified_pool": len(rows),
        "quality_eligible": len(quality_rows),
        "deduplicated_pool": len(deduplicated),
        "accepted": len(selected),
        "leakage_status": leakage["status"],
        "package_status": package_gate["status"],
        "checksums_sha256": hashlib.sha256(
            (control_root / "checksums.sha256").read_bytes()
        ).hexdigest(),
    }
    write_json(control_root / "finalization-report.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--kernel-train", type=Path, required=True)
    parser.add_argument("--kernel-dev", type=Path, required=True)
    parser.add_argument("--kernel-held-out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument(
        "--derive-quota-v2",
        action="store_true",
        help="Freeze achievable exact quotas from the leakage-clean verified pool",
    )
    args = parser.parse_args(argv)
    report = finalize(
        args.control_root,
        args.kernel_train,
        args.kernel_dev,
        args.kernel_held_out,
        target=args.target,
        derive_quota_v2=args.derive_quota_v2,
    )
    print(canonical_json(report))
    return 0 if report["status"] == "exact" else 2


if __name__ == "__main__":
    raise SystemExit(main())
