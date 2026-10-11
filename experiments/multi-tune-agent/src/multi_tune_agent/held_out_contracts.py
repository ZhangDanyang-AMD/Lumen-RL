"""Deterministic, eval-only held-out contracts from canonical Top10 factories."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .held_out_eval import (
    SCHEMA_VERSION,
    stable_hash,
    validate_eval_task,
    write_json,
    write_jsonl,
)
from .top10_canonical_templates import Top10Request, render_template, verify_locked_source
from .top10_inventory import AITER_GIT_SHA, AITER_ROOT, SOURCES, TOP10_FAMILIES


SUITES = ("geak_native", "aiter_derived", "adversarial_boundary")
LANES = ("hip_gfx942", "triton_gfx942")
VARIANT_VALUES = {
    "geak_native": (1, 5, 13, 29),
    "aiter_derived": (2, 7, 17, 31),
    "adversarial_boundary": (3, 9, 19, 37),
}
MATRIX_VERSION = "held-out-v5-contract-matrix-1"


class HeldOutContractError(ValueError):
    """The deterministic held-out contract build is incomplete or unsafe."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _source_rows() -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source in SOURCES:
        rows[str(source["family"])].append(source)
    missing = set(TOP10_FAMILIES) - set(rows)
    if missing:
        raise HeldOutContractError(f"Top10 source inventory is incomplete: {sorted(missing)}")
    return rows


def _lane(family: str, variant: int) -> str:
    # The canonical all-reduce factory is HIP-only. GEMM offsets its four HIP
    # units with four Triton units so every suite remains exactly 20/20.
    if family == "all_reduce":
        return "hip_gfx942"
    if family == "gemm":
        return "triton_gfx942"
    return LANES[variant % 2]


def _mutated_contract(
    source: Mapping[str, Any], suite: str, variant: int
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "operator": {
            "mha": "multi_head_attention",
            "mla": "multi_latent_attention",
            "paged_attention": "paged_attention",
            "fused_moe": "fused_moe",
            "gemm": "gemm",
            "rms_norm": "rms_norm",
            "rope_kv_cache": "rope_kv_cache",
            "blockscale_gemm": "blockscale_gemm",
            "all_reduce": "all_reduce",
            "sampling": "sampling",
        }[str(source["family"])],
        "mode": source["mode"],
        "shape": copy.deepcopy(source["shape"]),
        "input_dtype": "bf16",
        "output_dtype": "bf16",
    }
    if source["family"] == "all_reduce":
        contract["world_size"] = 2 if variant % 2 == 0 else 4
    if source["family"] == "sampling":
        contract["output_dtype"] = "int32"
    if source["family"] == "blockscale_gemm":
        contract.update(
            {
                "input_dtype": "fp8_e4m3fnuz",
                "weight_dtype": "fp8_e4m3fnuz",
                "scale": {"activation": "block128", "weight": "block128"},
            }
        )

    shape = contract["shape"]
    first_dimension = next(iter(shape))
    shape[first_dimension] = VARIANT_VALUES[suite][variant]
    return contract


def _request(
    source: Mapping[str, Any],
    suite: str,
    variant: int,
    lane: str,
    contract: Mapping[str, Any],
) -> Top10Request:
    family = str(source["family"])
    language = lane.split("_", 1)[0]
    lineage = f"heldv5:{suite}:{family}:matrix-{variant + 1:02d}"
    contract_id = f"HELDV5-{suite.upper()}-{family.upper()}-{variant + 1:02d}"
    seed = {
        "source_sha": AITER_GIT_SHA,
        "split_version": "v5",
        "split_group": "held_out",
        "source_artifacts": [
            {"path": str(source["path"]), "sha256": str(source["sha"])}
        ],
        "source_lineage_id": lineage,
        "contract_family_id": contract_id,
        "top10_family": family,
    }
    return Top10Request(
        request_id=f"heldv5-{suite}-{family}-{language}-{variant + 1:02d}",
        request_text="Deterministic eval-only held-out contract; generation is forbidden.",
        family=family,
        language=language,
        shape=tuple(int(value) for value in contract["shape"].values()),
        seed_provenance=seed,
        recognized_contract={
            "target_gpu": "gfx942",
            "language": language,
            "contract": dict(contract),
        },
    )


def _install(path: Path, content: str) -> str:
    payload = content.encode("utf-8")
    digest = _sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HeldOutContractError(f"append-safe artifact conflict: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return digest


def build_candidate_inventory(
    output_root: Path,
    *,
    aiter_root: Path = AITER_ROOT,
) -> list[dict[str, Any]]:
    """Build exactly 120 artifact-complete candidates without model generation."""
    verify_locked_source(aiter_root)
    source_rows = _source_rows()
    environment = {
        "architecture": "gfx942",
        "aiter_git_sha": AITER_GIT_SHA,
        "canonical_factory": "multi_tune_agent.top10_canonical_templates",
        "matrix_version": MATRIX_VERSION,
        "eval_only": True,
        "sft_enabled": False,
    }
    environment_hash = _sha256_bytes(_canonical_bytes(environment))
    environment_path = output_root / "private" / "environment.json"
    _install(
        environment_path,
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
    )

    records: list[dict[str, Any]] = []
    for suite in SUITES:
        for family in TOP10_FAMILIES:
            family_sources = source_rows[family]
            for variant in range(4):
                source = family_sources[variant % len(family_sources)]
                lane = _lane(family, variant)
                contract = _mutated_contract(source, suite, variant)
                request = _request(source, suite, variant, lane, contract)
                rendered = render_template(request)
                task_id = request.request_id
                artifact_root = output_root / "private" / "artifacts" / task_id
                source_hash = _install(
                    artifact_root / "initial_source.py", rendered["kernel.py"]
                )
                harness_hash = _install(
                    artifact_root / "harness.py", rendered["scripts/task_runner.py"]
                )
                oracle_hash = _install(
                    artifact_root / "oracle.py", rendered["scripts/task_runner.py"]
                )
                _install(artifact_root / "config.yaml", rendered["config.yaml"])
                _install(artifact_root / "factory_metadata.json", rendered["metadata.json"])
                protected_symbol = (
                    f"heldv5_{suite}_{family}_{lane.split('_', 1)[0]}_{variant + 1:02d}"
                )
                record = {
                    "task_id": task_id,
                    "source_lineage_id": request.seed_provenance["source_lineage_id"],
                    "contract_family_id": request.seed_provenance["contract_family_id"],
                    "implementation_family_id": f"independent:{task_id}",
                    "lane": lane,
                    "source_suite": suite,
                    "top10_family": family,
                    "contract": {
                        "matrix_version": MATRIX_VERSION,
                        "suite": suite,
                        "variant": variant + 1,
                        "kernel_contract": contract,
                    },
                    "initial_source": {
                        "path": str(
                            (artifact_root / "initial_source.py").relative_to(output_root)
                        ),
                        "sha256": source_hash,
                    },
                    "protected": {
                        "harness": {
                            "path": str(
                                (artifact_root / "harness.py").relative_to(output_root)
                            ),
                            "sha256": harness_hash,
                        },
                        "oracle": {
                            "path": str(
                                (artifact_root / "oracle.py").relative_to(output_root)
                            ),
                            "sha256": oracle_hash,
                            "selector": "independent oracle in canonical runner",
                        },
                    },
                    "environment_hash": environment_hash,
                    "protected_symbols": [protected_symbol],
                    "factory": {
                        "name": "multi_tune_agent.top10_canonical_templates",
                        "source_artifact": {
                            "path": str(source["path"]),
                            "sha256": str(source["sha"]),
                        },
                    },
                }
                records.append(record)

    records.sort(key=lambda item: str(item["task_id"]))
    if len(records) != 120:
        raise HeldOutContractError(f"expected 120 records, built {len(records)}")
    if len({(item["source_lineage_id"], item["lane"]) for item in records}) != 120:
        raise HeldOutContractError("candidate inventory has duplicate lineage/lane units")
    inventory_path = output_root / "inventory" / "task_candidates.jsonl"
    expected = "".join(
        json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in records
    )
    if inventory_path.exists() and inventory_path.read_text(encoding="utf-8") not in {
        "",
        expected,
    }:
        raise HeldOutContractError("append-safe candidate inventory conflict")
    write_jsonl(inventory_path, records)
    return records


def split_overlap_report(
    records: Iterable[Mapping[str, Any]],
    train_groups: Iterable[Mapping[str, Any]],
    dev_groups: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report held-out lineage, contract, and protected-symbol group overlap."""
    held = list(records)
    held_lineages = {str(item["source_lineage_id"]) for item in held}
    held_contracts = {str(item["contract_family_id"]) for item in held}
    held_symbols = {
        str(symbol) for item in held for symbol in item.get("protected_symbols", [])
    }
    hits = []
    for split, groups in (("train", train_groups), ("dev", dev_groups)):
        for group in groups:
            lineage = str(group.get("source_lineage_id") or "")
            if lineage in held_lineages:
                hits.append({"split": split, "kind": "lineage", "value": lineage})
            for contract in group.get("contract_family_ids", []):
                if str(contract) in held_contracts:
                    hits.append(
                        {"split": split, "kind": "contract", "value": str(contract)}
                    )
            for symbol in group.get("candidate_ids", []):
                if str(symbol) in held_symbols:
                    hits.append(
                        {"split": split, "kind": "symbol", "value": str(symbol)}
                    )
    return {
        "schema_version": "geak_kernel_held_out_split_overlap_v1",
        "status": "pass" if not hits else "fail",
        "held_out_records": len(held),
        "overlap_count": len(hits),
        "excluded_overlap_count": len(hits),
        "hits": sorted(hits, key=lambda item: (item["split"], item["kind"], item["value"])),
    }


def inventory_quota_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(records)
    lane = Counter(str(item["lane"]) for item in items)
    suite = Counter(str(item["source_suite"]) for item in items)
    family = Counter(str(item["top10_family"]) for item in items)
    return {
        "task_count": len(items),
        "lanes": dict(sorted(lane.items())),
        "source_suites": dict(sorted(suite.items())),
        "top10_families": dict(sorted(family.items())),
    }


def validate_materialized_artifacts(output_root: Path, tasks: Iterable[Mapping[str, Any]]) -> None:
    for task in tasks:
        validate_eval_task(task)
        for ref in (
            task["initial_source"],
            task["protected"]["harness"],
            task["protected"]["oracle"],
        ):
            path = output_root / str(ref["path"])
            if not path.is_file():
                raise HeldOutContractError(f"{task['task_id']}: missing artifact {path}")
            if _sha256_bytes(path.read_bytes()) != ref["sha256"]:
                raise HeldOutContractError(f"{task['task_id']}: artifact hash mismatch")


def write_private_manifest(output_root: Path, tasks: Iterable[Mapping[str, Any]]) -> None:
    items = list(tasks)
    write_json(
        output_root / "private" / "manifest.json",
        {
            "schema_version": "geak_kernel_held_out_private_package_v1",
            "eval_only": True,
            "sft_enabled": False,
            "task_count": len(items),
            "task_inventory_sha256": stable_hash(items),
            "contains_target_patches": False,
            "contains_trajectories": False,
            "contains_sft_events": False,
            "contains_model_generations": False,
        },
    )
