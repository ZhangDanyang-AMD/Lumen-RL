"""Strict-audit controller for configurable Phase 1 production waves."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .sft_dataset import (
    INPUT_MANIFEST_SCHEMA,
    audit_leakage,
    build_dataset,
    canonical_bytes,
    coverage_report,
    read_jsonl,
    sha256_file,
    validate_dataset,
    write_json,
)


MODE_QUOTAS = {
    "cold_start": 46,
    "profile_guided": 44,
    "direction_conditioned": 136,
    "error_recovery": 44,
    "regression_balance": 30,
}
LANE_QUOTAS = {"triton_gfx942": 150, "hip_gfx942": 150}
CELL_QUOTAS = {
    "cold_start": {"triton_gfx942": 23, "hip_gfx942": 23},
    "profile_guided": {"triton_gfx942": 22, "hip_gfx942": 22},
    "direction_conditioned": {"triton_gfx942": 68, "hip_gfx942": 68},
    "error_recovery": {"triton_gfx942": 22, "hip_gfx942": 22},
    "regression_balance": {"triton_gfx942": 15, "hip_gfx942": 15},
}
MODE_QUOTAS_2000 = {
    "cold_start": 307,
    "profile_guided": 293,
    "direction_conditioned": 907,
    "error_recovery": 293,
    "regression_balance": 200,
}
LANE_QUOTAS_2000 = {"triton_gfx942": 1000, "hip_gfx942": 1000}
CELL_QUOTAS_2000 = {
    "cold_start": {"triton_gfx942": 154, "hip_gfx942": 153},
    "profile_guided": {"triton_gfx942": 146, "hip_gfx942": 147},
    "direction_conditioned": {"triton_gfx942": 453, "hip_gfx942": 454},
    "error_recovery": {"triton_gfx942": 147, "hip_gfx942": 146},
    "regression_balance": {"triton_gfx942": 100, "hip_gfx942": 100},
}
PRIMARY_MODES = ("cold_start", "profile_guided", "direction_conditioned")
HARVEST_MAX_ATTEMPTS = 3
TOP10_FAMILIES = (
    "mha",
    "mla",
    "paged_attention",
    "fused_moe",
    "gemm",
    "rms_norm",
    "rope_kv_cache",
    "blockscale_gemm",
    "all_reduce",
    "sampling",
)
TOP10_OPERATOR_ALIASES = {
    "mha": "mha",
    "multi_head_attention": "mha",
    "flash_attention": "mha",
    "flash_attn": "mha",
    "mha_fwd": "mha",
    "attention": "mha",
    "mla": "mla",
    "mla_decode": "mla",
    "multi_latent_attention": "mla",
    "paged_attention": "paged_attention",
    "paged_attn": "paged_attention",
    "pa_fwd": "paged_attention",
    "fused_moe": "fused_moe",
    "moe": "fused_moe",
    "moe_op": "fused_moe",
    "ck_moe": "fused_moe",
    "triton_moe": "fused_moe",
    "grouped_gemm": "fused_moe",
    "grouped_gemm_a8w8": "fused_moe",
    "grouped_gemm_a16w16": "fused_moe",
    "grouped_gemm_a4w4": "fused_moe",
    "moe_grouped_gemm": "fused_moe",
    "moe_sorting": "fused_moe",
    "moe_align_block_size": "fused_moe",
    "moe_sum": "fused_moe",
    "fused_moe_fp8_blockscale": "fused_moe",
    "fused_moe_a16w16": "fused_moe",
    "fused_silu_mul": "fused_moe",
    "silu_and_mul": "fused_moe",
    "gemm": "gemm",
    "batched_gemm": "gemm",
    "gated_gemm": "gemm",
    "gemm_activation": "gemm",
    "scaled_quant_gemm": "gemm",
    "rms_norm": "rms_norm",
    "fused_add_rms_norm": "rms_norm",
    "rope": "rope_kv_cache",
    "rotary_embedding": "rope_kv_cache",
    "kv_cache": "rope_kv_cache",
    "reshape_and_cache": "rope_kv_cache",
    "rope_kv_cache": "rope_kv_cache",
    "blockscale_gemm": "blockscale_gemm",
    "block_scale_gemm": "blockscale_gemm",
    "block_scaled_gemm": "blockscale_gemm",
    "gemm_a8w8_blockscale": "blockscale_gemm",
    "all_reduce": "all_reduce",
    "allreduce": "all_reduce",
    "custom_all_reduce": "all_reduce",
    "sampling": "sampling",
    "topk": "sampling",
    "top_k": "sampling",
    "top_p_sampling": "sampling",
    "multinomial_sampling": "sampling",
}
DEFAULT_CASES = Path("/home/danyzhan/phase1_control/production-wave-300/production-cases.yaml")
DEFAULT_TRAJECTORY_ROOT = Path("/home/danyzhan/phase1_control/production-wave-300-v2")
DEFAULT_DATASET_ROOT = Path("/home/danyzhan/geak_sft_dataset/phase1-production-wave-300-v2")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mi300x.yaml"
DEFAULT_SEED_MANIFEST = Path(
    "/home/danyzhan/phase1_control/production-wave-300/audit/input-manifest.json"
)
WAVE_2000_CASES = Path(
    "/home/danyzhan/phase1_control/production-wave-2000-v1/production-cases.yaml"
)
WAVE_2000_TRAJECTORY_ROOT = Path("/home/danyzhan/phase1_control/production-wave-2000-v1")
WAVE_2000_DATASET_ROOT = Path(
    "/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1"
)
WAVE_2000_SEED_MANIFEST = Path(
    "/home/danyzhan/geak_sft_dataset/phase1-production-wave-300-v2/input_manifest.json"
)
FAILURE_WORDS = re.compile(
    r"\b(gpu|device|card)\b.{0,80}\b(busy|occupied|contention|in use|not idle)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    lane: str
    direction: str
    top10_family: str | None = None
    world_size: int = 1
    contract_hash: str = ""
    enforce_contract_hash: bool = False


@dataclass(frozen=True)
class AttemptSpec:
    number: int
    case_id: str
    lane: str
    mode: str
    gpu: int
    context: Mapping[str, Any] | None = None
    resume_workspace: str | None = None
    harvest_id: str | None = None
    gpus: tuple[int, ...] = ()
    world_size: int = 0

    def __post_init__(self) -> None:
        gpu_tuple = tuple(self.gpus) if self.gpus else (self.gpu,)
        world_size = self.world_size or len(gpu_tuple)
        if len(gpu_tuple) != world_size:
            raise ValueError("attempt GPU count must equal world_size")
        if len(set(gpu_tuple)) != len(gpu_tuple):
            raise ValueError("attempt GPUs must be distinct")
        if self.gpu != gpu_tuple[0]:
            raise ValueError("legacy gpu field must identify the first attempt GPU")
        object.__setattr__(self, "gpus", gpu_tuple)
        object.__setattr__(self, "world_size", world_size)

    @property
    def gpu_ids(self) -> str:
        return ",".join(str(gpu) for gpu in self.gpus)


@dataclass(frozen=True)
class WaveProfile:
    """Validated exact quotas and emitted dataset identity for one wave."""

    name: str
    dataset_version: str
    mode_quotas: Mapping[str, int]
    lane_quotas: Mapping[str, int]
    cell_quotas: Mapping[str, Mapping[str, int]]
    allowed_case_splits: Sequence[str] = ("train", "dev")
    family_minimum_quotas: Mapping[str, int] | None = None
    checkpoint_sample_count: int = 0

    def __post_init__(self) -> None:
        modes = dict(self.mode_quotas)
        lanes = dict(self.lane_quotas)
        cells = {mode: dict(values) for mode, values in self.cell_quotas.items()}
        if isinstance(self.allowed_case_splits, (str, bytes)) or not isinstance(
            self.allowed_case_splits, Sequence
        ):
            raise ValueError("allowed_case_splits must be a list of split names")
        allowed_splits = tuple(self.allowed_case_splits)
        family_minimums = (
            dict(self.family_minimum_quotas)
            if self.family_minimum_quotas is not None
            else {}
        )
        if not self.name or not self.dataset_version:
            raise ValueError("wave profile name and dataset_version must be non-empty")
        if not allowed_splits or any(
            not isinstance(split, str) or split not in {"train", "dev"}
            for split in allowed_splits
        ):
            raise ValueError(
                "allowed_case_splits must contain train and/or dev; held_out is forbidden"
            )
        allowed_splits = tuple(dict.fromkeys(allowed_splits))
        if set(modes) != set(PRIMARY_MODES) | {"error_recovery", "regression_balance"}:
            raise ValueError("wave profile must define all five production modes")
        if set(lanes) != {"triton_gfx942", "hip_gfx942"}:
            raise ValueError("wave profile must define both gfx942 lanes")
        for label, quotas in (("mode", modes), ("lane", lanes)):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in quotas.values()):
                raise ValueError(f"{label} quotas must be non-negative integers")
        if set(cells) != set(modes) or any(set(row) != set(lanes) for row in cells.values()):
            raise ValueError("cell quotas must define every mode/lane combination")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in cells.values()
            for value in row.values()
        ):
            raise ValueError("cell quotas must be non-negative integers")
        if any(sum(cells[mode].values()) != quota for mode, quota in modes.items()):
            raise ValueError("cell quota rows must equal mode quotas")
        if any(sum(cells[mode][lane] for mode in modes) != quota for lane, quota in lanes.items()):
            raise ValueError("cell quota columns must equal lane quotas")
        if sum(modes.values()) != sum(lanes.values()):
            raise ValueError("mode and lane quota totals must match")
        if (
            isinstance(self.checkpoint_sample_count, bool)
            or not isinstance(self.checkpoint_sample_count, int)
            or not 0 <= self.checkpoint_sample_count <= sum(modes.values())
        ):
            raise ValueError("checkpoint_sample_count must be within the wave sample count")
        if family_minimums and set(family_minimums) != set(TOP10_FAMILIES):
            raise ValueError("family minimum quotas must define all ten canonical Top10 families")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in family_minimums.values()
        ):
            raise ValueError("family minimum quotas must be non-negative integers")
        if sum(family_minimums.values()) > sum(modes.values()) - self.checkpoint_sample_count:
            raise ValueError("family minimum quotas exceed post-checkpoint capacity")
        object.__setattr__(self, "mode_quotas", modes)
        object.__setattr__(self, "lane_quotas", lanes)
        object.__setattr__(self, "cell_quotas", cells)
        object.__setattr__(self, "allowed_case_splits", allowed_splits)
        object.__setattr__(self, "family_minimum_quotas", family_minimums)

    @property
    def sample_count(self) -> int:
        return sum(self.mode_quotas.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_version": self.dataset_version,
            "mode_quotas": dict(self.mode_quotas),
            "lane_quotas": dict(self.lane_quotas),
            "cell_quotas": {
                mode: dict(lanes) for mode, lanes in self.cell_quotas.items()
            },
            "allowed_case_splits": list(self.allowed_case_splits),
            **(
                {"family_minimum_quotas": dict(self.family_minimum_quotas)}
                if self.family_minimum_quotas
                else {}
            ),
            **(
                {"checkpoint_sample_count": self.checkpoint_sample_count}
                if self.checkpoint_sample_count
                else {}
            ),
        }


DEFAULT_PROFILE = WaveProfile(
    "300",
    "phase1-production-wave-300-v2",
    MODE_QUOTAS,
    LANE_QUOTAS,
    CELL_QUOTAS,
)
WAVE_2000_PROFILE = WaveProfile(
    "2000",
    "phase1-production-wave-2000-v1",
    MODE_QUOTAS_2000,
    LANE_QUOTAS_2000,
    CELL_QUOTAS_2000,
    ("train",),
)
BUILTIN_PROFILES = {"300": DEFAULT_PROFILE, "2000": WAVE_2000_PROFILE}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def canonical_top10_family(value: Any) -> str | None:
    """Return a canonical family for an exact, case-normalized operator alias."""
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[-\s]+", "_", value.strip().lower())
    return TOP10_OPERATOR_ALIASES.get(normalized)


def load_cases(path: Path, profile: WaveProfile = DEFAULT_PROFILE) -> list[Case]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("production cases must contain a non-empty tasks list")
    result: list[Case] = []
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, Mapping):
            raise ValueError("every production case must be an object")
        case_id = str(item.get("id") or "")
        backend = str(item.get("backend") or "").lower()
        architecture = str(item.get("architecture") or "").lower()
        provenance = item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        seed = provenance.get("case_seed")
        seed = seed if isinstance(seed, Mapping) else {}
        lane = str(seed.get("target_lane") or f"{backend}_{architecture}")
        if not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate production case ID: {case_id!r}")
        if lane not in profile.lane_quotas:
            raise ValueError(f"{case_id}: unsupported production lane {lane!r}")
        if backend not in {"triton", "hip"} or architecture != "gfx942":
            raise ValueError(f"{case_id}: case must be Triton/HIP gfx942")
        split_group = seed.get("split_group")
        if split_group == "held_out":
            raise ValueError(f"{case_id}: held_out cases are forbidden for production")
        if split_group not in {"train", "dev"}:
            raise ValueError(f"{case_id}: case does not have a validated train/dev split")
        if not seed.get("source_lineage_id") or not seed.get("split_version"):
            raise ValueError(f"{case_id}: case seed provenance is incomplete")
        seen.add(case_id)
        if split_group not in profile.allowed_case_splits:
            continue
        explicit_family = item.get("top10_family")
        if explicit_family is None:
            explicit_family = provenance.get("top10_family")
        if explicit_family is None:
            explicit_family = seed.get("top10_family")
        recognized = item.get("recognized_contract")
        if explicit_family is None and isinstance(recognized, Mapping):
            explicit_family = recognized.get("top10_family")
        if explicit_family is not None:
            family = canonical_top10_family(explicit_family)
            if family != explicit_family:
                raise ValueError(
                    f"{case_id}: top10_family must be one of the ten canonical family names"
                )
        else:
            family = canonical_top10_family(item.get("operator"))
        world_size = 1
        if family == "all_reduce" and isinstance(recognized, Mapping):
            contract = recognized.get("contract")
            if isinstance(contract, Mapping):
                candidate_world_size = contract.get("world_size")
                if (
                    not isinstance(candidate_world_size, bool)
                    and isinstance(candidate_world_size, int)
                    and candidate_world_size in {2, 4}
                ):
                    world_size = candidate_world_size
        result.append(
            Case(
                case_id,
                lane,
                str(item.get("direction") or ""),
                family,
                world_size,
                str(item.get("contract_hash") or ""),
                bool((item.get("trust") or {}).get("enforce_contract_hash")),
            )
        )
    if set(case.lane for case in result) != set(profile.lane_quotas):
        allowed = "/".join(profile.allowed_case_splits)
        raise ValueError(
            f"production cases allowed by the {profile.name} profile ({allowed}) "
            "must cover both gfx942 lanes"
        )
    return result


def _artifact_entry(run_dir: Path, candidate_ids: Iterable[str] | None = None) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("round_*/candidates.jsonl"))
    artifacts = [run_dir / "sft_manifest.json", run_dir / "environment.json"]
    for path in candidates:
        artifacts.extend((path, path.parent / "frozen_input.json"))
        plan = path.parent / "plan.json"
        if plan.is_file():
            artifacts.append(plan)
    entry: dict[str, Any] = {
        "path": str(run_dir),
        "candidate_files": [str(path.relative_to(run_dir)) for path in candidates],
        "artifacts": {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in artifacts
            if path.is_file()
        },
    }
    if candidate_ids is not None:
        entry["candidate_ids"] = sorted(set(candidate_ids))
    return entry


def write_input_manifest(
    path: Path,
    runs: Sequence[Path],
    dataset_root: Path,
    blob_root: Path,
    *,
    selected: Mapping[str, set[str]] | None = None,
    seed_manifest: Path | None = None,
    profile: WaveProfile = DEFAULT_PROFILE,
    pinned_keys: set[tuple[str, str]] | None = None,
    task_families: Mapping[str, str | None] | None = None,
) -> None:
    entries: list[dict[str, Any]] = []
    environment_overrides: dict[str, Any] = {}
    if seed_manifest is not None and seed_manifest.is_file():
        seed = json.loads(seed_manifest.read_text(encoding="utf-8"))
        seed_blob_root = seed.get("blob_root")
        seed_entries = seed.get("runs")
        if not isinstance(seed_entries, list):
            raise ValueError("seed input manifest runs must be a list")
        for raw_entry in seed_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("seed input manifest run must be an object")
            entry = dict(raw_entry)
            run_id = Path(str(entry.get("path") or "")).name
            ids = selected.get(run_id, set()) if selected is not None else None
            if selected is not None and not ids:
                continue
            if ids is not None:
                entry["candidate_ids"] = sorted(ids)
            entry.setdefault("blob_root", seed_blob_root)
            entries.append(entry)
        raw_overrides = seed.get("environment_overrides")
        if isinstance(raw_overrides, Mapping):
            environment_overrides.update(raw_overrides)
    for run in sorted(runs):
        ids = selected.get(run.name, set()) if selected is not None else None
        if selected is not None and not ids:
            continue
        entries.append(_artifact_entry(run, ids))
    if not entries and selected is not None and runs:
        # Keep the ETL authoritative even before the first acceptance: an empty
        # allow-list on one pinned run deterministically rebuilds an empty dataset.
        entries.append(_artifact_entry(sorted(runs)[0], []))
    if not entries:
        raise ValueError("no complete SFT runs are available for the input manifest")
    write_json(
        path,
        {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "dataset_version": profile.dataset_version,
            "output_root": str(dataset_root),
            "blob_root": str(blob_root),
            "runs": entries,
            **(
                {
                    "task_families": {
                        task_id: family
                        for task_id, family in sorted(task_families.items())
                        if family is not None
                    }
                }
                if task_families
                else {}
            ),
            **(
                {
                    "pinned_sample_keys": [
                        list(key) for key in sorted(pinned_keys)
                    ]
                }
                if pinned_keys
                else {}
            ),
            "coverage_targets": {
                "sample_count": profile.sample_count,
                "task_type": profile.mode_quotas,
                "lane": profile.lane_quotas,
                **(
                    {
                        "checkpoint_sample_count": profile.checkpoint_sample_count,
                        "post_checkpoint_sample_count": (
                            profile.sample_count - profile.checkpoint_sample_count
                        ),
                        "post_checkpoint_top10_family": profile.family_minimum_quotas,
                    }
                    if profile.family_minimum_quotas
                    else {}
                ),
            },
            **(
                {"environment_overrides": environment_overrides}
                if environment_overrides
                else {}
            ),
        },
    )


def _max_flow_allocations(
    samples: Sequence[Mapping[str, Any]],
    profile: WaveProfile = DEFAULT_PROFILE,
    consumed: Mapping[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    """Select the largest quota-bounded mode/lane matrix with a tiny max flow."""
    available = Counter(
        (str(item.get("task_type")), str(item.get("provenance", {}).get("lane")))
        for item in samples
    )
    source, sink = "source", "sink"
    capacity: dict[tuple[str, str], int] = {}
    consumed = consumed or {}
    for mode, quota in profile.mode_quotas.items():
        capacity[(source, f"m:{mode}")] = quota - sum(
            consumed.get((mode, lane), 0) for lane in profile.lane_quotas
        )
        for lane in profile.lane_quotas:
            capacity[(f"m:{mode}", f"l:{lane}")] = min(
                available[(mode, lane)],
                profile.cell_quotas[mode][lane] - consumed.get((mode, lane), 0),
            )
    for lane, quota in profile.lane_quotas.items():
        capacity[(f"l:{lane}", sink)] = quota - sum(
            consumed.get((mode, lane), 0) for mode in profile.mode_quotas
        )
    flow: Counter[tuple[str, str]] = Counter()
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            neighbors = {right for left, right in capacity if left == node}
            neighbors |= {left for left, right in capacity if right == node}
            for other in sorted(neighbors):
                residual = (
                    capacity.get((node, other), 0) - flow[(node, other)]
                    if (node, other) in capacity
                    else flow[(other, node)]
                )
                if residual > 0 and other not in parent:
                    parent[other] = node
                    queue.append(other)
        if sink not in parent:
            break
        amount = 10**9
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            residual = (
                capacity.get((previous, node), 0) - flow[(previous, node)]
                if (previous, node) in capacity
                else flow[(node, previous)]
            )
            amount = min(amount, residual)
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            if (previous, node) in capacity:
                flow[(previous, node)] += amount
            else:
                flow[(node, previous)] -= amount
            node = previous
    return {
        (mode, lane): flow[(f"m:{mode}", f"l:{lane}")]
        for mode in profile.mode_quotas
        for lane in profile.lane_quotas
    }


def _sample_key(sample: Mapping[str, Any]) -> tuple[str, str]:
    provenance = sample.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return str(provenance.get("run_id") or ""), str(provenance.get("candidate_id") or "")


def sample_top10_family(
    sample: Mapping[str, Any],
    task_families: Mapping[str, str | None] | None = None,
) -> str | None:
    """Resolve family from stable sample provenance without fuzzy matching."""
    provenance = sample.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    explicit = provenance.get("top10_family")
    if explicit is not None:
        return str(explicit) if explicit in TOP10_FAMILIES else None
    family = canonical_top10_family(provenance.get("operator"))
    if family is not None:
        return family
    task_id = str(provenance.get("task_id") or "")
    return (task_families or {}).get(task_id)


def _family_minimum_allocations(
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    pinned_cells: Mapping[tuple[str, str], int],
    profile: WaveProfile,
    task_families: Mapping[str, str | None] | None,
) -> dict[tuple[str, tuple[str, str]], int]:
    """Allocate family lower bounds across exact mode/lane cells."""
    minimums = dict(profile.family_minimum_quotas or {})
    if not minimums:
        return {}
    cells = [
        (mode, lane)
        for mode in profile.mode_quotas
        for lane in profile.lane_quotas
    ]
    needs = dict(minimums)
    source, sink = ("source",), ("sink",)
    capacity: dict[tuple[tuple[Any, ...], tuple[Any, ...]], int] = {}
    for family, need in needs.items():
        family_node = ("family", family)
        capacity[(source, family_node)] = need
        for cell in cells:
            available = sum(
                sample_top10_family(sample, task_families) == family
                for sample in grouped.get(cell, ())
            )
            capacity[(family_node, ("cell", *cell))] = available
    for mode, lane in cells:
        capacity[(("cell", mode, lane), sink)] = (
            profile.cell_quotas[mode][lane]
            - int(pinned_cells.get((mode, lane), 0))
        )

    flow: Counter[tuple[tuple[Any, ...], tuple[Any, ...]]] = Counter()
    while True:
        parent: dict[tuple[Any, ...], tuple[Any, ...] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            neighbors = {right for left, right in capacity if left == node}
            neighbors |= {left for left, right in capacity if right == node}
            for other in sorted(neighbors):
                residual = (
                    capacity.get((node, other), 0) - flow[(node, other)]
                    if (node, other) in capacity
                    else flow[(other, node)]
                )
                if residual > 0 and other not in parent:
                    parent[other] = node
                    queue.append(other)
        if sink not in parent:
            break
        amount = 10**9
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            residual = (
                capacity.get((previous, node), 0) - flow[(previous, node)]
                if (previous, node) in capacity
                else flow[(node, previous)]
            )
            amount = min(amount, residual)
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            if (previous, node) in capacity:
                flow[(previous, node)] += amount
            else:
                flow[(node, previous)] -= amount
            node = previous
    allocated = sum(flow[(source, ("family", family))] for family in needs)
    required = sum(needs.values())
    if allocated != required:
        deficits = ", ".join(
            f"{family}={needs[family] - flow[(source, ('family', family))]}"
            for family in TOP10_FAMILIES
            if needs.get(family, 0) > flow[(source, ("family", family))]
        )
        raise ValueError(f"family minimum quotas are infeasible: {deficits}")
    return {
        (family, cell): flow[(("family", family), ("cell", *cell))]
        for family in TOP10_FAMILIES
        for cell in cells
        if flow[(("family", family), ("cell", *cell))]
    }


def seed_sample_keys(
    samples: Sequence[Mapping[str, Any]], seed_manifest: Path | None
) -> set[tuple[str, str]]:
    """Resolve the seed manifest allow-list against built, accepted samples."""
    if seed_manifest is None or not seed_manifest.is_file():
        return set()
    payload = json.loads(seed_manifest.read_text(encoding="utf-8"))
    entries = payload.get("runs")
    if not isinstance(entries, list):
        raise ValueError("seed input manifest runs must be a list")
    restrictions: dict[str, set[str] | None] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("seed input manifest run must be an object")
        run_id = Path(str(raw_entry.get("path") or "")).name
        if not run_id or run_id in restrictions:
            raise ValueError(f"seed input manifest has invalid or duplicate run: {run_id!r}")
        raw_ids = raw_entry.get("candidate_ids")
        if raw_ids is None:
            restrictions[run_id] = None
        elif isinstance(raw_ids, list) and all(isinstance(value, str) and value for value in raw_ids):
            restrictions[run_id] = set(raw_ids)
        else:
            raise ValueError(f"seed run {run_id!r} candidate_ids must be a string list")
    available = {_sample_key(sample) for sample in samples}
    requested = {
        (run_id, candidate_id)
        for run_id, candidate_ids in restrictions.items()
        if candidate_ids is not None
        for candidate_id in candidate_ids
    }
    missing = sorted(requested - available)
    if missing:
        preview = ", ".join(f"{run}:{candidate}" for run, candidate in missing[:5])
        raise ValueError(f"{len(missing)} pinned seed candidates are unavailable after ETL: {preview}")
    return {
        key
        for key in available
        if key[0] in restrictions
        and (restrictions[key[0]] is None or key[1] in restrictions[key[0]])
    }


def select_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    profile: WaveProfile = DEFAULT_PROFILE,
    pinned_keys: Iterable[tuple[str, str]] = (),
    task_families: Mapping[str, str | None] | None = None,
    require_complete: bool = True,
) -> dict[str, set[str]]:
    pinned = set(pinned_keys)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    pinned_counts: Counter[tuple[str, str]] = Counter()
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for sample in samples:
        key = _sample_key(sample)
        if not all(key):
            raise ValueError("sample is missing provenance run_id/candidate_id")
        if key in by_key:
            raise ValueError(f"duplicate sample candidate identity: {key[0]}:{key[1]}")
        by_key[key] = sample
        provenance = sample.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        cell = (str(sample.get("task_type")), str(provenance.get("lane")))
        if cell[0] not in profile.mode_quotas or cell[1] not in profile.lane_quotas:
            if key in pinned:
                raise ValueError(f"pinned seed sample {key[0]}:{key[1]} has invalid quota cell {cell}")
            continue
        if key in pinned:
            pinned_counts[cell] += 1
        else:
            grouped[cell].append(sample)
    missing_pins = sorted(pinned - set(by_key))
    if missing_pins:
        raise ValueError(f"{len(missing_pins)} pinned seed samples are absent from candidates")
    if profile.family_minimum_quotas and len(pinned) != profile.checkpoint_sample_count:
        raise ValueError(
            "pinned seed sample count does not match checkpoint_sample_count: "
            f"{len(pinned)} != {profile.checkpoint_sample_count}"
        )
    exceeded = {
        cell: count
        for cell, count in pinned_counts.items()
        if count > profile.cell_quotas[cell[0]][cell[1]]
    }
    if exceeded:
        details = ", ".join(
            f"{mode}/{lane}={count}>{profile.cell_quotas[mode][lane]}"
            for (mode, lane), count in sorted(exceeded.items())
        )
        raise ValueError(f"pinned seed samples exceed wave cell quota: {details}")
    selected: dict[str, set[str]] = defaultdict(set)
    selected_keys = set(pinned)
    for run_id, candidate_id in sorted(pinned):
        selected[run_id].add(candidate_id)
    incomplete_cells = any(
        pinned_counts[(mode, lane)] + len(grouped[(mode, lane)])
        < profile.cell_quotas[mode][lane]
        for mode in profile.mode_quotas
        for lane in profile.lane_quotas
    )
    try:
        family_allocations = (
            {}
            if incomplete_cells and not require_complete
            else _family_minimum_allocations(
                grouped,
                pinned_counts,
                profile,
                task_families,
            )
        )
    except ValueError:
        if require_complete:
            raise
        family_allocations = {}
        incomplete_cells = True
    if incomplete_cells and not require_complete:
        allocations = _max_flow_allocations(
            [sample for values in grouped.values() for sample in values],
            profile,
            pinned_counts,
        )
        deficits = {
            family: quota
            for family, quota in (profile.family_minimum_quotas or {}).items()
        }
        for cell, count in allocations.items():
            ordered = sorted(
                grouped[cell],
                key=lambda sample: (
                    -deficits.get(sample_top10_family(sample, task_families) or "", 0),
                    str(sample["sample_id"]),
                ),
            )
            for sample in ordered[:count]:
                key = _sample_key(sample)
                selected[key[0]].add(key[1])
        return dict(selected)
    selected_by_cell: Counter[tuple[str, str]] = Counter(pinned_counts)
    for (family, cell), count in sorted(family_allocations.items()):
        ordered = sorted(
            (
                sample
                for sample in grouped[cell]
                if sample_top10_family(sample, task_families) == family
                and _sample_key(sample) not in selected_keys
            ),
            key=lambda item: str(item["sample_id"]),
        )
        if len(ordered) < count:
            raise ValueError(f"family minimum quota allocation changed for {family}/{cell}")
        for sample in ordered[:count]:
            key = _sample_key(sample)
            selected[key[0]].add(key[1])
            selected_keys.add(key)
            selected_by_cell[cell] += 1
    for cell in (
        (mode, lane)
        for mode in profile.mode_quotas
        for lane in profile.lane_quotas
    ):
        count = profile.cell_quotas[cell[0]][cell[1]] - selected_by_cell[cell]
        ordered = sorted(grouped[cell], key=lambda item: str(item["sample_id"]))
        remaining = [sample for sample in ordered if _sample_key(sample) not in selected_keys]
        if len(remaining) < count:
            raise ValueError(
                f"exact wave quota is infeasible for {cell[0]}/{cell[1]}: "
                f"need {count}, have {len(remaining)}"
            )
        for sample in remaining[:count]:
            key = _sample_key(sample)
            selected[key[0]].add(key[1])
            selected_keys.add(key)
    return dict(selected)


def processed_samples(dataset_root: Path) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for split in ("train", "dev", "held_out"):
        records, errors = read_jsonl(dataset_root / "processed" / f"{split}.jsonl")
        if errors:
            raise ValueError(f"processed {split} JSONL is invalid: {errors[0]}")
        values.extend(records)
    return values


def sample_counts(
    samples: Sequence[Mapping[str, Any]],
    profile: WaveProfile = DEFAULT_PROFILE,
    task_families: Mapping[str, str | None] | None = None,
    pinned_keys: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    modes = Counter(str(item.get("task_type")) for item in samples)
    lanes = Counter(str(item.get("provenance", {}).get("lane")) for item in samples)
    cells = Counter(
        (str(item.get("task_type")), str(item.get("provenance", {}).get("lane")))
        for item in samples
    )
    pinned = set(pinned_keys)
    families = Counter(sample_top10_family(item, task_families) for item in samples)
    checkpoint_families = Counter(
        sample_top10_family(item, task_families)
        for item in samples
        if _sample_key(item) in pinned
    )
    minimums = dict(profile.family_minimum_quotas or {})
    family_counts = {family: families[family] for family in TOP10_FAMILIES}
    checkpoint_family_counts = {
        family: checkpoint_families[family] for family in TOP10_FAMILIES
    }
    post_checkpoint_counts = {
        family: family_counts[family] - checkpoint_family_counts[family]
        for family in TOP10_FAMILIES
    }
    deficits = {
        family: max(0, quota - post_checkpoint_counts[family])
        for family, quota in minimums.items()
    }
    return {
        "accepted": len(samples),
        "task_type": {key: modes[key] for key in profile.mode_quotas},
        "lane": {key: lanes[key] for key in profile.lane_quotas},
        "cell": {
            mode: {lane: cells[(mode, lane)] for lane in profile.lane_quotas}
            for mode in profile.mode_quotas
        },
        "top10_family": family_counts,
        "checkpoint_top10_family": checkpoint_family_counts,
        "post_checkpoint_top10_family": post_checkpoint_counts,
        "family_status": {
            "minimum_quotas": minimums,
            "minimum_total": sum(minimums.values()),
            "checkpoint_samples": len(pinned),
            "checkpoint_top10": sum(checkpoint_family_counts.values()),
            "accepted_top10": sum(post_checkpoint_counts.values()),
            "deficits": deficits,
            "met": not any(deficits.values()),
        },
    }


def quotas_pass(
    counts: Mapping[str, Any], profile: WaveProfile = DEFAULT_PROFILE
) -> bool:
    return (
        counts.get("accepted") == profile.sample_count
        and all(counts.get("task_type", {}).get(key, 0) == value for key, value in profile.mode_quotas.items())
        and all(counts.get("lane", {}).get(key, 0) == value for key, value in profile.lane_quotas.items())
        and all(
            counts.get("cell", {}).get(mode, {}).get(lane, 0) == quota
            for mode, lanes in profile.cell_quotas.items()
            for lane, quota in lanes.items()
        )
        and all(
            counts.get("post_checkpoint_top10_family", {}).get(family, 0) >= quota
            for family, quota in (profile.family_minimum_quotas or {}).items()
        )
    )


class ProductionController:
    def __init__(
        self,
        *,
        cases_path: Path,
        trajectory_root: Path,
        dataset_root: Path,
        base_config: Path,
        gpus: Sequence[int],
        max_attempts: int | None,
        poll_seconds: float,
        seed_manifest: Path | None = None,
        profile: WaveProfile = DEFAULT_PROFILE,
    ) -> None:
        self.profile = profile
        self.cases_path = cases_path.resolve()
        self.trajectory_root = trajectory_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.base_config = base_config.resolve()
        self.gpus = tuple(dict.fromkeys(gpus))
        if not self.gpus:
            raise ValueError("at least one GPU is required")
        if 0 in self.gpus:
            raise ValueError("GPU0 is reserved for the model server")
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.seed_manifest = seed_manifest.resolve() if seed_manifest else None
        self.cases = load_cases(self.cases_path, self.profile)
        self.task_families = {
            case.case_id: case.top10_family for case in self.cases
        }
        self.task_contracts = {
            case.case_id: case.contract_hash
            for case in self.cases
            if case.enforce_contract_hash
        }
        self.control_root = self.trajectory_root / "controller"
        self.state_path = self.control_root / "state.json"
        self.attempt_root = self.control_root / "attempts"
        self.audit_root = self.control_root / "audit-build"
        self.state = self._load_state()
        self._capture_family_checkpoint()

    def _load_state(self) -> dict[str, Any]:
        identity = {
            "profile": self.profile.as_dict(),
            "cases_path": str(self.cases_path),
            "dataset_root": str(self.dataset_root),
            "seed_manifest": str(self.seed_manifest) if self.seed_manifest else None,
        }
        if self.state_path.is_file():
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != "phase1_production_controller_v1":
                raise ValueError("controller state schema is incompatible")
            saved_identity = value.get("wave_identity")
            if saved_identity is None:
                if self.profile != DEFAULT_PROFILE:
                    raise ValueError(
                        "legacy controller state has no wave profile; refusing non-300 resume"
                    )
                value["wave_identity"] = identity
            elif saved_identity != identity:
                legacy_identity = json.loads(json.dumps(identity))
                legacy_profile = legacy_identity.get("profile", {})
                legacy_profile.pop("allowed_case_splits", None)
                legacy_profile.pop("family_minimum_quotas", None)
                legacy_profile.pop("checkpoint_sample_count", None)
                profile_upgrade_identity = json.loads(json.dumps(identity))
                upgrade_profile = profile_upgrade_identity.get("profile", {})
                upgrade_profile.pop("family_minimum_quotas", None)
                upgrade_profile.pop("checkpoint_sample_count", None)
                if (
                    self.profile.family_minimum_quotas
                    and saved_identity == profile_upgrade_identity
                    and int(value.get("counts", {}).get("accepted", -1))
                    == self.profile.checkpoint_sample_count
                ):
                    value["wave_identity"] = identity
                    value["family_profile_upgraded_at"] = time.time()
                elif self.profile == DEFAULT_PROFILE and saved_identity == legacy_identity:
                    value["wave_identity"] = identity
                else:
                    raise ValueError(
                        "controller state wave/profile mismatch; use a new trajectory root "
                        "or the original profile, cases, dataset, and seed manifest"
                    )
            counts = value.setdefault("counts", {})
            counts.setdefault("top10_family", dict.fromkeys(TOP10_FAMILIES, 0))
            counts.setdefault("checkpoint_top10_family", dict.fromkeys(TOP10_FAMILIES, 0))
            counts.setdefault(
                "post_checkpoint_top10_family",
                dict(counts["top10_family"]),
            )
            counts.setdefault(
                "family_status",
                {
                    "minimum_quotas": dict(self.profile.family_minimum_quotas or {}),
                    "minimum_total": sum((self.profile.family_minimum_quotas or {}).values()),
                    "checkpoint_samples": self.profile.checkpoint_sample_count,
                    "checkpoint_top10": sum(counts["checkpoint_top10_family"].values()),
                    "accepted_top10": sum(
                        counts["post_checkpoint_top10_family"].values()
                    ),
                    "deficits": dict(self.profile.family_minimum_quotas or {}),
                    "met": not bool(self.profile.family_minimum_quotas),
                },
            )
            for attempt in value.setdefault("attempts", []):
                raw_gpus = attempt.get("gpus")
                if raw_gpus is None:
                    raw_gpus = [attempt["gpu"]]
                if (
                    not isinstance(raw_gpus, list)
                    or not raw_gpus
                    or any(
                        isinstance(gpu, bool) or not isinstance(gpu, int)
                        for gpu in raw_gpus
                    )
                    or len(set(raw_gpus)) != len(raw_gpus)
                    or 0 in raw_gpus
                ):
                    raise ValueError("controller attempt gang GPU state is invalid")
                world_size = attempt.get("world_size", len(raw_gpus))
                if world_size != len(raw_gpus):
                    raise ValueError("controller attempt world_size does not match GPUs")
                attempt["gpu"] = raw_gpus[0]
                attempt["gpus"] = raw_gpus
                attempt["world_size"] = world_size
            return value
        return {
            "schema_version": "phase1_production_controller_v1",
            "wave_identity": identity,
            "status": "running",
            "attempts_started": 0,
            "attempts": [],
            "attempted_case_modes": {},
            "harvested": {},
            "queues": {"error_recovery": [], "regression_balance": []},
            "counts": {
                "accepted": 0,
                "task_type": {},
                "lane": {},
                "top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "checkpoint_top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "post_checkpoint_top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "family_status": {
                    "minimum_quotas": dict(self.profile.family_minimum_quotas or {}),
                    "minimum_total": sum((self.profile.family_minimum_quotas or {}).values()),
                    "checkpoint_samples": self.profile.checkpoint_sample_count,
                    "checkpoint_top10": 0,
                    "accepted_top10": 0,
                    "deficits": dict(self.profile.family_minimum_quotas or {}),
                    "met": not bool(self.profile.family_minimum_quotas),
                },
            },
            "created_at": time.time(),
        }

    def _capture_family_checkpoint(self) -> None:
        """Freeze the accepted checkpoint identities before the next rebuild."""
        if not self.profile.family_minimum_quotas:
            return
        raw_keys = self.state.get("family_checkpoint_keys")
        if raw_keys is not None:
            if (
                not isinstance(raw_keys, list)
                or len(raw_keys) != self.profile.checkpoint_sample_count
                or any(
                    not isinstance(key, list)
                    or len(key) != 2
                    or not all(isinstance(value, str) and value for value in key)
                    for key in raw_keys
                )
            ):
                raise ValueError("controller family checkpoint identities are invalid")
            return
        processed = self.dataset_root / "processed"
        if not processed.is_dir():
            return
        samples = processed_samples(self.dataset_root)
        if len(samples) != self.profile.checkpoint_sample_count:
            raise ValueError(
                "current dataset does not match checkpoint_sample_count: "
                f"{len(samples)} != {self.profile.checkpoint_sample_count}"
            )
        keys = sorted(_sample_key(sample) for sample in samples)
        if any(not all(key) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("current checkpoint samples have invalid candidate identities")
        self.state["family_checkpoint_keys"] = [list(key) for key in keys]
        # Persist before rebuild can replace dataset_root/input_manifest.json.
        self._save()

    def _family_checkpoint_keys(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> set[tuple[str, str]]:
        raw_keys = self.state.get("family_checkpoint_keys")
        if raw_keys is None:
            keys = seed_sample_keys(samples, self.seed_manifest)
            if len(keys) != self.profile.checkpoint_sample_count:
                raise ValueError(
                    "seed manifest does not match checkpoint_sample_count: "
                    f"{len(keys)} != {self.profile.checkpoint_sample_count}"
                )
            self.state["family_checkpoint_keys"] = [
                list(key) for key in sorted(keys)
            ]
            return keys
        keys = {(str(key[0]), str(key[1])) for key in raw_keys}
        available = {_sample_key(sample) for sample in samples}
        missing = sorted(keys - available)
        if missing:
            raise ValueError(
                f"{len(missing)} frozen checkpoint samples are unavailable after ETL"
            )
        return keys

    def _save(self) -> None:
        self.state["updated_at"] = time.time()
        atomic_json(self.state_path, self.state)

    def complete_runs(self) -> list[Path]:
        runs = self.trajectory_root / "runs"
        if not runs.is_dir():
            return []
        return [
            path
            for path in runs.iterdir()
            if path.is_dir()
            and (path / "sft_manifest.json").is_file()
            and (path / "environment.json").is_file()
            and any(path.glob("round_*/candidates.jsonl"))
        ]

    def rebuild(self) -> dict[str, Any]:
        runs = self.complete_runs()
        if not runs and not (
            self.seed_manifest is not None and self.seed_manifest.is_file()
        ):
            counts = {
                "accepted": 0,
                "task_type": dict.fromkeys(self.profile.mode_quotas, 0),
                "lane": dict.fromkeys(self.profile.lane_quotas, 0),
                "cell": {
                    mode: dict.fromkeys(self.profile.lane_quotas, 0)
                    for mode in self.profile.mode_quotas
                },
                "top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "checkpoint_top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "post_checkpoint_top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
                "family_status": {
                    "minimum_quotas": dict(self.profile.family_minimum_quotas or {}),
                    "minimum_total": sum((self.profile.family_minimum_quotas or {}).values()),
                    "checkpoint_samples": self.profile.checkpoint_sample_count,
                    "checkpoint_top10": 0,
                    "accepted_top10": 0,
                    "deficits": dict(self.profile.family_minimum_quotas or {}),
                    "met": not bool(self.profile.family_minimum_quotas),
                },
            }
            self.state["counts"] = counts
            self._save()
            return {"counts": counts, "quality": None, "leakage": None, "coverage": None}
        raw_checkpoint_keys = (
            self.state.get("family_checkpoint_keys")
            if self.profile.family_minimum_quotas
            else None
        )
        checkpoint_keys = (
            {(str(key[0]), str(key[1])) for key in raw_checkpoint_keys}
            if isinstance(raw_checkpoint_keys, list)
            else set()
        )
        all_manifest = self.control_root / "all-input-manifest.json"
        write_input_manifest(
            all_manifest,
            runs,
            self.audit_root,
            self.dataset_root / "blobs" / "sha256",
            seed_manifest=self.seed_manifest,
            profile=self.profile,
            pinned_keys=checkpoint_keys,
            task_families=self.task_families,
        )
        build_dataset(all_manifest, self.audit_root)
        all_samples = processed_samples(self.audit_root)
        all_samples = [
            sample
            for sample in all_samples
            if (
                str(sample.get("provenance", {}).get("task_id") or "")
                not in self.task_contracts
                or str(sample.get("provenance", {}).get("contract_hash") or "")
                == self.task_contracts[
                    str(sample.get("provenance", {}).get("task_id") or "")
                ]
            )
        ]
        pinned_keys = (
            self._family_checkpoint_keys(all_samples)
            if self.profile.family_minimum_quotas
            else seed_sample_keys(all_samples, self.seed_manifest)
        )
        selected = select_samples(
            all_samples,
            profile=self.profile,
            pinned_keys=pinned_keys,
            task_families=self.task_families,
            require_complete=False,
        )
        pinned = self.dataset_root / "input_manifest.json"
        write_input_manifest(
            pinned,
            runs,
            self.dataset_root,
            self.dataset_root / "blobs" / "sha256",
            selected=selected,
            seed_manifest=self.seed_manifest,
            profile=self.profile,
            pinned_keys=pinned_keys,
            task_families=self.task_families,
        )
        result = build_dataset(pinned, self.dataset_root)
        manifest = Path(result["manifest"])
        quality = validate_dataset(manifest)
        leakage = audit_leakage(manifest)
        coverage = coverage_report(manifest)
        counts = sample_counts(
            processed_samples(self.dataset_root),
            self.profile,
            self.task_families,
            pinned_keys,
        )
        self.state.update(
            {
                "counts": counts,
                "input_manifest": str(pinned),
                "dataset_manifest": str(manifest),
                "audit": {
                    "quality": quality["status"],
                    "leakage": leakage["status"],
                    "coverage": coverage["status"],
                },
            }
        )
        self._save()
        return {"counts": counts, "quality": quality, "leakage": leakage, "coverage": coverage}

    def harvest(self) -> None:
        harvested = self.state["harvested"]
        queues = self.state["queues"]
        for run in self.complete_runs():
            workspaces = self._session_workspaces(run)
            for candidate_file in sorted(run.glob("round_*/candidates.jsonl")):
                records, _ = read_jsonl(candidate_file)
                frozen_path = candidate_file.parent / "frozen_input.json"
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                frozen_body = frozen.get("input") if isinstance(frozen.get("input"), Mapping) else {}
                for record in records:
                    body = record.get("candidate")
                    body = body if isinstance(body, Mapping) else {}
                    candidate_id = str(body.get("candidate_id") or "")
                    key = f"{run.name}:{candidate_id}"
                    verify = record.get("verify_result")
                    verify = verify if isinstance(verify, Mapping) else {}
                    evaluation = verify.get("evaluation")
                    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
                    workspace = workspaces.get(str(record.get("candidate_session_id") or ""))
                    if not workspace or not Path(workspace).is_dir():
                        continue
                    common = {
                        "case_id": str(frozen_body.get("contract", {}).get("case_id") or ""),
                        "source_run_dir": str(run),
                        "source_candidate_id": candidate_id,
                        "resume_workspace": workspace,
                        "baseline": frozen_body.get("baseline"),
                    }
                    if key + ":error" not in harvested and (
                        evaluation.get("compiled") is False
                        or (
                            evaluation.get("compiled") is True
                            and evaluation.get("correct") is False
                        )
                    ):
                        item = {
                            **common,
                            "harvest_id": key + ":error",
                            "mode": "error_recovery",
                            "error_feedback": {
                                "schema_version": "geak_exact_failure_v1",
                                "failed_workspace": workspace,
                                "candidate_id": candidate_id,
                                "verify_result": verify,
                            },
                        }
                        harvested[item["harvest_id"]] = item
                        queues["error_recovery"].append(item)
                    base = evaluation.get("baseline_ms") or frozen_body.get("baseline", {}).get("per_case_ms")
                    candidate = evaluation.get("candidate_ms")
                    regressions = {}
                    if isinstance(base, Mapping) and isinstance(candidate, Mapping):
                        regressions = {
                            str(name): {
                                "baseline_ms": float(base[name]),
                                "candidate_ms": float(candidate[name]),
                            }
                            for name in set(base) & set(candidate)
                            if float(candidate[name]) > float(base[name])
                        }
                    if key + ":regression" not in harvested and regressions:
                        item = {
                            **common,
                            "harvest_id": key + ":regression",
                            "mode": "regression_balance",
                            "per_case_benchmark": {
                                "schema_version": "geak_frozen_per_case_benchmark_v1",
                                "baseline_ms": dict(base),
                                "candidate_ms": dict(candidate),
                                "regressions": regressions,
                                "verify_result": verify,
                            },
                            "regression_constraints": {
                                "schema_version": "geak_frozen_regression_constraints_v1",
                                "maximum_ms": dict(base),
                                "rule": "no shared benchmark case may regress",
                            },
                        }
                        harvested[item["harvest_id"]] = item
                        queues["regression_balance"].append(item)
        self._save()

    @staticmethod
    def _session_workspaces(run: Path) -> dict[str, str]:
        records, _ = read_jsonl(run / "trajectory.jsonl")
        result = {}
        for record in records:
            if record.get("event") != "environment_create":
                continue
            payload = record.get("payload")
            if isinstance(payload, Mapping) and payload.get("session_id") and payload.get("workspace"):
                result[str(payload["session_id"])] = str(payload["workspace"])
        return result

    def _gpu_busy(self, gpu: int) -> bool:
        forced = {
            int(value)
            for value in os.environ.get("GEAK_CONTROLLER_BUSY_GPUS", "").split(",")
            if value.strip().isdigit()
        }
        if gpu in forced:
            return True
        try:
            proc = subprocess.run(
                ["/opt/rocm/bin/rocm-smi", "--showpids", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if proc.returncode:
            return False
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return False
        card = next(
            (
                value
                for key, value in payload.items()
                if re.search(rf"(?:card|gpu|\D){gpu}(?:\D|$)", str(key), re.IGNORECASE)
            ),
            None,
        )
        if not isinstance(card, Mapping):
            return False
        text = json.dumps(card).lower()
        return bool(re.search(r'"pid[^"]*"\s*:\s*"?[1-9]\d*', text))

    def _choose_case(self, lane: str, mode: str) -> Case:
        candidates = [case for case in self.cases if case.lane == lane]
        usage = self.state["attempted_case_modes"]
        family_counts = self.state.get("counts", {}).get(
            "post_checkpoint_top10_family",
            {},
        )
        minimums = self.profile.family_minimum_quotas or {}
        deficits = {
            family: max(0, quota - int(family_counts.get(family, 0)))
            for family, quota in minimums.items()
        }
        attempted_families: Counter[str | None] = Counter()
        for known_case in self.cases:
            attempted_families[known_case.top10_family] += int(
                usage.get(f"{known_case.case_id}:{mode}", 0)
            )
        post_checkpoint_total = max(
            0,
            int(self.state.get("counts", {}).get("accepted", 0))
            - self.profile.checkpoint_sample_count,
        )
        accepted_top10 = sum(int(value) for value in family_counts.values())
        accepted_non_top10 = max(0, post_checkpoint_total - accepted_top10)
        non_top10_capacity = (
            self.profile.sample_count
            - self.profile.checkpoint_sample_count
            - sum(minimums.values())
        )

        def family_priority(case: Case) -> tuple[float, float, int, str]:
            family = case.top10_family
            if family in minimums:
                quota = max(1, int(minimums[family]))
                normalized_attempts = (
                    attempted_families[family] / quota
                    if deficits.get(family, 0) > 0
                    else float("inf")
                )
                normalized_deficit = deficits.get(family, 0) / quota
            else:
                quota = max(1, non_top10_capacity)
                normalized_attempts = (
                    attempted_families[None] / quota
                    if accepted_non_top10 < non_top10_capacity
                    else float("inf")
                )
                normalized_deficit = max(
                    0, non_top10_capacity - accepted_non_top10
                ) / quota
            return (
                normalized_attempts,
                -normalized_deficit,
                int(usage.get(f"{case.case_id}:{mode}", 0)),
                case.case_id,
            )

        ordered = sorted(
            candidates,
            key=family_priority,
        )
        if getattr(self, "_gang_running_for_next_spec", False):
            available_count = len(
                getattr(self, "_available_gpus_for_next_spec", ())
            )
            runnable = [
                case for case in ordered if case.world_size <= available_count
            ]
            if runnable:
                return runnable[0]
        return ordered[0]

    def _next_spec(self, gpu: int, reserved: set[str]) -> AttemptSpec | None:
        if self.max_attempts is not None and self.state["attempts_started"] >= self.max_attempts:
            return None
        counts = self.state["counts"]
        lane_counts = counts.get("lane", {})
        cell_counts = counts.get("cell", {})
        deficits = {
            (mode, lane): quota
            - int(cell_counts.get(mode, {}).get(lane, 0))
            for mode, lanes in self.profile.cell_quotas.items()
            for lane, quota in lanes.items()
        }
        mode = ""
        target_lane = ""
        context = None
        resume = None
        harvest_id = None
        case_id = ""
        case_lanes = {case.case_id: case.lane for case in self.cases}
        for candidate_mode, candidate_lane in sorted(
            deficits,
            key=lambda value: (-deficits[value], value),
        ):
            if deficits[(candidate_mode, candidate_lane)] <= 0:
                continue
            if candidate_mode in PRIMARY_MODES:
                mode = candidate_mode
                target_lane = candidate_lane
                break
            queue = self.state["queues"][candidate_mode]
            item = next(
                (
                    value
                    for value in queue
                    if value["harvest_id"] not in reserved
                    and not self.state["harvested"]
                    .get(value["harvest_id"], {})
                    .get("consumed")
                    and sum(
                        attempt.get("harvest_id") == value["harvest_id"]
                        for attempt in self.state["attempts"]
                    )
                    < HARVEST_MAX_ATTEMPTS
                    and Path(value["resume_workspace"]).is_dir()
                    and case_lanes.get(value["case_id"]) == candidate_lane
                ),
                None,
            )
            if item is not None:
                mode = candidate_mode
                target_lane = candidate_lane
                context = {key: value for key, value in item.items() if key not in {"mode", "consumed"}}
                resume = item["resume_workspace"]
                harvest_id = item["harvest_id"]
                case_id = item["case_id"]
                break
        if not mode:
            # Generate more independent failures/regressions without admitting
            # extra primary samples beyond their fixed final quotas.
            mode = "direction_conditioned"
            family_counts = counts.get("post_checkpoint_top10_family", {})
            family_deficits = {
                family: max(0, int(quota) - int(family_counts.get(family, 0)))
                for family, quota in (self.profile.family_minimum_quotas or {}).items()
            }

            def lane_family_deficit(lane: str) -> int:
                supported = {
                    case.top10_family
                    for case in self.cases
                    if case.lane == lane and case.top10_family is not None
                }
                return sum(family_deficits.get(family, 0) for family in supported)

            target_lane = min(
                self.profile.lane_quotas,
                key=lambda lane: (
                    -lane_family_deficit(lane),
                    int(lane_counts.get(lane, 0))
                    - self.profile.lane_quotas[lane],
                ),
            )
        lane = next(
            (
                lane
                for lane, _ in sorted(
                    self.profile.lane_quotas.items(),
                    key=lambda pair: (
                        int(lane_counts.get(pair[0], 0)) - pair[1],
                        pair[0],
                    ),
                )
                if lane == target_lane
                and (
                    not case_id
                    or any(case.case_id == case_id and case.lane == lane for case in self.cases)
                )
            ),
            None,
        )
        if lane is None:
            return None
        case = next((item for item in self.cases if item.case_id == case_id), None) if case_id else self._choose_case(lane, mode)
        if case is None:
            return None
        available_gpus = tuple(
            candidate
            for candidate in getattr(self, "_available_gpus_for_next_spec", (gpu,))
            if candidate in self.gpus and candidate != 0
        )
        if gpu not in available_gpus:
            return None
        ordered_gpus = (gpu,) + tuple(
            candidate for candidate in available_gpus if candidate != gpu
        )
        if len(ordered_gpus) < case.world_size:
            return None
        attempt_gpus = ordered_gpus[: case.world_size]
        number = int(self.state["attempts_started"]) + 1
        key = f"{case.case_id}:{mode}"
        self.state["attempted_case_modes"][key] = int(self.state["attempted_case_modes"].get(key, 0)) + 1
        self.state["attempts_started"] = number
        if harvest_id:
            reserved.add(harvest_id)
        spec = AttemptSpec(
            number,
            case.case_id,
            case.lane,
            mode,
            gpu,
            context,
            resume,
            harvest_id,
            attempt_gpus,
            case.world_size,
        )
        self.state["attempts"].append(
            {
                "number": number,
                "case_id": case.case_id,
                "lane": case.lane,
                "mode": mode,
                "gpu": gpu,
                "gpus": list(attempt_gpus),
                "world_size": case.world_size,
                "harvest_id": harvest_id,
                "status": "started",
                "started_at": time.time(),
            }
        )
        self._save()
        return spec

    def _attempt_config(self, spec: AttemptSpec) -> Path:
        payload = yaml.safe_load(self.base_config.read_text(encoding="utf-8")) or {}
        for key in ("geak_root", "aiter_root", "generated_template_root"):
            if key in payload:
                value = Path(str(payload[key])).expanduser()
                payload[key] = str(
                    value.resolve()
                    if value.is_absolute()
                    else (self.base_config.parent / value).resolve()
                )
        payload.update(
            {
                "cases_path": str(self.cases_path),
                "trajectory_root": str(self.trajectory_root),
                "gpu_ids": spec.gpu_ids,
                "sft_enabled": True,
                "sft_dataset_root": str(self.dataset_root),
                "sft_task_type": spec.mode,
                "keep_sessions": True,
            }
        )
        directory = self.attempt_root / f"{spec.number:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        if spec.context is not None:
            atomic_json(directory / "context.json", spec.context)
        return path

    def _run_attempt(self, spec: AttemptSpec) -> dict[str, Any]:
        directory = self.attempt_root / f"{spec.number:05d}"
        config = self._attempt_config(spec)
        argv = [
            sys.executable,
            "-m",
            "multi_tune_agent.cli",
            "--config",
            str(config),
            "run",
            "--case",
            spec.case_id,
            "--sft-task-type",
            spec.mode,
        ]
        if spec.context is not None:
            argv.extend(["--context-json", str(directory / "context.json")])
        if spec.resume_workspace:
            argv.extend(["--resume-workspace", spec.resume_workspace])
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parents[2]
        python_path = os.pathsep.join((str(project_root / "src"), str(project_root)))
        if env.get("PYTHONPATH"):
            python_path += os.pathsep + env["PYTHONPATH"]
        env.update(
            {
                "HIP_VISIBLE_DEVICES": spec.gpu_ids,
                "GEAK_GPU_ALLOWED": spec.gpu_ids,
                "GEAK_GPU_GANG": "1" if spec.world_size > 1 else "0",
                "GEAK_GPU_REQUIRE_IDLE": "1",
                "PYTHONPATH": python_path,
            }
        )
        contention_deadline = time.monotonic() + 1200
        contention_errors: list[str] = []
        while True:
            proc = subprocess.run(
                argv,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            combined = proc.stdout + "\n" + proc.stderr
            if not FAILURE_WORDS.search(combined) or time.monotonic() >= contention_deadline:
                break
            contention_errors.append(proc.stderr)
            time.sleep(5)
        (directory / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(
            "\n".join((*contention_errors, proc.stderr)),
            encoding="utf-8",
        )
        run_dirs: list[str] = []
        try:
            summaries = json.loads(proc.stdout)
            if isinstance(summaries, list):
                run_dirs = [
                    str(item["run_dir"])
                    for item in summaries
                    if isinstance(item, Mapping) and item.get("run_dir")
                ]
        except json.JSONDecodeError:
            pass
        positive_candidates = 0
        for run_dir in run_dirs:
            try:
                manifest = json.loads((Path(run_dir) / "sft_manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("dataset_eligible"):
                positive_candidates += int(manifest.get("positive_candidate_count") or 0)
        return {
            "returncode": proc.returncode,
            "status": "gpu_contention" if FAILURE_WORDS.search(combined) else ("completed" if proc.returncode == 0 else "failed"),
            "run_dirs": run_dirs,
            "positive_candidates": positive_candidates,
            "finished_at": time.time(),
        }

    def run(self) -> int:
        self.control_root.mkdir(parents=True, exist_ok=True)
        lock = (self.control_root / "controller.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another production controller owns this wave") from exc
        try:
            stale_attempts = [
                attempt
                for attempt in self.state["attempts"]
                if attempt.get("status") == "started"
            ]
            for attempt in stale_attempts:
                attempt.update(
                    {
                        "status": "controller_restarted",
                        "finished_at": time.time(),
                    }
                )
            if stale_attempts:
                self._save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
                futures: dict[concurrent.futures.Future[dict[str, Any]], AttemptSpec] = {}
                accepting = True
                needs_rebuild = True
                while True:
                    if needs_rebuild:
                        audit = self.rebuild()
                        self.harvest()
                        gates_pass = all(
                            audit[name] is not None and audit[name]["status"] == "pass"
                            for name in ("quality", "leakage", "coverage")
                        )
                        if quotas_pass(audit["counts"], self.profile) and gates_pass:
                            accepting = False
                            if not futures:
                                self.state["status"] = "complete"
                                self.state["stop_reason"] = (
                                    "all_processed_deduplicated_quotas_and_gates_pass"
                                )
                                self._save()
                                return 0
                        elif (
                            self.max_attempts is not None
                            and self.state["attempts_started"] >= self.max_attempts
                        ):
                            accepting = False
                            if not futures:
                                self.state["status"] = "attempt_ceiling"
                                self.state["stop_reason"] = "max_attempts_reached"
                                self._save()
                                return 2
                        needs_rebuild = False

                    running_gpus = {
                        gpu
                        for spec in futures.values()
                        for gpu in spec.gpus
                    }
                    reserved = {
                        spec.harvest_id
                        for spec in futures.values()
                        if spec.harvest_id is not None
                    }
                    if accepting:
                        free_gpus = [
                            gpu
                            for gpu in self.gpus
                            if gpu not in running_gpus and not self._gpu_busy(gpu)
                        ]
                        while free_gpus:
                            self._available_gpus_for_next_spec = tuple(free_gpus)
                            self._gang_running_for_next_spec = any(
                                spec.world_size > 1 for spec in futures.values()
                            )
                            spec = self._next_spec(free_gpus[0], reserved)
                            if spec is None:
                                break
                            if (
                                not set(spec.gpus).issubset(free_gpus)
                                or len(spec.gpus) != spec.world_size
                            ):
                                raise RuntimeError(
                                    "attempt requested GPUs outside its atomic reservation"
                                )
                            futures[pool.submit(self._run_attempt, spec)] = spec
                            free_gpus = [
                                gpu for gpu in free_gpus if gpu not in spec.gpus
                            ]
                        self.__dict__.pop("_available_gpus_for_next_spec", None)
                        self.__dict__.pop("_gang_running_for_next_spec", None)

                    if not futures:
                        time.sleep(self.poll_seconds)
                        needs_rebuild = True
                        continue

                    done, _ = concurrent.futures.wait(
                        futures,
                        timeout=self.poll_seconds,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for future in done:
                        spec = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "status": "controller_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "finished_at": time.time(),
                            }
                        attempt = next(item for item in self.state["attempts"] if item["number"] == spec.number)
                        attempt.update(result)
                        if spec.harvest_id and int(result.get("positive_candidates", 0)) > 0:
                            self.state["harvested"][spec.harvest_id]["consumed"] = True
                        self._save()
                    needs_rebuild = True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave-profile",
        choices=sorted(BUILTIN_PROFILES),
        default="300",
        help="validated built-in quota/path preset (default: 300)",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="YAML/JSON profile with name, dataset_version, and exact quota maps",
    )
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--trajectory-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=None,
        help="pinned legacy wave manifest whose accepted samples seed the quotas",
    )
    parser.add_argument(
        "--gpus",
        default="1-7",
        help="comma/range list; GPU0 is excluded because it hosts the model server",
    )
    parser.add_argument("--max-attempts", type=int, help="explicit ceiling; omitted means run to quota")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def load_wave_profile(path: Path) -> WaveProfile:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("wave profile file must contain an object")
    allowed = {
        "name",
        "dataset_version",
        "mode_quotas",
        "lane_quotas",
        "cell_quotas",
        "allowed_case_splits",
        "family_minimum_quotas",
        "checkpoint_sample_count",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown wave profile fields: {', '.join(sorted(unknown))}")
    try:
        return WaveProfile(
            name=str(payload["name"]),
            dataset_version=str(payload["dataset_version"]),
            mode_quotas=payload["mode_quotas"],
            lane_quotas=payload["lane_quotas"],
            cell_quotas=payload["cell_quotas"],
            allowed_case_splits=payload.get("allowed_case_splits", ("train", "dev")),
            family_minimum_quotas=payload.get("family_minimum_quotas"),
            checkpoint_sample_count=payload.get("checkpoint_sample_count", 0),
        )
    except KeyError as exc:
        raise ValueError(f"wave profile is missing {exc.args[0]!r}") from exc


def preset_paths(name: str) -> tuple[Path, Path, Path, Path]:
    if name == "2000":
        return (
            WAVE_2000_CASES,
            WAVE_2000_TRAJECTORY_ROOT,
            WAVE_2000_DATASET_ROOT,
            WAVE_2000_SEED_MANIFEST,
        )
    return (
        DEFAULT_CASES,
        DEFAULT_TRAJECTORY_ROOT,
        DEFAULT_DATASET_ROOT,
        DEFAULT_SEED_MANIFEST,
    )


def resolve_seed_manifest(
    explicit: Path | None,
    default: Path,
    *,
    custom_profile: bool,
    profile: WaveProfile,
) -> Path | None:
    """Avoid implicitly importing another wave into a zero-checkpoint profile."""
    if explicit is not None:
        return explicit
    if custom_profile and profile.checkpoint_sample_count == 0:
        return None
    return default


def parse_gpus(value: str) -> tuple[int, ...]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+-\d+", part):
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError(f"invalid GPU range: {part}")
            result.update(range(start, end + 1))
        elif part.isdigit():
            result.add(int(part))
        else:
            raise ValueError(f"invalid GPU selector: {part!r}")
    if not result:
        raise ValueError("at least one GPU is required")
    if 0 in result:
        raise ValueError("GPU0 is reserved for the model server")
    return tuple(sorted(result))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_attempts is not None and args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    profile = load_wave_profile(args.profile_file) if args.profile_file else BUILTIN_PROFILES[args.wave_profile]
    default_cases, default_trajectory, default_dataset, default_seed = preset_paths(
        args.wave_profile
    )
    controller = ProductionController(
        cases_path=args.cases or default_cases,
        trajectory_root=args.trajectory_root or default_trajectory,
        dataset_root=args.dataset_root or default_dataset,
        base_config=args.config,
        gpus=parse_gpus(args.gpus),
        max_attempts=args.max_attempts,
        poll_seconds=args.poll_seconds,
        seed_manifest=resolve_seed_manifest(
            args.seed_manifest,
            default_seed,
            custom_profile=args.profile_file is not None,
            profile=profile,
        ),
        profile=profile,
    )
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
