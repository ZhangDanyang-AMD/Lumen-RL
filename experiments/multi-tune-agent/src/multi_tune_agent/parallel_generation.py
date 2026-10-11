"""Safe multi-GPU driver for versioned generation request manifests."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .config import MultiTuneConfig


GPU_IDS = tuple(range(1, 8))
COLLECTABLE_SPLITS = frozenset(("train", "dev"))
REQUIRED_TASK_FILES = (
    "config.yaml",
    "kernel.py",
    "metadata.json",
    "scripts/task_runner.py",
)
REQUIRED_BASE_TRUST_FLAGS = (
    "trusted",
    "static_valid",
    "compiled",
    "correct",
    "performance_valid",
)
REQUIRED_BASE_TRUST_COMMANDS = ("compile", "correctness", "performance")


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a mapping")
    return payload


def _balanced_smoke_sample(
    requests: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Deterministically alternate target lanes in a bounded smoke sample."""
    lanes: dict[str, list[dict[str, Any]]] = {}
    for raw in requests:
        item = dict(raw)
        provenance = item.get("seed_provenance")
        lane = (
            str(provenance.get("target_lane") or "unknown")
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        lanes.setdefault(lane, []).append(item)
    if len(lanes) < 2:
        return [dict(item) for item in requests[:limit]]
    for items in lanes.values():
        items.sort(key=lambda item: str(item["id"]))
    selected: list[dict[str, Any]] = []
    lane_names = sorted(lanes)
    while len(selected) < limit:
        progressed = False
        for lane in lane_names:
            if lanes[lane]:
                selected.append(lanes[lane].pop(0))
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def load_requests(
    path: Path,
    smoke_limit: int | None = None,
    *,
    split: str = "train",
    top10_only: bool = False,
) -> list[dict[str, Any]]:
    """Load and deterministically order a source-complete request manifest."""
    if split not in COLLECTABLE_SPLITS:
        raise ValueError("split must be train or dev")
    selected_split = split
    payload = _load_mapping(path, "generation manifest")
    if payload.get("version") != 1:
        raise ValueError("generation manifest must have version: 1")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("generation manifest requires a non-empty requests list")
    if smoke_limit is not None and smoke_limit < 1:
        raise ValueError("--smoke-limit must be positive")

    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_requests, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request {index} must be a mapping")
        item = dict(raw)
        case_id = str(item.get("id") or "").strip()
        request = str(item.get("request") or "").strip()
        if not case_id or not request:
            raise ValueError(f"request {index} requires id and request")
        if case_id in seen:
            raise ValueError(f"duplicate request ID: {case_id}")
        seen.add(case_id)
        provenance = item.get("seed_provenance")
        request_split = (
            str(provenance.get("split_group") or "").strip()
            if isinstance(provenance, Mapping)
            else ""
        )
        if request_split == "held_out":
            raise ValueError(f"held_out request is forbidden: {case_id}")
        if not request_split:
            raise ValueError(f"request {case_id} is missing seed_provenance.split_group")
        if request_split not in COLLECTABLE_SPLITS:
            raise ValueError(
                f"request {case_id} has non-collectable split: {request_split}"
            )
        has_top10_family = bool(
            isinstance(provenance, Mapping) and provenance.get("top10_family")
        )
        if request_split == selected_split and (not top10_only or has_top10_family):
            requests.append(item)

    requests.sort(key=lambda item: str(item["id"]))
    if not requests:
        raise ValueError(f"generation manifest has no {selected_split} requests")
    return (
        _balanced_smoke_sample(requests, smoke_limit)
        if smoke_limit is not None
        else requests
    )


def shard_requests(
    requests: Sequence[Mapping[str, Any]],
    gpu_ids: Sequence[int] = GPU_IDS,
) -> dict[int, list[dict[str, Any]]]:
    """Round-robin sorted requests across fixed GPU IDs."""
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be non-empty and unique")
    ordered = sorted((dict(item) for item in requests), key=lambda item: str(item["id"]))
    shards = {gpu_id: [] for gpu_id in gpu_ids}
    for index, item in enumerate(ordered):
        shards[gpu_ids[index % len(gpu_ids)]].append(item)
    return shards


def _yaml_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(child) for child in value]
    return value


def prepare_shards(
    base_config_path: Path,
    manifest_path: Path,
    work_root: Path,
    requests: Sequence[Mapping[str, Any]],
    gpu_ids: Sequence[int] = GPU_IDS,
) -> dict[int, dict[str, Path]]:
    """Create stable, isolated inputs and mutable roots for every GPU."""
    config = MultiTuneConfig.from_yaml(base_config_path)
    manifest_header = _load_mapping(manifest_path, "generation manifest")
    manifest_header.pop("requests", None)
    paths: dict[int, dict[str, Path]] = {}
    for gpu_id, items in shard_requests(requests, gpu_ids).items():
        root = work_root / f"gpu{gpu_id}"
        trajectory_root = root / "trajectories"
        template_root = root / "generated-templates"
        catalog = root / "generated-cases.yaml"
        shard_manifest = root / "generation-requests.yaml"
        shard_config = root / "config.yaml"
        log_path = root / "generate.log"
        root.mkdir(parents=True, exist_ok=True)
        template_root.mkdir(parents=True, exist_ok=True)
        if not (template_root / "templates.yaml").exists():
            _atomic_write_yaml(template_root / "templates.yaml", {"templates": []})
        _atomic_write_yaml(shard_manifest, {**manifest_header, "requests": items})
        values = asdict(config)
        values.update(
            {
                "gpu_ids": str(gpu_id),
                "cases_path": catalog,
                "trajectory_root": trajectory_root,
                "generated_template_root": template_root,
                "sft_enabled": False,
            }
        )
        _atomic_write_yaml(shard_config, _yaml_value(values))
        paths[gpu_id] = {
            "root": root,
            "config": shard_config,
            "manifest": shard_manifest,
            "catalog": catalog,
            "templates": template_root / "templates.yaml",
            "trajectory_root": trajectory_root,
            "log": log_path,
        }
    return paths


def run_shards(
    shard_paths: Mapping[int, Mapping[str, Path]],
    *,
    python: str = sys.executable,
) -> dict[int, int]:
    """Run non-empty shards concurrently and retain combined stdout/stderr logs."""
    running: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    results: dict[int, int] = {}
    for gpu_id, paths in sorted(shard_paths.items()):
        manifest = _load_mapping(paths["manifest"], "shard manifest")
        if not manifest.get("requests"):
            results[gpu_id] = 0
            continue
        log_handle = paths["log"].open("a", encoding="utf-8")
        command = [
            python,
            "-m",
            "multi_tune_agent.cli",
            "--config",
            str(paths["config"]),
            "generate",
            "--manifest",
            str(paths["manifest"]),
            "--output-catalog",
            str(paths["catalog"]),
            "--stream",
        ]
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_handle.close()
            raise
        running[gpu_id] = (process, log_handle)

    for gpu_id, (process, log_handle) in running.items():
        try:
            results[gpu_id] = process.wait()
        finally:
            log_handle.close()
    return results


def _latest_statuses(paths: Mapping[str, Path]) -> dict[str, str]:
    results_path = (
        paths["trajectory_root"]
        / "requests"
        / f"{paths['manifest'].stem}-generation-results.jsonl"
    )
    statuses: dict[str, str] = {}
    if not results_path.is_file():
        return statuses
    for line_number, raw in enumerate(
        results_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{results_path}:{line_number}: invalid JSON") from exc
        if isinstance(row, Mapping) and row.get("case_id") and row.get("status"):
            statuses[str(row["case_id"])] = str(row["status"])
    return statuses


def _resolve_catalog_path(catalog: Path, raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    return (catalog.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_complete_task_files(case_id: str, kernel_path: Path) -> dict[str, Any]:
    if not kernel_path.is_dir():
        raise ValueError(f"task {case_id} kernel path is not a directory")
    for entry in kernel_path.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"task {case_id} kernel path contains a symlink")
    for relative in REQUIRED_TASK_FILES:
        required = kernel_path / relative
        if required.is_symlink() or not required.is_file():
            raise ValueError(f"task {case_id} is missing {relative}")
    try:
        metadata = json.loads(
            (kernel_path / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"task {case_id} has invalid metadata.json") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"task {case_id} metadata.json must contain an object")
    return metadata


def _already_trusted_base_tasks(
    catalog: Path,
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate production tasks whose original worker registries are unavailable."""
    payload = _load_mapping(catalog, "trusted base catalog")
    raw_tasks = payload.get("tasks") or []
    if not isinstance(raw_tasks, list):
        raise ValueError("trusted base catalog 'tasks' must be a list")
    tasks: dict[str, dict[str, Any]] = {}
    for raw in raw_tasks:
        if not isinstance(raw, Mapping) or not raw.get("id"):
            raise ValueError("trusted base catalog contains an invalid task")
        task = dict(raw)
        case_id = str(task["id"])
        if case_id in tasks:
            raise ValueError(f"duplicate trusted base task ID: {case_id}")
        provenance = task.get("provenance")
        if not isinstance(provenance, Mapping) or not isinstance(
            provenance.get("case_seed"), Mapping
        ):
            raise ValueError(f"trusted base task {case_id} lacks case_seed provenance")
        task_split = provenance["case_seed"].get("split_group")
        if task_split != split:
            raise ValueError(
                f"trusted base task {case_id} split {task_split!r} does not match "
                f"selected split {split!r}"
            )
        contract_hash = str(task.get("contract_hash") or "")
        if not contract_hash or provenance.get("contract_hash") != contract_hash:
            raise ValueError(f"trusted base task {case_id} has invalid contract provenance")
        kernel_path = _resolve_catalog_path(catalog, task.get("kernel_path"))
        metadata = _validate_complete_task_files(case_id, kernel_path)
        if metadata.get("contract_hash") != contract_hash:
            raise ValueError(f"trusted base task {case_id} metadata hash mismatch")
        trust = metadata.get("trust")
        if not isinstance(trust, Mapping) or any(
            trust.get(flag) is not True for flag in REQUIRED_BASE_TRUST_FLAGS
        ):
            raise ValueError(
                f"trusted base task {case_id} lacks complete metadata trust flags"
            )
        commands = trust.get("commands")
        if not isinstance(commands, Mapping):
            raise ValueError(f"trusted base task {case_id} lacks trust commands")
        for name in REQUIRED_BASE_TRUST_COMMANDS:
            result = commands.get(name)
            if (
                not isinstance(result, Mapping)
                or result.get("ok") is not True
                or result.get("returncode") != 0
                or result.get("timed_out") is not False
            ):
                raise ValueError(
                    f"trusted base task {case_id} has invalid {name} trust result"
                )
        task["kernel_path"] = str(kernel_path)
        tasks[case_id] = task
    return payload, tasks


def initialize_from_base_catalog(
    base_catalog: Path,
    output_catalog: Path,
    *,
    split: str = "train",
) -> int:
    """Atomically seed or reconcile output with independently trusted base tasks."""
    if split not in COLLECTABLE_SPLITS:
        raise ValueError("split must be train or dev")
    base_payload, base_tasks = _already_trusted_base_tasks(base_catalog, split=split)
    payload = {key: value for key, value in base_payload.items() if key != "tasks"}
    existing_tasks: dict[str, dict[str, Any]] = {}
    if output_catalog.is_file():
        existing_payload, existing_tasks = _already_trusted_base_tasks(
            output_catalog, split=split
        )
        payload.update(
            {key: value for key, value in existing_payload.items() if key != "tasks"}
        )
    for case_id, task in base_tasks.items():
        existing = existing_tasks.get(case_id)
        if existing is not None and existing != task:
            raise ValueError(f"output conflicts with trusted base task ID: {case_id}")
        existing_tasks[case_id] = task
    payload["tasks"] = [existing_tasks[key] for key in sorted(existing_tasks)]
    _atomic_write_yaml(output_catalog, payload)
    return len(base_tasks)


def _verified_registry(paths: Mapping[str, Path]) -> dict[str, Path]:
    payload = _load_mapping(paths["templates"], "template registry")
    records = payload.get("templates") or []
    if not isinstance(records, list):
        raise ValueError("template registry 'templates' must be a list")
    verified: dict[str, Path] = {}
    root = paths["templates"].parent.resolve()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("template registry entry must be a mapping")
        contract_hash = str(raw.get("contract_hash") or "")
        template = Path(str(raw.get("template_path") or "")).expanduser()
        if not template.is_absolute():
            template = paths["templates"].parent / template
        template = template.resolve()
        try:
            template.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"registered template escapes isolated root: {template}") from exc
        metadata = json.loads((template / "metadata.json").read_text(encoding="utf-8"))
        if (
            not contract_hash
            or metadata.get("contract_hash") != contract_hash
            or not isinstance(metadata.get("trust"), Mapping)
            or metadata["trust"].get("trusted") is not True
        ):
            raise ValueError(f"untrusted template registry entry: {template}")
        verified[contract_hash] = template
    return verified


def _verified_tasks(
    paths: Mapping[str, Path],
    expected_ids: set[str],
    selected_split: str,
) -> list[dict[str, Any]]:
    catalog = paths["catalog"]
    if not catalog.is_file():
        return []
    payload = _load_mapping(catalog, "shard catalog")
    raw_tasks = payload.get("tasks") or []
    if not isinstance(raw_tasks, list):
        raise ValueError("shard catalog 'tasks' must be a list")
    statuses = _latest_statuses(paths)
    registry = _verified_registry(paths)
    accepted: list[dict[str, Any]] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValueError("shard catalog task must be a mapping")
        task = dict(raw)
        case_id = str(task.get("id") or "")
        if case_id not in expected_ids:
            continue
        if statuses.get(case_id) not in {"generated", "already_registered"}:
            continue
        provenance = task.get("provenance")
        contract_hash = str(task.get("contract_hash") or "")
        kernel_path = _resolve_catalog_path(catalog, task.get("kernel_path"))
        if not isinstance(provenance, Mapping) or not isinstance(
            provenance.get("case_seed"), Mapping
        ):
            raise ValueError(f"task {case_id} lacks case_seed provenance")
        if provenance.get("contract_hash") != contract_hash:
            raise ValueError(f"task {case_id} has invalid contract provenance")
        task_split = provenance["case_seed"].get("split_group")
        if task_split != selected_split:
            raise ValueError(
                f"task {case_id} split {task_split!r} does not match "
                f"selected split {selected_split!r}"
            )
        if registry.get(contract_hash) != kernel_path:
            raise ValueError(f"task {case_id} is not backed by its verified registry")
        _validate_complete_task_files(case_id, kernel_path)
        task["kernel_path"] = str(kernel_path)
        accepted.append(task)
    return accepted


def merge_catalogs(
    output_catalog: Path,
    shard_paths: Mapping[int, Mapping[str, Path]],
    requests: Sequence[Mapping[str, Any]],
    *,
    split: str = "train",
) -> int:
    """Validate shard outputs and atomically upsert them into production."""
    if split not in COLLECTABLE_SPLITS:
        raise ValueError("split must be train or dev")
    expected_ids = {str(item["id"]) for item in requests}
    merged: dict[str, dict[str, Any]] = {}
    payload: dict[str, Any] = {}
    if output_catalog.is_file():
        payload, merged = _already_trusted_base_tasks(
            output_catalog, split=split
        )

    generated: dict[str, dict[str, Any]] = {}
    for gpu_id, paths in sorted(shard_paths.items()):
        for task in _verified_tasks(paths, expected_ids, split):
            case_id = str(task["id"])
            if case_id in generated and generated[case_id] != task:
                raise ValueError(f"conflicting shard task ID: {case_id}")
            generated[case_id] = task
    for case_id, task in generated.items():
        existing = merged.get(case_id)
        if existing is not None and existing != task:
            raise ValueError(f"generated task would displace existing ID: {case_id}")
        if existing is None:
            merged[case_id] = task
    payload["tasks"] = [merged[key] for key in sorted(merged)]
    _atomic_write_yaml(output_catalog, payload)
    return len(generated)


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--output-catalog", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=sorted(COLLECTABLE_SPLITS),
        default="train",
        help="collect exactly one split (default: train)",
    )
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument(
        "--top10-only",
        action="store_true",
        help="collect only requests carrying seed_provenance.top10_family",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    lock_path = work_root / ".orchestrator.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another orchestrator is using work root: {work_root}"
            ) from exc
        requests = load_requests(
            manifest, split=args.split, top10_only=args.top10_only
        )
        base_count = initialize_from_base_catalog(
            args.base_catalog.expanduser().resolve(),
            args.output_catalog.expanduser().resolve(),
            split=args.split,
        )
        output_payload = _load_mapping(
            args.output_catalog.expanduser().resolve(), "output catalog"
        )
        existing_ids = {
            str(task.get("id") or "")
            for task in output_payload.get("tasks", [])
            if isinstance(task, Mapping)
        }
        requests = [
            request for request in requests if str(request["id"]) not in existing_ids
        ]
        if args.smoke_limit is not None:
            requests = _balanced_smoke_sample(requests, args.smoke_limit)
        if not requests:
            raise ValueError("generation manifest has no requests outside the base catalog")
        shards = prepare_shards(
            args.config.expanduser().resolve(),
            manifest,
            work_root,
            requests,
        )
        return_codes = run_shards(shards, python=args.python)
        merged = merge_catalogs(
            args.output_catalog.expanduser().resolve(),
            shards,
            requests,
            split=args.split,
        )
    summary = {
        "requests": len(requests),
        "base_tasks": base_count,
        "merged": merged,
        "split": args.split,
        "return_codes": return_codes,
        "work_root": str(work_root),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if all(code == 0 for code in return_codes.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
