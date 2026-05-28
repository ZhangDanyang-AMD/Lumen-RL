"""Dispatch and collect DataProto across workers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from lumenrl.core.protocol import DataProto


class DispatchMode(str, Enum):
    """Supported dispatch semantics for worker calls."""

    DP_COMPUTE_PROTO = "dp_compute_proto"
    ONE_TO_ALL = "one_to_all"
    ALL_TO_ALL = "all_to_all"


def _normalize_mode(mode: DispatchMode | str | None) -> DispatchMode:
    if mode is None:
        return DispatchMode.DP_COMPUTE_PROTO
    if isinstance(mode, DispatchMode):
        return mode
    return DispatchMode(mode)


def _round_robin_chunks(chunks: list[DataProto], num_workers: int) -> list[DataProto]:
    if len(chunks) == num_workers:
        return chunks
    if not chunks:
        return [DataProto() for _ in range(num_workers)]
    return [chunks[i % len(chunks)] for i in range(num_workers)]


def _build_nd_dispatch(
    data: DataProto,
    num_workers: int,
    mesh_mapping: list[int] | None,
) -> list[DataProto]:
    if mesh_mapping is None:
        return data.split(num_workers)
    if len(mesh_mapping) != num_workers:
        raise ValueError("mesh_mapping length must equal num_workers.")
    if not mesh_mapping:
        return []

    num_groups = max(mesh_mapping) + 1
    grouped = data.split(num_groups)
    return [grouped[group_id] for group_id in mesh_mapping]


def dispatch_proto(
    data: DataProto,
    num_workers: int,
    mode: DispatchMode | str | None = None,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    """Create per-worker inputs according to dispatch mode.

    `lazy_state` + `lazy_key` allow caller-side caching of expensive ND mapping.
    """
    dispatch_mode = _normalize_mode(mode)
    if num_workers <= 0:
        return []

    if dispatch_mode == DispatchMode.ONE_TO_ALL:
        return [data for _ in range(num_workers)]
    if dispatch_mode == DispatchMode.ALL_TO_ALL:
        chunks = data.split(num_workers)
        return _round_robin_chunks(chunks, num_workers)

    mapping = mesh_mapping
    if lazy_state is not None and lazy_key is not None:
        cached = lazy_state.get(lazy_key)
        if cached is not None and isinstance(cached, list):
            mapping = cached
        elif mesh_mapping is not None:
            lazy_state[lazy_key] = mesh_mapping
    return _build_nd_dispatch(data, num_workers, mapping)


def collect_proto(
    results: list[DataProto],
    mode: DispatchMode | str | None = None,
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    """Collect worker outputs into a merged DataProto.

    For ONE_TO_ALL, callers can optionally deduplicate repeated shared results.
    """
    dispatch_mode = _normalize_mode(mode)
    if not results:
        return DataProto()

    if dispatch_mode == DispatchMode.ONE_TO_ALL and deduplicate_by_identity:
        unique: list[DataProto] = []
        seen: set[int] = set()
        for result in results:
            marker = id(result)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(result)
        return DataProto.merge(unique)

    return DataProto.merge(results)
