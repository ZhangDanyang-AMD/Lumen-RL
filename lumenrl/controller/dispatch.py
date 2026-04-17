"""Dispatch and collect DataProto across workers."""

from __future__ import annotations

from lumenrl.core.protocol import DataProto


def dispatch_proto(data: DataProto, num_workers: int) -> list[DataProto]:
    """Split a DataProto into chunks for DP workers."""
    return data.split(num_workers)


def collect_proto(results: list[DataProto]) -> DataProto:
    """Merge DataProto results from DP workers."""
    return DataProto.merge(results)
