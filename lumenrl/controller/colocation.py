"""Co-location utilities for placing actor + rollout on the same GPUs."""

from __future__ import annotations

import logging
from typing import Any, Type

import ray

logger = logging.getLogger(__name__)


def create_colocated_worker_cls(
    cls_a: Type,
    cls_b: Type,
    name: str = "ColocatedWorker",
) -> Type:
    """Create a Ray actor class that hosts two worker instances on the same GPU.

    This avoids cross-node weight transfer by co-locating training (actor)
    and inference (rollout) workers on the same GPU set.
    """

    class _ColocatedWorker:
        def __init__(self, rank: int, world_size: int, **kwargs: Any) -> None:
            self.worker_a = cls_a(rank=rank, world_size=world_size, **kwargs)
            self.worker_b = cls_b(rank=rank, world_size=world_size, **kwargs)

        def call_a(self, method: str, *args: Any, **kwargs: Any) -> Any:
            return getattr(self.worker_a, method)(*args, **kwargs)

        def call_b(self, method: str, *args: Any, **kwargs: Any) -> Any:
            return getattr(self.worker_b, method)(*args, **kwargs)

    _ColocatedWorker.__name__ = name
    _ColocatedWorker.__qualname__ = name
    return _ColocatedWorker
