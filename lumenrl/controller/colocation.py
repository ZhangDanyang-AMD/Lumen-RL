"""Co-location utilities for placing actor + rollout on the same GPUs."""

from __future__ import annotations

import logging
from typing import Any, Type

logger = logging.getLogger(__name__)


def _public_method_names(worker_cls: Type) -> list[str]:
    """Instance methods Ray must see as real attributes, not ``__getattr__``."""
    names: list[str] = []
    seen: set[str] = set()
    for base in worker_cls.__mro__:
        if base is object:
            continue
        for name, member in base.__dict__.items():
            if name.startswith("_") or name in seen:
                continue
            if isinstance(member, (staticmethod, classmethod, property)):
                continue
            if not callable(member):
                continue
            seen.add(name)
            names.append(name)
    return names


def _make_prefixed_method(prefix: str, method_name: str):
    def _call(self, *args, **kwargs):
        return getattr(self._workers[prefix], method_name)(*args, **kwargs)

    _call.__name__ = f"{prefix}_{method_name}"
    return _call


def create_fused_worker_cls(
    worker_map: dict[str, Type],
    name: str = "FusedWorker",
) -> Type:
    """Create a worker that fuses multiple worker roles into one actor.

    Exposed methods are prefixed as ``{prefix}_{method}``, e.g.
    ``actor_train_step`` / ``rollout_generate``. They are real functions on the
    class. Ray only exports those; a method that exists solely through
    ``__getattr__`` is missing from the actor handle.
    """
    if not worker_map:
        raise ValueError("worker_map must not be empty for fused workers.")

    class _FusedWorker:
        def __init__(self, rank: int, world_size: int, **kwargs: Any) -> None:
            self._workers = {
                prefix: worker_cls(rank=rank, world_size=world_size, **kwargs)
                for prefix, worker_cls in worker_map.items()
            }

        def health_check(self) -> dict[str, bool]:
            return {prefix: True for prefix in self._workers}

    for prefix, worker_cls in worker_map.items():
        for method_name in _public_method_names(worker_cls):
            exported = f"{prefix}_{method_name}"
            if exported in _FusedWorker.__dict__:
                continue
            method = _make_prefixed_method(prefix, method_name)
            method.__qualname__ = f"{name}.{exported}"
            setattr(_FusedWorker, exported, method)

    _FusedWorker.__name__ = name
    _FusedWorker.__qualname__ = name
    return _FusedWorker


def create_colocated_worker_cls(
    cls_a: Type,
    cls_b: Type,
    name: str = "ColocatedWorker",
) -> Type:
    """Create a Ray actor class that hosts two worker instances on the same GPU.

    This avoids cross-node weight transfer by co-locating training (actor)
    and inference (rollout) workers on the same GPU set.
    """

    return create_fused_worker_cls({"a": cls_a, "b": cls_b}, name=name)
