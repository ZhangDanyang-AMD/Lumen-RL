"""Ray WorkerGroup: manages a set of Ray actors for a single role."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

import ray

from lumenrl.controller.ray_cluster import ResourcePool
from lumenrl.core.protocol import DataProto
from lumenrl.controller.dispatch import dispatch_proto, collect_proto

logger = logging.getLogger(__name__)


class RayWorkerGroup:
    """A group of Ray actor workers for a single RL role.

    Handles creation, dispatch, collection, and lifecycle of workers.
    """

    def __init__(
        self,
        worker_cls: Type,
        pool: ResourcePool,
        num_workers: int,
        worker_kwargs: dict | None = None,
    ) -> None:
        self.worker_cls = worker_cls
        self.pool = pool
        self.num_workers = num_workers
        self.worker_kwargs = worker_kwargs or {}
        self._actors: list[ray.actor.ActorHandle] = []

    def start(self) -> None:
        """Create and start all workers in this group."""
        if self.pool.num_gpus <= 0:
            gpus_per_worker = 0
        else:
            gpus_per_worker = max(1, self.pool.num_gpus // self.num_workers)

        RemoteWorker = ray.remote(
            num_gpus=gpus_per_worker,
            num_cpus=1,
        )(self.worker_cls)

        for rank in range(self.num_workers):
            actor = RemoteWorker.remote(
                rank=rank,
                world_size=self.num_workers,
                **self.worker_kwargs,
            )
            self._actors.append(actor)

        logger.info(
            "Started %d workers of type %s (%d GPUs each)",
            self.num_workers, self.worker_cls.__name__, gpus_per_worker,
        )

    def stop(self) -> None:
        """Stop all workers."""
        for actor in self._actors:
            ray.kill(actor)
        self._actors.clear()

    def call_all(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on all workers and collect results."""
        refs = [getattr(a, method).remote(*args, **kwargs) for a in self._actors]
        return ray.get(refs)

    def call_single(self, worker_idx: int, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on a single worker."""
        ref = getattr(self._actors[worker_idx], method).remote(*args, **kwargs)
        return ray.get(ref)

    def dispatch_and_call(self, method: str, data: DataProto, **kwargs: Any) -> DataProto:
        """Split data across workers, call method, and merge results."""
        chunks = dispatch_proto(data, self.num_workers)
        refs = [
            getattr(self._actors[i], method).remote(chunks[i], **kwargs)
            for i in range(self.num_workers)
        ]
        results = ray.get(refs)
        return collect_proto(results)

    @property
    def actors(self) -> list[ray.actor.ActorHandle]:
        return self._actors
