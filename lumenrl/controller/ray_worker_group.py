"""Ray WorkerGroup: manages a set of Ray actors for a single role."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Type

import ray

from lumenrl.controller.dispatch import DispatchMode, collect_proto, dispatch_proto
from lumenrl.controller.ray_cluster import ResourcePool
from lumenrl.controller.worker_group_factory import resolve_worker_class
from lumenrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


@dataclass
class _SpawnSpec:
    prefix: str
    group: "RayWorkerGroup"


def _prefixed_method(prefix: str, method: str) -> str:
    if not prefix:
        return method
    return f"{prefix}_{method}"


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
        role_key: str | None = None,
        dispatch_mode: DispatchMode | str = DispatchMode.DP_COMPUTE_PROTO,
        detached: bool | None = None,
        worker_names: list[str] | None = None,
        method_prefix: str = "",
    ) -> None:
        if role_key is not None:
            worker_cls = resolve_worker_class(role_key)
        self.worker_cls = worker_cls
        self.pool = pool
        self.num_workers = num_workers if num_workers > 0 else pool.world_size
        self.worker_kwargs = worker_kwargs or {}
        self.dispatch_mode = DispatchMode(dispatch_mode)
        self.detached = pool.detached if detached is None else detached
        self.method_prefix = method_prefix
        self._actors: list[ray.actor.ActorHandle] = []
        self._worker_names: list[str] = worker_names[:] if worker_names is not None else []
        self._lazy_dispatch_state: dict[str, Any] = {}
        self._spawned_groups: dict[str, _SpawnSpec] = {}

    def start(self) -> None:
        """Create and start all workers in this group."""
        if self._actors:
            return

        if self.detached and self._worker_names:
            self._actors = [ray.get_actor(name) for name in self._worker_names]
            return

        if self.pool.num_gpus <= 0:
            gpus_per_worker = 0.0
        else:
            base = self.pool.num_gpus / max(1, self.num_workers)
            gpus_per_worker = min(1.0, max(base, 1.0 / max(1, self.pool.max_colocate_count)))

        RemoteWorker = ray.remote(
            num_gpus=gpus_per_worker,
            num_cpus=1,
        )(self.worker_cls)

        for rank in range(self.num_workers):
            actor_name = f"{self.pool.name}:{self.worker_cls.__name__}:{rank}"
            options_kwargs: dict[str, Any] = {}
            if self.detached:
                options_kwargs["name"] = actor_name
                options_kwargs["lifetime"] = "detached"
            actor_ctor = RemoteWorker.options(**options_kwargs) if options_kwargs else RemoteWorker
            actor = actor_ctor.remote(
                rank=rank,
                world_size=self.num_workers,
                **self.worker_kwargs,
            )
            self._actors.append(actor)
            if self.detached:
                self._worker_names.append(actor_name)

        logger.info(
            "Started %d workers of type %s (%s GPUs each)",
            self.num_workers,
            self.worker_cls.__name__,
            gpus_per_worker,
        )

    def stop(self) -> None:
        """Stop all workers."""
        if self.detached:
            logger.info("Skip killing detached workers for pool '%s'.", self.pool.name)
            self._actors.clear()
            return
        for actor in self._actors:
            ray.kill(actor)
        self._actors.clear()

    def is_alive(self) -> bool:
        """Best-effort liveness check for all workers."""
        if not self._actors:
            return False
        refs = [actor.__ray_ready__.remote() for actor in self._actors]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=2.0)
        return len(ready) == len(refs)

    def execute_all_async(self, method: str, *args: Any, **kwargs: Any) -> list[ray.ObjectRef]:
        target_method = _prefixed_method(self.method_prefix, method)
        return [getattr(a, target_method).remote(*args, **kwargs) for a in self._actors]

    def execute_all_sync(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        refs = self.execute_all_async(method, *args, **kwargs)
        return ray.get(refs)

    def execute_rank_zero_sync(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.call_single(0, method, *args, **kwargs)

    def call_all(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on all workers and collect results."""
        return self.execute_all_sync(method, *args, **kwargs)

    def call_single(self, worker_idx: int, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on a single worker."""
        target_method = _prefixed_method(self.method_prefix, method)
        ref = getattr(self._actors[worker_idx], target_method).remote(*args, **kwargs)
        return ray.get(ref)

    def call_single_async(self, worker_idx: int, method: str, *args: Any, **kwargs: Any) -> ray.ObjectRef:
        target_method = _prefixed_method(self.method_prefix, method)
        return getattr(self._actors[worker_idx], target_method).remote(*args, **kwargs)

    def call_with_timeout(
        self,
        worker_idx: int,
        method: str,
        timeout_s: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ref = self.call_single_async(worker_idx, method, *args, **kwargs)
        ready, _ = ray.wait([ref], timeout=timeout_s)
        if not ready:
            raise TimeoutError(f"{method} on worker {worker_idx} timed out in {timeout_s}s")
        return ray.get(ref)

    def dispatch_and_call(
        self,
        method: str,
        data: DataProto,
        mode: DispatchMode | str | None = None,
        mesh_mapping: list[int] | None = None,
        lazy_key: str | None = None,
        **kwargs: Any,
    ) -> DataProto:
        """Split data across workers, call method, and merge results."""
        dispatch_mode = self.dispatch_mode if mode is None else DispatchMode(mode)
        chunks = dispatch_proto(
            data,
            self.num_workers,
            mode=dispatch_mode,
            mesh_mapping=mesh_mapping,
            lazy_state=self._lazy_dispatch_state,
            lazy_key=lazy_key,
        )
        target_method = _prefixed_method(self.method_prefix, method)
        if not chunks:
            return DataProto()

        if len(chunks) == 1:
            refs = [getattr(self._actors[0], target_method).remote(chunks[0], **kwargs)]
        elif len(chunks) == self.num_workers:
            refs = [
                getattr(self._actors[i], target_method).remote(chunks[i], **kwargs)
                for i in range(self.num_workers)
            ]
        else:
            raise ValueError(
                f"dispatch produced {len(chunks)} chunks for {self.num_workers} workers; "
                "expected 1 (rank-zero) or num_workers."
            )
        results = ray.get(refs)
        return collect_proto(results, mode=dispatch_mode)

    def spawn(self, prefixes: list[str]) -> dict[str, "RayWorkerGroup"]:
        """Create prefixed logical views over this worker group.

        Spawned groups share the same actor handles and differ only by method prefix.
        """
        spawned: dict[str, RayWorkerGroup] = {}
        for prefix in prefixes:
            if prefix in self._spawned_groups:
                spawned[prefix] = self._spawned_groups[prefix].group
                continue
            child = RayWorkerGroup(
                worker_cls=self.worker_cls,
                pool=self.pool,
                num_workers=self.num_workers,
                worker_kwargs=self.worker_kwargs,
                dispatch_mode=self.dispatch_mode,
                detached=self.detached,
                worker_names=self._worker_names,
                method_prefix=prefix,
            )
            child._actors = self._actors
            self._spawned_groups[prefix] = _SpawnSpec(prefix=prefix, group=child)
            spawned[prefix] = child
        return spawned

    def fuse(self, prefix: str) -> "RayWorkerGroup":
        """Return a single prefixed logical group."""
        return self.spawn([prefix])[prefix]

    @classmethod
    def from_detached(
        cls,
        worker_cls: Type,
        pool: ResourcePool,
        worker_names: list[str],
        *,
        worker_kwargs: dict | None = None,
        dispatch_mode: DispatchMode | str = DispatchMode.DP_COMPUTE_PROTO,
    ) -> "RayWorkerGroup":
        group = cls(
            worker_cls=worker_cls,
            pool=pool,
            num_workers=len(worker_names),
            worker_kwargs=worker_kwargs,
            dispatch_mode=dispatch_mode,
            detached=True,
            worker_names=worker_names,
        )
        group.start()
        return group

    @property
    def actors(self) -> list[ray.actor.ActorHandle]:
        return self._actors

    @property
    def worker_names(self) -> list[str]:
        return self._worker_names[:]
