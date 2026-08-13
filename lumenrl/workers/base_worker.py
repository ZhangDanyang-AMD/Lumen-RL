"""Abstract base class for Ray-side LumenRL workers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


def get_nested_config(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return ``config[k1][k2]...`` when present, else ``default``."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


class BaseWorker(ABC):
    """Single-process worker contract used by Ray actors.

    The controller stays in one process; each worker actor inherits from this
    class, owns local devices, and exchanges batches via :class:`DataProto`.
    """

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config: dict[str, Any] = dict(config or {})
        self._log = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._configure_logging()
        self._ensure_gpu_isolation()

    def _ensure_gpu_isolation(self) -> None:
        """Narrow this actor to its assigned GPU if Ray did not do it.

        Everything downstream -- ``setup_distributed``'s ``local_rank=0``,
        ``init_model``'s ``model.to(local_device)``, FSDP2's DeviceMesh -- assumes
        Ray has already set CUDA/HIP_VISIBLE_DEVICES so this actor's only device
        is ``cuda:0``. On some images Ray's accelerator detection fails and it
        never writes that variable: every actor then sees all 8 GPUs, lands on
        card 0, and the training step dies with "Multiple ranks detected using the
        same GPU on this node". Ray's *assignment* is still right, so restore the
        assumption instead of teaching each consumer about physical indices.

        ⚠️ Must run before anything initializes CUDA in this process -- torch reads
        the variable at its first CUDA call and caches the result. The actor
        constructor is the earliest hook we own.
        """
        import os

        if any(os.environ.get(v) for v in
               ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
            return
        try:
            import ray

            assigned = ray.get_gpu_ids()
        except Exception:
            return
        if not assigned:
            return
        ids = ",".join(str(g) for g in assigned)
        os.environ["CUDA_VISIBLE_DEVICES"] = ids
        os.environ["HIP_VISIBLE_DEVICES"] = ids
        self._log.warning(
            "Ray left the visible-device variables unset; pinning this actor to "
            "its assigned GPU(s) %s so local_rank 0 is correct", ids,
        )

    def _configure_logging(self) -> None:
        """Ensure worker logs include rank for multi-actor debugging."""
        if not self._log.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                fmt=f"[rank{self.rank}/{self.world_size}] %(name)s: %(levelname)s %(message)s"
            )
            handler.setFormatter(fmt)
            self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)

    @abstractmethod
    def init_model(self) -> None:
        """Allocate models, optimizers, and device state."""

    # ------------------------------------------------------------------
    # Distributed rendezvous (Ray controller path, verl-aligned)
    # ------------------------------------------------------------------
    def get_node_ip(self) -> str:
        """Return this actor's node IP (used to pick the dist master address)."""
        try:
            import ray
            return ray.util.get_node_ip_address()
        except Exception:
            import socket
            return socket.gethostbyname(socket.gethostname())

    def find_free_port(self) -> int:
        """Bind an ephemeral TCP port on this actor's node and return it."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def get_colocation_info(self) -> dict[str, Any]:
        """Return node id + physical GPU ids so a rollout replica can be pinned
        to this actor's GPU (verl HYBRID colocation)."""
        import ray
        ctx = ray.get_runtime_context()
        try:
            gpu_ids = ray.get_gpu_ids()
        except Exception:
            gpu_ids = []
        return {
            "rank": self.rank,
            "node_id": ctx.get_node_id(),
            "gpu_ids": [str(g) for g in gpu_ids],
        }

    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        local_rank: int = 0,
        backend: str = "cpu:gloo,cuda:nccl",
        timeout_s: int = 7200,
    ) -> bool:
        """Join the cross-actor ``torch.distributed`` group for FSDP sharding.

        verl-aligned: each Ray actor is pinned to one GPU (Ray sets
        ``CUDA_VISIBLE_DEVICES`` so its device is ``cuda:0`` => ``local_rank=0``),
        but participates in a single ``world_size``-rank process group so FSDP2
        ``fully_shard`` shards parameters and reduce-scatters gradients across
        all actors. Without this, the FSDP backend falls back to an unsharded
        full replica per actor with no gradient sync (weights diverge).
        """
        import os
        from datetime import timedelta

        import torch

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

        self.rank = int(rank)
        self.world_size = int(world_size)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=int(timeout_s)),
            )
        self._log.info(
            "setup_distributed: rank=%d world_size=%d master=%s:%s local_rank=%d",
            rank, world_size, master_addr, master_port, local_rank,
        )
        return True

    def cleanup(self) -> None:
        """Release GPU memory and tear down runtime hooks."""
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass
        self._log.info("%s.cleanup: default no-op complete.", self.__class__.__name__)
