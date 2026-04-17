"""Ray cluster initialization and resource pool management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import ray

from lumenrl.core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class ResourcePool:
    """A pool of GPU resources for a worker group."""
    name: str
    num_gpus: int
    num_cpus: int = 0
    node_indices: list[int] = field(default_factory=list)
    colocate_with: Optional[str] = None


class RayCluster:
    """Manages Ray cluster lifecycle and resource pools."""

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        self._pools: dict[str, ResourcePool] = {}
        self._initialized = False

    def init(self) -> None:
        """Initialize the Ray cluster."""
        if self._initialized:
            return

        ray_kwargs: dict = {}
        if self.config.ray_address:
            ray_kwargs["address"] = self.config.ray_address
        else:
            ray_kwargs["num_gpus"] = self.config.num_nodes * self.config.gpus_per_node

        ray.init(**ray_kwargs, ignore_reinit_error=True)
        self._initialized = True
        logger.info(
            "Ray cluster initialized: %d nodes x %d GPUs",
            self.config.num_nodes, self.config.gpus_per_node,
        )

    def shutdown(self) -> None:
        if self._initialized:
            ray.shutdown()
            self._initialized = False

    def create_pool(self, name: str, num_gpus: int,
                    colocate_with: Optional[str] = None) -> ResourcePool:
        """Create a named resource pool."""
        pool = ResourcePool(
            name=name,
            num_gpus=num_gpus,
            num_cpus=num_gpus,
            colocate_with=colocate_with,
        )
        self._pools[name] = pool
        logger.info("Created resource pool '%s' with %d GPUs", name, num_gpus)
        return pool

    def get_pool(self, name: str) -> ResourcePool:
        if name not in self._pools:
            raise KeyError(f"Resource pool '{name}' not found")
        return self._pools[name]

    @property
    def total_gpus(self) -> int:
        return self.config.num_nodes * self.config.gpus_per_node
