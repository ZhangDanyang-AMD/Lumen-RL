"""Ray cluster initialization and resource pool management."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ray

from lumenrl.core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class ResourcePool:
    """A pool of GPU resources for a worker group.

    ``process_on_nodes`` models complex topology layouts (e.g. [4,4,2]).
    If omitted, we default to a single-node layout of ``num_gpus`` workers.
    """

    name: str
    num_gpus: int
    num_cpus: int = 0
    node_indices: list[int] = field(default_factory=list)
    colocate_with: Optional[str] = None
    process_on_nodes: list[int] = field(default_factory=list)
    max_colocate_count: int = 1
    detached: bool = False
    topology_tags: dict[str, str] = field(default_factory=dict)
    use_placement_groups: bool = False

    def __post_init__(self) -> None:
        if not self.process_on_nodes:
            self.process_on_nodes = [max(0, int(self.num_gpus))]

    @property
    def world_size(self) -> int:
        return sum(self.process_on_nodes)

    def split(
        self, split_size: int | list[int], suffix: str | None = None
    ) -> list["ResourcePool"]:
        """Split current pool into sub-pools by worker counts."""
        if isinstance(split_size, int):
            if split_size <= 0 or self.world_size % split_size != 0:
                raise ValueError(
                    "split_size must be a positive divisor of pool world_size."
                )
            split_list = [split_size] * (self.world_size // split_size)
        else:
            split_list = list(split_size)
            if any(v <= 0 for v in split_list) or sum(split_list) != self.world_size:
                raise ValueError(
                    "split_size list must be positive and sum to pool world_size."
                )

        out: list[ResourcePool] = []
        for idx, sz in enumerate(split_list):
            name = (
                f"{self.name}_split_{idx}"
                if suffix is None
                else f"{self.name}_{suffix}_{idx}"
            )
            out.append(
                ResourcePool(
                    name=name,
                    num_gpus=sz,
                    num_cpus=sz,
                    node_indices=self.node_indices,
                    colocate_with=self.colocate_with,
                    process_on_nodes=[sz],
                    max_colocate_count=self.max_colocate_count,
                    detached=self.detached,
                    topology_tags=dict(self.topology_tags),
                )
            )
        return out

    def merge_with(
        self, other: "ResourcePool", name: str | None = None
    ) -> "ResourcePool":
        """Merge two compatible pools into one."""
        if self.max_colocate_count != other.max_colocate_count:
            raise ValueError("Cannot merge pools with different max_colocate_count.")
        merged_name = name or f"{self.name}_{other.name}"
        return ResourcePool(
            name=merged_name,
            num_gpus=self.num_gpus + other.num_gpus,
            num_cpus=self.num_cpus + other.num_cpus,
            node_indices=self.node_indices + other.node_indices,
            colocate_with=self.colocate_with or other.colocate_with,
            process_on_nodes=self.process_on_nodes + other.process_on_nodes,
            max_colocate_count=self.max_colocate_count,
            detached=self.detached and other.detached,
            topology_tags={**self.topology_tags, **other.topology_tags},
        )


class RayCluster:
    """Manages Ray cluster lifecycle and resource pools."""

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        self._pools: dict[str, ResourcePool] = {}
        self._initialized = False
        self._borrowed_aligned_job = False

    def init(self) -> None:
        """Initialize the Ray cluster."""
        aligned_env = None
        if os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") == "1":
            required = (
                "LUMENRL_ATOM_OP_ALIGNMENT",
                "LUMENRL_ATOM_ATTENTION_CONFIG",
                "LUMENRL_ATOM_ATTENTION_CONFIG_SHA256",
                "PYTHONPATH",
            )
            if any(not os.environ.get(key) for key in required):
                raise RuntimeError(
                    "Aligned Ray job requires shared Attention config/SHA and explicit PYTHONPATH"
                )
            aligned_env = {key: os.environ[key] for key in required}
            config_path = aligned_env["LUMENRL_ATOM_ATTENTION_CONFIG"]
            expected_sha = aligned_env["LUMENRL_ATOM_ATTENTION_CONFIG_SHA256"]
            if (
                hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
                != expected_sha.lower()
            ):
                raise RuntimeError("Aligned Ray job Attention config SHA256 mismatch")
            for key in (
                "LUMENRL_ATOM_TRUE_VOCAB_SIZE",
                "LUMEN_ATOM_BF16_POLICY",
                "LUMEN_MXFP4_ASM",
                "LUMEN_MXFP4_PRESHUFFLE",
                "LUMENRL_SMCI_GPU_IDENTITY_MANIFEST",
            ):
                if key in os.environ:
                    aligned_env[key] = os.environ[key]

            def verify_current_job() -> None:
                # Do not infer an existing job's environment from the driver.
                current = getattr(ray.get_runtime_context(), "runtime_env", None)
                current_vars = (
                    current.get("env_vars") if isinstance(current, dict) else None
                )
                if not isinstance(current_vars, dict) or any(
                    current_vars.get(key) != value for key, value in aligned_env.items()
                ):
                    raise RuntimeError(
                        "Existing Ray job lacks or conflicts with the aligned runtime environment"
                    )

            if ray.is_initialized():
                if os.environ.get("LUMENRL_RAY_PLASMA_DIRECTORY"):
                    raise RuntimeError(
                        "Cannot attest private Ray plasma for an already initialized job"
                    )
                verify_current_job()
                if not self._initialized:
                    self._borrowed_aligned_job = True
                self._initialized = True
                return
            if self._initialized:
                raise RuntimeError("Aligned Ray job is no longer initialized")
        if self._initialized:
            return

        ray_kwargs: dict = {}
        if self.config.ray_address:
            ray_kwargs["address"] = self.config.ray_address
        else:
            ray_kwargs["num_gpus"] = self.config.num_nodes * self.config.gpus_per_node

        if aligned_env is not None:
            ray_kwargs["runtime_env"] = {"env_vars": aligned_env}
        # Default-off private Ray storage.  On a shared node with a nearly
        # full /dev/shm, TMPDIR/RAY_TMPDIR alone do not relocate plasma.
        # Bound its allocation and pin the mmap directory before any actor
        # starts; the run's storage preflight checks the capacity separately.
        plasma_dir = os.environ.get("LUMENRL_RAY_PLASMA_DIRECTORY")
        if plasma_dir:
            if self.config.ray_address:
                raise RuntimeError("Private plasma cannot be applied to a remote Ray cluster")
            if not os.path.isdir(plasma_dir) or not os.access(plasma_dir, os.W_OK):
                raise RuntimeError(f"Ray plasma directory unavailable: {plasma_dir}")
            mib = os.environ.get("LUMENRL_RAY_OBJECT_STORE_MIB")
            if not mib or not mib.isdecimal() or int(mib) < 1024:
                raise RuntimeError("private Ray plasma needs a >=1024 MiB object-store budget")
            ray_kwargs["_plasma_directory"] = plasma_dir
            ray_kwargs["object_store_memory"] = int(mib) * 1024 * 1024
            temp_dir = os.environ.get("LUMENRL_RAY_PRIVATE_TEMP_DIR")
            if not temp_dir or not os.path.isdir(temp_dir) or not os.access(temp_dir, os.W_OK):
                raise RuntimeError("private Ray plasma needs a writable private temp directory")
            ray_kwargs["_temp_dir"] = temp_dir
            logger.info("Private Ray plasma: %s (%s MiB), temp=%s", plasma_dir, mib, temp_dir)
        ray.init(**ray_kwargs, ignore_reinit_error=True)
        if aligned_env is not None:
            verify_current_job()
        self._initialized = True
        logger.info(
            "Ray cluster initialized: %d nodes x %d GPUs",
            self.config.num_nodes,
            self.config.gpus_per_node,
        )

    def shutdown(self) -> None:
        if self._initialized:
            # An alignment wrapper may borrow an already configured job. Only
            # its owner may disconnect that job and its non-detached actors.
            if not self._borrowed_aligned_job:
                ray.shutdown()
            self._initialized = False
            self._borrowed_aligned_job = False

    def create_pool(
        self,
        name: str,
        num_gpus: int,
        colocate_with: Optional[str] = None,
        process_on_nodes: Optional[list[int]] = None,
        max_colocate_count: int = 1,
        detached: bool = False,
        topology_tags: Optional[dict[str, str]] = None,
    ) -> ResourcePool:
        """Create a named resource pool."""
        explicit_pg = process_on_nodes is not None
        process_layout = process_on_nodes[:] if explicit_pg else [num_gpus]
        pool = ResourcePool(
            name=name,
            num_gpus=num_gpus,
            num_cpus=num_gpus,
            colocate_with=colocate_with,
            process_on_nodes=process_layout,
            max_colocate_count=max_colocate_count,
            detached=detached,
            topology_tags=dict(topology_tags or {}),
            use_placement_groups=explicit_pg,
        )
        self._pools[name] = pool
        logger.info(
            "Created resource pool '%s' with %d GPUs (layout=%s, colocate=%s)",
            name,
            num_gpus,
            process_layout,
            colocate_with,
        )
        return pool

    def get_pool(self, name: str) -> ResourcePool:
        if name not in self._pools:
            raise KeyError(f"Resource pool '{name}' not found")
        return self._pools[name]

    @property
    def total_gpus(self) -> int:
        return self.config.num_nodes * self.config.gpus_per_node

    def map_topology(self, mapping: dict[str, str]) -> dict[str, ResourcePool]:
        """Map role names to existing pools for complex topology plans."""
        resolved: dict[str, ResourcePool] = {}
        for role, pool_name in mapping.items():
            resolved[role] = self.get_pool(pool_name)
        return resolved
