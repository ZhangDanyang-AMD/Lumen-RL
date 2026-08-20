"""ATOM RolloutReplica adapter for verl.

Bridges verl's ``RolloutReplica`` interface to LumenRL's
``ATOMRayServer`` / ``ATOMReplicaManager`` so that verl can use ATOM
as a rollout backend without any verl-side code changes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__name__)


class ATOMRolloutReplica(RolloutReplica):
    """verl ``RolloutReplica`` backed by LumenRL's ``ATOMRayServer``."""

    # ── lifecycle ────────────────────────────────────────────────────────

    async def launch_servers(self):
        """Launch ATOM inference servers colocated with the training workers."""
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from lumenrl.engine.inference.atom_ray_server import ATOMRayServer
        from verl.utils.device import get_resource_name

        # Get node_id and GPU id from each verl worker (same pattern as vLLMReplica)
        worker_infos = await asyncio.gather(
            *[
                w.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for w in self.workers
            ]
        )

        atom_tp = self.config.tensor_model_parallel_size
        num_workers = len(worker_infos)
        num_replicas = max(1, num_workers // atom_tp)

        model_path = self.model_config.path if hasattr(self.model_config, "path") else str(self.model_config)

        engine_kwargs = self._build_engine_kwargs()

        remote_cls = ray.remote(ATOMRayServer)

        for r in range(num_replicas):
            group = worker_infos[r * atom_tp: (r + 1) * atom_tp]
            node_id = group[0][0]
            gpu_ids_str = ",".join(str(info[1]) for info in group)

            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids_str,
                "HIP_VISIBLE_DEVICES": gpu_ids_str,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "MASTER_PORT": str(29500 + r),
                "LUMEN_REPLICA_RANK": str(r),
            }
            for key in (
                "ATOM_ISOLATE_TORCH_COMPILE_CACHE",
                "TORCHDYNAMO_DISABLE",
            ):
                if key in os.environ:
                    env_vars[key] = os.environ[key]

            server = remote_cls.options(
                num_gpus=0,
                num_cpus=1,
                name=f"lumen-atom-verl-replica-{self.replica_rank}-{r}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=model_path,
                engine_kwargs=engine_kwargs,
                replica_rank=r,
                base_seed=self.config.seed if hasattr(self.config, "seed") else None,
            )
            self.servers.append(server)

        for i, s in enumerate(self.servers):
            await s.launch.remote()
            logger.info("ATOMRolloutReplica: server %d/%d launched", i + 1, num_replicas)

        self._server_handle = self.servers[0]
        logger.info(
            "ATOMRolloutReplica: %d servers ready (tp=%d, workers=%d)",
            num_replicas, atom_tp, num_workers,
        )

    # ── weight sync (ATOM manages its own via sleep/wake) ────────────────

    async def wake_up(self):
        await asyncio.gather(*[s.wake_up.remote() for s in self.servers])

    async def sleep(self):
        await asyncio.gather(*[s.sleep.remote() for s in self.servers])

    async def abort_all_requests(self):
        await asyncio.gather(
            *[s.wait_for_requests_to_drain.remote() for s in self.servers]
        )

    async def resume_generation(self):
        pass

    async def clear_kv_cache(self):
        await asyncio.gather(*[s.reset_prefix_cache.remote() for s in self.servers])

    async def release_kv_cache(self):
        await self.sleep()

    async def resume_kv_cache(self):
        await self.wake_up()

    async def start_profile(self, **kwargs):
        pass

    async def stop_profile(self):
        pass

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Convert verl RolloutConfig fields to ATOM engine kwargs."""
        cfg = self.config
        kwargs: dict[str, Any] = {}

        kwargs["tensor_parallel_size"] = cfg.tensor_model_parallel_size
        kwargs["dtype"] = str(cfg.dtype)
        kwargs["enforce_eager"] = getattr(cfg, "enforce_eager", True)
        kwargs["gpu_memory_utilization"] = cfg.gpu_memory_utilization
        kwargs["max_model_len"] = cfg.max_model_len
        kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
        kwargs["max_num_seqs"] = cfg.max_num_seqs
        kwargs["enable_chunked_prefill"] = getattr(cfg, "enable_chunked_prefill", True)

        if cfg.quantization:
            kwargs["quantization"] = cfg.quantization

        if hasattr(cfg, "enable_prefix_caching"):
            kwargs["enable_prefix_caching"] = cfg.enable_prefix_caching

        # Merge any extra engine_kwargs.atom from config
        extra = getattr(cfg, "engine_kwargs", None)
        if extra and hasattr(extra, "get"):
            atom_extra = extra.get("atom", {})
            if hasattr(atom_extra, "items"):
                kwargs.update(dict(atom_extra.items()))

        return kwargs
