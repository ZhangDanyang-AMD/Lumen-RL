"""Ray-colocated ATOM rollout server for LumenRL.

This mirrors the native Ray vLLM rollout path, but hosts ATOM's
``AsyncLLMEngine`` in each colocated Ray actor. The public RPC surface matches
``VLLMRayServer`` closely enough for ``VLLMHttpEngine`` to route token-in
generation and for the trainer to reuse the existing ZMQ CUDA-IPC weight sync.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import socket
from multiprocessing import shared_memory
from typing import Any, Optional
from uuid import uuid4

import torch
import zmq

logger = logging.getLogger(__name__)


class ATOMRayServer:
    """Ray actor hosting one ATOM AsyncLLMEngine on a colocated GPU."""

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any],
        replica_rank: int,
        base_seed: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.replica_rank = int(replica_rank)
        self.base_seed = base_seed
        self.engine = None

    async def launch(self) -> bool:
        from atom.rollout.async_engine import AsyncLLMEngine

        kwargs = dict(self.engine_kwargs)
        kwargs.setdefault("model", self.model_name)
        kwargs.setdefault("master_addr", self._get_node_ip())
        kwargs.setdefault("port", self._get_free_port())
        self.engine = AsyncLLMEngine(**kwargs)
        logger.info(
            "ATOMRayServer[%d]: AsyncLLMEngine ready (master=%s:%s online_quant=%s).",
            self.replica_rank,
            kwargs.get("master_addr"),
            kwargs.get("port"),
            kwargs.get("online_quant_config"),
        )
        return True

    @staticmethod
    def _get_node_ip() -> str:
        try:
            import ray
            return ray.util.get_node_ip_address()
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def _get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def ready(self) -> bool:
        return self.engine is not None

    def _build_sampling_params(self, params: dict[str, Any], prompt_length: int):
        from atom.sampling_params import SamplingParams

        params = dict(params)
        max_model_len = int(self.engine_kwargs.get("max_model_len") or 0)
        max_possible = None
        if max_model_len > 0:
            max_possible = max(1, max_model_len - int(prompt_length))

        max_tokens = params.pop("max_tokens", params.pop("max_new_tokens", 128))
        if max_possible is not None:
            max_tokens = min(int(max_tokens), max_possible)

        seed = params.pop("seed", None)
        if seed is None:
            env_seed = os.getenv("ATOM_SAMPLING_SEED")
            seed = int(env_seed) if env_seed not in (None, "") else None
        if seed is None and self.base_seed is not None:
            seed = int(self.base_seed) + self.replica_rank

        # ATOM accepts bool/int for logprobs. LumenRL passes vLLM-style
        # logprobs=0 when token logprobs are requested.
        logprobs = params.pop("logprobs", None)
        if logprobs is not None:
            if isinstance(logprobs, bool):
                logprobs = logprobs
            else:
                logprobs = int(logprobs) >= 0

        if params.pop("do_sample", None) is False:
            params["temperature"] = 0.0
            params["top_k"] = -1
            params["top_p"] = 1.0

        for key in ("repetition_penalty", "stop_token_ids", "min_tokens"):
            if key in params:
                logger.debug("ATOM rollout dropping unsupported sampling param %s=%r", key, params.pop(key))
        if params:
            logger.debug("ATOM rollout dropping unsupported sampling params: %s", sorted(params))

        return SamplingParams(
            max_tokens=max(1, int(max_tokens)),
            temperature=float(params.pop("temperature", 1.0)),
            top_p=float(params.pop("top_p", 1.0)),
            top_k=int(params.pop("top_k", -1)),
            logprobs=logprobs,
            seed=seed,
            ignore_eos=bool(params.pop("ignore_eos", False)),
            stop_strings=params.pop("stop_strings", params.pop("stop", None)),
        )

    async def generate(
        self,
        prompt: list[int],
        sampling_params: dict[str, Any],
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called before generate().")

        prompt_ids = list(prompt)
        sp = self._build_sampling_params(sampling_params, prompt_length=len(prompt_ids))
        rid = request_id or uuid4().hex

        def _generate_blocking():
            return self.engine.generate([prompt_ids], sp, request_ids=[rid])[0]

        out = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        token_ids = list(out.get("token_ids", [])) if isinstance(out, dict) else []
        logprobs = out.get("logprobs") if isinstance(out, dict) else None
        return {
            "text": out.get("text", "") if isinstance(out, dict) else "",
            "prompt_token_ids": prompt_ids,
            "token_ids": token_ids,
            "logprobs": [float(x) for x in logprobs] if logprobs is not None else None,
        }

    async def generate_batch(
        self,
        prompts: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called before generate_batch().")

        prompt_ids_list = [list(p) for p in prompts]
        request_ids = [uuid4().hex for _ in prompt_ids_list]
        params = [
            self._build_sampling_params(sampling_params, prompt_length=len(p))
            for p in prompt_ids_list
        ]

        def _generate_blocking():
            return self.engine.generate(prompt_ids_list, params, request_ids=request_ids)

        outs = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        results: list[dict[str, Any]] = []
        for p_ids, out in zip(prompt_ids_list, outs):
            token_ids = list(out.get("token_ids", [])) if isinstance(out, dict) else []
            logprobs = out.get("logprobs") if isinstance(out, dict) else None
            results.append({
                "text": out.get("text", "") if isinstance(out, dict) else "",
                "prompt_token_ids": p_ids,
                "token_ids": token_ids,
                "logprobs": [float(x) for x in logprobs] if logprobs is not None else None,
            })
        return results

    async def update_weights_from_ipc(self, use_shm: bool = False) -> bool:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called before weight sync.")
        if use_shm:
            self._update_weights_from_shm_sync()
        else:
            self._update_weights_from_ipc_sync()
        return True

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-0.sock"

    @staticmethod
    def _bucket_meta(raw_bucket_meta: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
        bucket_meta: dict[str, dict[str, Any]] = {}
        used_bytes = 0
        for name, meta in raw_bucket_meta.items():
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            offset = int(meta["offset"])
            nbytes = int(dtype.itemsize * torch.Size(shape).numel())
            bucket_meta[name] = {
                "shape": shape,
                "dtype": str(dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
            used_bytes = max(used_bytes, offset + nbytes)
        return bucket_meta, used_bytes

    def _update_weights_from_ipc_sync(self) -> None:
        from torch.multiprocessing.reductions import reduce_tensor

        from atom.rollout.weight_sync import rebuild_ipc_handle

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._get_zmq_handle())

        staging_buffer = None
        staging_handle = None
        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            ipc_buffer = rebuild_ipc_handle(comm_metadata, device_id=0)
            staging_size = int(ipc_buffer.numel())
            stats = {"buckets": 0, "weights": 0}

            while True:
                metadata = socket.recv_pyobj()
                raw_bucket_meta = metadata["bucket_meta"]
                is_last = bool(metadata["is_last"])
                bucket_meta, used_bytes = self._bucket_meta(raw_bucket_meta)

                # Large direct-send tensors carry their own IPC handles. Materialize
                # those into a receiver-owned staging buffer so ATOM's runner sees a
                # single contiguous bucket, just like its native load_weights_via_ipc.
                direct_tensors = {
                    name: rebuild_ipc_handle(meta["handle"], device_id=0)
                    for name, meta in raw_bucket_meta.items()
                    if meta.get("handle") is not None
                }
                ipc_handle = comm_metadata
                ipc_handles = None
                if direct_tensors:
                    if staging_buffer is None or used_bytes > staging_size:
                        staging_size = max(used_bytes, staging_size)
                        staging_buffer = torch.empty(staging_size, dtype=torch.uint8, device="cuda:0")
                        staging_handle = reduce_tensor(staging_buffer)
                    for name, tensor in direct_tensors.items():
                        meta = raw_bucket_meta[name]
                        nbytes = meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                        offset = int(meta["offset"])
                        staging_buffer[offset : offset + nbytes].copy_(
                            tensor.contiguous().view(-1).view(torch.uint8),
                            non_blocking=True,
                        )
                    torch.cuda.synchronize(0)
                    ipc_handle = staging_handle

                self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=ipc_handle,
                    ipc_handles=ipc_handles,
                    bucket_meta=bucket_meta,
                    is_last=is_last,
                )
                stats["buckets"] += 1
                stats["weights"] += len(bucket_meta)
                socket.send(b"")
                if is_last:
                    break
            logger.info("ATOM online weight reload: buckets=%d weights=%d", stats["buckets"], stats["weights"])
        finally:
            socket.close()
            ctx.term()
            del staging_buffer
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _update_weights_from_shm_sync(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._get_zmq_handle())
        shm = None
        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            shm = shared_memory.SharedMemory(name=comm_metadata["name"])
            while True:
                metadata = socket.recv_pyobj()
                bucket_meta, _used_bytes = self._bucket_meta(metadata["bucket_meta"])
                self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_shm",
                    shm_name=shm.name,
                    bucket_meta=bucket_meta,
                    is_last=bool(metadata["is_last"]),
                )
                socket.send(b"")
                if metadata["is_last"]:
                    break
        finally:
            if shm is not None:
                shm.close()
            socket.close()
            ctx.term()

    async def sleep(self, level: int = 2) -> bool:
        if self.engine is not None and hasattr(self.engine, "sleep"):
            self.engine.sleep(level=level)
        return True

    async def wake_up(self, tags: Optional[list[str]] = None) -> bool:
        if self.engine is not None and hasattr(self.engine, "wake_up"):
            self.engine.wake_up(tags=tags or ["weights", "kv_cache"])
        return True

    async def reset_prefix_cache(self) -> bool:
        if self.engine is not None and hasattr(self.engine, "clear_kv_cache"):
            self.engine.clear_kv_cache()
        elif self.engine is not None and hasattr(self.engine, "core_mgr"):
            self.engine.core_mgr.broadcast_utility_command_sync("clear_kv_cache")
        return True

    async def wait_for_requests_to_drain(self, timeout_s: float = 60.0) -> bool:
        return True

    async def shutdown(self) -> bool:
        try:
            if self.engine is not None and hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            elif self.engine is not None and hasattr(self.engine, "close"):
                self.engine.close()
        except Exception:
            pass
        self.engine = None
        return True


class ATOMReplicaManager:
    """Driver-side controller for colocated ATOM rollout actors."""

    def __init__(
        self,
        actor_wg,
        model_name: str,
        engine_kwargs: dict[str, Any],
        *,
        max_concurrency: int = 64,
        base_seed: Optional[int] = None,
    ) -> None:
        self.actor_wg = actor_wg
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.max_concurrency = int(max_concurrency)
        self.base_seed = base_seed
        self.num_replicas = actor_wg.num_workers
        self.servers: list = []

    def create(self) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        job_id = ray.get_runtime_context().get_job_id()
        infos = self.actor_wg.execute_all_sync("get_colocation_info")
        logger.info("ATOMReplicaManager: colocation infos = %s", infos)

        remote_cls = ray.remote(ATOMRayServer)
        for i, info in enumerate(infos):
            node_id = info["node_id"]
            gpu_ids = ",".join(str(g) for g in info["gpu_ids"])
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids,
                "HIP_VISIBLE_DEVICES": gpu_ids,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "LUMEN_REPLICA_RANK": str(i),
                "LUMEN_RAY_JOB_ID": str(job_id),
            }
            for key in (
                "ATOM_ISOLATE_TORCH_COMPILE_CACHE",
                "ATOM_USE_TORCH_RMSNORM",
                "VERL_ATOM_AGENT_LOG",
                "VERL_MEMORY_AGENT_LOG",
            ):
                if key in os.environ:
                    env_vars[key] = os.environ[key]

            server = remote_cls.options(
                num_gpus=0,
                num_cpus=1,
                name=f"lumen-atom-replica-{i}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=self.model_name,
                engine_kwargs=self.engine_kwargs,
                replica_rank=i,
                base_seed=self.base_seed,
            )
            self.servers.append(server)

        ray.get([s.launch.remote() for s in self.servers])
        logger.info("ATOMReplicaManager: launched %d colocated rollout replicas.", len(self.servers))

    def sleep_all(self, level: int = 2) -> None:
        import ray
        ray.get([s.sleep.remote(level) for s in self.servers])

    def wake_all(self, tags: Optional[list[str]] = None) -> None:
        import ray
        ray.get([s.wake_up.remote(tags) for s in self.servers])

    def drain_all(self) -> None:
        import ray
        ray.get([s.wait_for_requests_to_drain.remote() for s in self.servers])

    def shutdown(self) -> None:
        import ray
        try:
            ray.get([s.shutdown.remote() for s in self.servers])
        except Exception:
            pass
        for s in self.servers:
            try:
                ray.kill(s)
            except Exception:
                pass
        self.servers = []
