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

        temperature = float(params.pop("temperature", 1.0))
        top_p = float(params.pop("top_p", 1.0))
        top_k = int(params.pop("top_k", -1))
        ignore_eos = bool(params.pop("ignore_eos", False))
        stop_strings = params.pop("stop_strings", params.pop("stop", None))

        for key in ("repetition_penalty", "stop_token_ids", "min_tokens"):
            if key in params:
                logger.debug("ATOM rollout dropping unsupported sampling param %s=%r", key, params.pop(key))
        if params:
            logger.debug("ATOM rollout dropping unsupported sampling params: %s", sorted(params))

        sp_kwargs = {
            "max_tokens": max(1, int(max_tokens)),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "logprobs": logprobs,
            "seed": seed,
            "ignore_eos": ignore_eos,
            "stop_strings": stop_strings,
        }
        fields = getattr(SamplingParams, "__dataclass_fields__", None)
        if fields:
            sp_kwargs = {k: v for k, v in sp_kwargs.items() if k in fields}
        return SamplingParams(**sp_kwargs)

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
        grouped_prompts: list[list[int]] = []
        grouped_counts: list[int] = []
        for prompt_ids in prompt_ids_list:
            if grouped_prompts and prompt_ids == grouped_prompts[-1]:
                grouped_counts[-1] += 1
            else:
                grouped_prompts.append(prompt_ids)
                grouped_counts.append(1)

        request_ids = [uuid4().hex for _ in grouped_prompts]
        params = []
        for prompt_ids, n in zip(grouped_prompts, grouped_counts):
            sp = self._build_sampling_params(sampling_params, prompt_length=len(prompt_ids))
            if n > 1 and hasattr(sp, "n"):
                sp.n = n
            params.append(sp)

        def _generate_blocking():
            return self.engine.generate(grouped_prompts, params, request_ids=request_ids)

        outs = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        results: list[dict[str, Any]] = []
        expanded_prompts = [p for p, n in zip(grouped_prompts, grouped_counts) for _ in range(n)]
        for p_ids, out in zip(expanded_prompts, outs):
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

        per_gpu_buffers = None
        per_gpu_ipc_handles = None
        ipc_buffer = None
        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            ipc_buffer = rebuild_ipc_handle(comm_metadata, device_id=0)
            bucket_size = int(ipc_buffer.numel())
            num_gpus = int(self.engine_kwargs.get("tensor_parallel_size", 1) or 1) * int(
                self.engine_kwargs.get("data_parallel_size", 1) or 1
            )
            per_gpu_buffers = {
                gpu_idx: torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
                for gpu_idx in range(num_gpus)
            }
            per_gpu_ipc_handles = {gpu_idx: reduce_tensor(buf) for gpu_idx, buf in per_gpu_buffers.items()}
            stats = {"buckets": 0, "weights": 0}

            while True:
                metadata = socket.recv_pyobj()
                raw_bucket_meta = metadata["bucket_meta"]
                is_last = bool(metadata["is_last"])
                bucket_meta, used_bytes = self._bucket_meta(raw_bucket_meta)

                # Large direct-send tensors carry their own IPC handles. Materialize
                # those into a receiver-owned staging buffer so ATOM's runner sees a
                # single stable buffer handle for the whole update cycle, just like
                # verl/ATOM's native load_weights_via_ipc. ModelRunner caches the
                # first IPC mapping until is_last; passing a different handle for
                # direct-send buckets and normal buckets makes later buckets read
                # stale bytes from the first large tensor.
                direct_tensors = {
                    name: rebuild_ipc_handle(meta["handle"], device_id=0)
                    for name, meta in raw_bucket_meta.items()
                    if meta.get("handle") is not None
                }
                if used_bytes > bucket_size:
                    del per_gpu_buffers
                    del per_gpu_ipc_handles
                    bucket_size = used_bytes
                    per_gpu_buffers = {
                        gpu_idx: torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
                        for gpu_idx in range(num_gpus)
                    }
                    per_gpu_ipc_handles = {gpu_idx: reduce_tensor(buf) for gpu_idx, buf in per_gpu_buffers.items()}

                for gpu_idx, dst in per_gpu_buffers.items():
                    for name, tensor in direct_tensors.items():
                        meta = raw_bucket_meta[name]
                        nbytes = meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                        offset = int(meta["offset"])
                        dst[offset : offset + nbytes].copy_(
                            tensor.contiguous().view(-1).view(torch.uint8),
                            non_blocking=True,
                        )
                    if not direct_tensors:
                        dst[:used_bytes].copy_(ipc_buffer[:used_bytes], non_blocking=True)
                    torch.cuda.synchronize(gpu_idx)

                self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=None,
                    ipc_handles=per_gpu_ipc_handles,
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
            del per_gpu_buffers
            del per_gpu_ipc_handles
            del ipc_buffer
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

    async def reload_weights_from_path(self, weight_dir: str) -> bool:
        """Reload weights from a safetensors directory (for multi-GPU TP replicas)."""
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called before reload_weights_from_path().")

        import json

        from safetensors.torch import load_file

        from atom.rollout.weight_sync import load_weights_via_shm

        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            files = sorted(f for f in os.listdir(weight_dir) if f.endswith(".safetensors"))

        def weight_iter():
            for fname in files:
                sd = load_file(os.path.join(weight_dir, fname))
                for name, tensor in sd.items():
                    yield name, tensor

        load_weights_via_shm(self.engine.core_mgr, weight_iter(), bucket_size_mb=2048)
        logger.info("ATOMRayServer[%d]: reloaded weights from %s", self.replica_rank, weight_dir)
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

        atom_tp = int(self.engine_kwargs.get("tensor_parallel_size", 1) or 1)
        num_workers = len(infos)
        if num_workers % atom_tp != 0:
            raise ValueError(
                f"num_workers ({num_workers}) must be divisible by "
                f"atom tensor_parallel_size ({atom_tp})"
            )
        num_replicas = max(1, num_workers // atom_tp)
        self.num_replicas = num_replicas

        true_vocab_size = self._get_true_vocab_size()
        # ATOM's no-eager compilation_config.level>0 rollout needs a live Dynamo, but the
        # FSDP2 training actors must stay on TORCHDYNAMO_DISABLE=1. Scope the opt-in to the
        # rollout actors here instead of letting the launcher export it process-tree wide.
        dynamo_required = self._torch_compile_enabled()
        remote_cls = ray.remote(ATOMRayServer)

        for r in range(num_replicas):
            group = infos[r * atom_tp : (r + 1) * atom_tp]
            node_id = group[0]["node_id"]
            all_gpu_ids: list[str] = []
            for info in group:
                all_gpu_ids.extend(str(g) for g in info["gpu_ids"])
            gpu_ids_str = ",".join(all_gpu_ids)

            nccl_port = 29500 + r
            disable_custom_ar = os.environ.get("LUMENRL_DISABLE_CUSTOM_AR", "1" if atom_tp > 1 else "0")
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids_str,
                "HIP_VISIBLE_DEVICES": gpu_ids_str,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "MASTER_PORT": str(nccl_port),
                "LUMENRL_DISABLE_CUSTOM_AR": disable_custom_ar,
                **({"ATOM_USE_CUSTOM_ALL_GATHER": "0"} if disable_custom_ar in ("1", "true", "True") else {}),
                "LUMEN_REPLICA_RANK": str(r),
                "LUMEN_RAY_JOB_ID": str(job_id),
            }
            if true_vocab_size is not None:
                env_vars["LUMENRL_ATOM_TRUE_VOCAB_SIZE"] = str(true_vocab_size)
            if dynamo_required:
                env_vars["TORCHDYNAMO_DISABLE"] = "0"
            for key in (
                "ATOM_ISOLATE_TORCH_COMPILE_CACHE",
                "ATOM_LOG_LEVEL",
                "ATOM_USE_TORCH_RMSNORM",
                "ATOM_FORCE_ATTN_TRITON",
                "VERL_ATOM_AGENT_LOG",
                "VERL_MEMORY_AGENT_LOG",
            ):
                if key in os.environ:
                    env_vars[key] = os.environ[key]

            engine_kwargs = self._engine_kwargs_for_replica(r, job_id)
            if disable_custom_ar in ("1", "true", "True"):
                engine_kwargs.setdefault(
                    "runner_qualname",
                    "lumenrl.engine.inference.model_runner_nocustomar.NoCustomARModelRunner",
                )

            _dbg = os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True")
            if _dbg:
                logger.info(
                    "[DBG] ATOMReplicaManager: replica %d — gpus=%s disable_ca=%s runner=%s",
                    r, gpu_ids_str, disable_custom_ar, engine_kwargs.get("runner_qualname", "default"),
                )

            server = remote_cls.options(
                num_gpus=0,
                num_cpus=1,
                name=f"lumen-atom-replica-{r}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=self.model_name,
                engine_kwargs=engine_kwargs,
                replica_rank=r,
                base_seed=self.base_seed,
            )
            self.servers.append(server)

        logger.info(
            "ATOMReplicaManager: rollout-scoped TORCHDYNAMO_DISABLE=%s (driver keeps %s)",
            "0" if dynamo_required else "<inherited>",
            os.environ.get("TORCHDYNAMO_DISABLE", "<unset>"),
        )
        for i, s in enumerate(self.servers):
            ray.get(s.launch.remote())
            logger.info("ATOMReplicaManager: replica %d/%d launched.", i + 1, num_replicas)
        logger.info(
            "ATOMReplicaManager: launched %d colocated rollout replicas (atom_tp=%d, workers=%d).",
            num_replicas, atom_tp, num_workers,
        )

    def _torch_compile_enabled(self) -> bool:
        """True when the ATOM engine will run torch.compile (no-eager or level>0)."""
        comp_cfg = self.engine_kwargs.get("compilation_config") or {}
        level = int(comp_cfg.get("level", 0) or 0)
        return level > 0 or not bool(self.engine_kwargs.get("enforce_eager", True))

    def _get_true_vocab_size(self) -> Optional[int]:
        try:
            from transformers import AutoTokenizer

            vocab_size = len(AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True))
            logger.info("ATOMReplicaManager: true tokenizer vocab size = %d", vocab_size)
            return int(vocab_size)
        except Exception as exc:
            logger.warning("ATOMReplicaManager: failed to resolve tokenizer vocab size: %s", exc)
            return None

    def _engine_kwargs_for_replica(self, replica_rank: int, job_id: str) -> dict[str, Any]:
        kwargs = dict(self.engine_kwargs)
        if os.getenv("ATOM_ISOLATE_TORCH_COMPILE_CACHE", "0") not in {"1", "true", "TRUE", "yes", "YES"}:
            return kwargs

        if not self._torch_compile_enabled():
            return kwargs

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        cache_root = os.getenv("ATOM_TORCH_COMPILE_CACHE_ROOT", "/tmp/atom_torch_compile_cache")
        safe_job_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(job_id))
        comp_cfg["cache_dir"] = os.path.join(cache_root, safe_job_id, f"replica_{replica_rank}")
        kwargs["compilation_config"] = comp_cfg
        logger.info(
            "ATOMReplicaManager: replica %d torch compile cache_dir=%s",
            replica_rank,
            comp_cfg["cache_dir"],
        )
        return kwargs

    def sleep_all(self, level: int = 2) -> None:
        import ray
        ray.get([s.sleep.remote(level) for s in self.servers])

    def wake_all(self, tags: Optional[list[str]] = None) -> None:
        import ray
        ray.get([s.wake_up.remote(tags) for s in self.servers])

    def drain_all(self) -> None:
        import ray
        ray.get([s.wait_for_requests_to_drain.remote() for s in self.servers])

    def reload_weights_from_path(self, weight_dir: str) -> None:
        import ray
        ray.get([s.reload_weights_from_path.remote(weight_dir) for s in self.servers])

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
