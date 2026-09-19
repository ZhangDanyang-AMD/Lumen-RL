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
import time
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
        self._pin_cudagraph_mode(kwargs)
        self._pin_sleep_keeps_memory_resident(kwargs)
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
    def _is_no_eager(kwargs: dict[str, Any]) -> bool:
        """Will this rollout run torch.compile and capture CUDA graphs?

        Both pins below apply exactly here and nowhere else, so they ask once:
        graphs that are captured but released on sleep, or memory kept resident
        with no graphs to protect, is neither of the two configurations the
        reference values were measured in.
        """
        comp_cfg = kwargs.get("compilation_config") or {}
        level = int(comp_cfg.get("level", 0) or 0)
        return level > 0 or not bool(kwargs.get("enforce_eager", True))

    def _pin_cudagraph_mode(self, kwargs: dict[str, Any]) -> None:
        """Choose ATOM's CUDA-graph strategy for a no-eager rollout.

        Only applies once torch.compile is on (``enforce_eager=false`` or
        ``compilation_config.level>0``). ATOM leaves ``cudagraph_mode`` unset and
        then defaults it to PIECEWISE, which gives every compiled dense piece its
        own graph and asserts that the piece's inputs keep their capture-time
        addresses. The attention between two pieces runs eager and allocates its
        output afresh each call, so the very first replay aborts all rollout
        workers with "Input addresses for cudagraphs are different during
        replay". FULL captures the whole forward instead and has no such
        boundary.

        Override with ``atom_cfg.engine_kwargs.compilation_config.cudagraph_mode``.
        ATOM builds that pin the mode themselves still win — this only supplies a
        value.
        """
        if not self._is_no_eager(kwargs):
            return

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        level = int(comp_cfg.get("level", 0) or 0)
        mode = comp_cfg.get("cudagraph_mode") or "FULL"
        if isinstance(mode, str):
            from atom.config import CUDAGraphMode

            try:
                mode = CUDAGraphMode[mode.upper()]
            except KeyError as exc:
                supported = ", ".join(m.name for m in CUDAGraphMode)
                raise ValueError(
                    f"unknown ATOM cudagraph_mode {mode!r}; supported: {supported}"
                ) from exc

        comp_cfg["cudagraph_mode"] = mode
        kwargs["compilation_config"] = comp_cfg
        logger.info(
            "ATOMRayServer[%d]: no-eager rollout with compilation level=%d, "
            "cudagraph_mode=%s",
            self.replica_rank,
            level,
            getattr(mode, "name", mode),
        )

    def _pin_sleep_keeps_memory_resident(self, kwargs: dict[str, Any]) -> None:
        """Keep a no-eager rollout's weights and KV pool allocated across sleep.

        This is the behaviour the release measurements were taken against: ATOM
        up to `28721a50` kept both resident in no-eager mode unconditionally,
        because a decode graph captures the base address of the KV pool and
        recapturing on wake faults. `ROCm/ATOM#2028` turned that into
        `Config.sleep_keeps_memory_resident`, defaulting to release, so leaving it
        unset silently changes what a colocated ATOM rollout does at every step.

        Releasing is not merely slower here, it does not work: ATOM re-derives the
        KV block count on each wake as `gpu_memory_utilization x total` minus
        everything resident on the card, and after the first optimizer step the
        colocated trainer is 52 GB of that. Qwen3-30B-A3B (example 9) then asks
        for a negative pool and all eight replicas assert in `resume_memory`. The
        8B examples have the headroom to survive it, and merely pay the recapture.

        No-op on the older pin: ATOM filters engine kwargs against its `Config`
        fields, and a build without this one drops the kwarg. It also has no
        effect under `enforce_eager`, where there are no graphs to keep valid, so
        this only supplies a value where `_pin_cudagraph_mode` supplies one too.

        Override with `atom_cfg.engine_kwargs.sleep_keeps_memory_resident`.
        """
        if not self._is_no_eager(kwargs):
            return

        kwargs.setdefault("sleep_keeps_memory_resident", True)
        logger.info(
            "ATOMRayServer[%d]: sleep_keeps_memory_resident=%s",
            self.replica_rank,
            kwargs["sleep_keeps_memory_resident"],
        )

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
                logger.debug(
                    "ATOM rollout dropping unsupported sampling param %s=%r",
                    key,
                    params.pop(key),
                )
        if params:
            logger.debug(
                "ATOM rollout dropping unsupported sampling params: %s", sorted(params)
            )

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
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before generate()."
            )

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
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before generate_batch()."
            )

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
            sp = self._build_sampling_params(
                sampling_params, prompt_length=len(prompt_ids)
            )
            if n > 1 and hasattr(sp, "n"):
                sp.n = n
            params.append(sp)

        def _generate_blocking():
            return self.engine.generate(
                grouped_prompts, params, request_ids=request_ids
            )

        outs = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        results: list[dict[str, Any]] = []
        expanded_prompts = [
            p for p, n in zip(grouped_prompts, grouped_counts) for _ in range(n)
        ]
        for p_ids, out in zip(expanded_prompts, outs):
            token_ids = list(out.get("token_ids", [])) if isinstance(out, dict) else []
            logprobs = out.get("logprobs") if isinstance(out, dict) else None
            results.append(
                {
                    "text": out.get("text", "") if isinstance(out, dict) else "",
                    "prompt_token_ids": p_ids,
                    "token_ids": token_ids,
                    "logprobs": (
                        [float(x) for x in logprobs] if logprobs is not None else None
                    ),
                }
            )
        return results

    async def update_weights_from_ipc(
        self, use_shm: bool = False, version: int | None = None
    ) -> bool:
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before weight sync."
            )
        if use_shm:
            self._update_weights_from_shm_sync(version)
        else:
            self._update_weights_from_ipc_sync(version)
        return True

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return (
            f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-0.sock"
        )

    def _counts_are_exact(self) -> bool:
        """Whether ATOM's per-bucket ``updated`` can be compared for equality.

        Only on a BF16 rollout. With online quantization on, a fused parameter's
        shards accumulate in a staging buffer and are requantized when the last
        one arrives, so ATOM counts one update for the group and nothing for the
        shards ahead of it -- a bucket that ends mid-group reports fewer updates
        than it holds, with nothing wrong. See assert_bucket_fully_applied.
        """
        return not (self.engine_kwargs.get("online_quant_config") or {})

    @staticmethod
    def _bucket_meta(
        raw_bucket_meta: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], int]:
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

    def _update_weights_from_ipc_sync(self, version: int | None = None) -> None:
        from torch.multiprocessing.reductions import reduce_tensor

        from atom.rollout.weight_sync import rebuild_ipc_handle

        from lumenrl.engine.inference.atom_moe_weight_sync import (
            assert_bucket_fully_applied,
            atom_routes_fused_experts,
            fused_expert_renames,
            relayout_fused_experts,
            rename_bucket_meta,
            require_unsharded_experts,
        )
        from lumenrl.engine.inference.bucketed_weight_transfer import (
            check_bucket_version,
        )

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
            num_gpus = int(
                self.engine_kwargs.get("tensor_parallel_size", 1) or 1
            ) * int(self.engine_kwargs.get("data_parallel_size", 1) or 1)
            per_gpu_buffers = {
                gpu_idx: torch.empty(
                    bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}"
                )
                for gpu_idx in range(num_gpus)
            }
            per_gpu_ipc_handles = {
                gpu_idx: reduce_tensor(buf) for gpu_idx, buf in per_gpu_buffers.items()
            }
            stats = {"buckets": 0, "weights": 0, "experts": 0}

            while True:
                metadata = socket.recv_pyobj()
                check_bucket_version(metadata, version)
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
                    # Only the first bucket may grow the staging buffers: the
                    # runner has not mapped anything yet, so it picks up the new
                    # handles. Later on it is still holding the first mapping and
                    # would read a short, stale view of a fresh allocation.
                    if stats["buckets"] > 0:
                        raise RuntimeError(
                            f"bucket needs {used_bytes} B but the staging buffer is "
                            f"{bucket_size} B and ATOM's runner has already mapped it; "
                            "the sender must size its bucket to the largest tensor "
                            "(see LumenActorWorker.update_weights_ipc_send)"
                        )
                    del per_gpu_buffers
                    del per_gpu_ipc_handles
                    bucket_size = used_bytes
                    per_gpu_buffers = {
                        gpu_idx: torch.empty(
                            bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}"
                        )
                        for gpu_idx in range(num_gpus)
                    }
                    per_gpu_ipc_handles = {
                        gpu_idx: reduce_tensor(buf)
                        for gpu_idx, buf in per_gpu_buffers.items()
                    }

                # transformers-5.x ships MoE experts as fused tensors under names
                # an older ATOM's updater cannot resolve, and its unquantized MoE
                # path keeps those buffers in an aiter-shuffled layout that
                # nothing re-establishes after an update. Both are handled in the
                # staging buffer, before the runner reads it -- unless the ATOM
                # in this process does it itself, in which case the trainer's
                # names go through untouched.
                renames = (
                    {}
                    if atom_routes_fused_experts()
                    else fused_expert_renames(bucket_meta)
                )
                if renames:
                    require_unsharded_experts(
                        self.engine_kwargs.get("tensor_parallel_size", 1),
                        self.engine_kwargs.get("enable_expert_parallel", False),
                    )

                for gpu_idx, dst in per_gpu_buffers.items():
                    for name, tensor in direct_tensors.items():
                        meta = raw_bucket_meta[name]
                        nbytes = (
                            meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                        )
                        offset = int(meta["offset"])
                        dst[offset : offset + nbytes].copy_(
                            tensor.contiguous().view(-1).view(torch.uint8),
                            non_blocking=True,
                        )
                    if not direct_tensors:
                        dst[:used_bytes].copy_(
                            ipc_buffer[:used_bytes], non_blocking=True
                        )
                    torch.cuda.synchronize(gpu_idx)
                    # After the copy: this rewrites the staged bytes, so it must
                    # not race the fill, and each per-GPU buffer needs its own
                    # pass because the runners read them independently. The
                    # device context is for the aiter kernels behind the shuffle,
                    # which launch on the current device, not the tensor's.
                    if renames:
                        with torch.cuda.device(gpu_idx):
                            relayout_fused_experts(dst, bucket_meta, renames)
                            torch.cuda.synchronize(gpu_idx)

                responses = self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=None,
                    ipc_handles=per_gpu_ipc_handles,
                    bucket_meta=rename_bucket_meta(bucket_meta, renames),
                    is_last=is_last,
                )
                assert_bucket_fully_applied(
                    responses,
                    bucket_meta,
                    context="ipc",
                    exact=self._counts_are_exact(),
                )
                stats["buckets"] += 1
                stats["weights"] += len(bucket_meta)
                stats["experts"] += len(renames)
                socket.send(b"")
                if is_last:
                    break
            logger.info(
                "ATOM online weight reload: buckets=%d weights=%d fused_experts=%d",
                stats["buckets"],
                stats["weights"],
                stats["experts"],
            )
        finally:
            socket.close()
            ctx.term()
            del per_gpu_buffers
            del per_gpu_ipc_handles
            del ipc_buffer
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _update_weights_from_shm_sync(self, version: int | None = None) -> None:
        from lumenrl.engine.inference.atom_moe_weight_sync import (
            assert_bucket_fully_applied,
            atom_routes_fused_experts,
            fused_expert_renames,
        )
        from lumenrl.engine.inference.bucketed_weight_transfer import (
            check_bucket_version,
        )

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
                check_bucket_version(metadata, version)
                bucket_meta, _used_bytes = self._bucket_meta(metadata["bucket_meta"])
                if (
                    fused_expert_renames(bucket_meta)
                    and not atom_routes_fused_experts()
                ):
                    # An older ATOM needs the fused names rewritten and its
                    # shuffled layout re-established, which the IPC path does in
                    # a staging buffer it owns. Here the segment belongs to the
                    # sender and lives in host memory, where aiter's shuffle does
                    # not run, so there is nowhere to do the same work. Refuse
                    # rather than repeat the silent-skip bug. An ATOM that routes
                    # the fused names itself does the layout on the device, after
                    # the copy, so this transport is fine there.
                    raise RuntimeError(
                        "ATOM rollout of a fused-expert MoE model needs the CUDA-IPC "
                        "weight transport; set use_shm=false. See "
                        "lumenrl/engine/inference/atom_moe_weight_sync.py."
                    )
                responses = self.engine.core_mgr.broadcast_utility_command_sync(
                    "update_weights_shm",
                    shm_name=shm.name,
                    bucket_meta=bucket_meta,
                    is_last=bool(metadata["is_last"]),
                )
                assert_bucket_fully_applied(
                    responses,
                    bucket_meta,
                    context="shm",
                    exact=self._counts_are_exact(),
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

    async def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 600.0,
    ) -> list[Any]:
        """Invoke *method* on every ATOM model runner, DP-major then TP rank.

        Mirrors ``VLLMRayServer.collective_rpc`` so the same caller drives
        either backend. Returns one entry per rank; a rank that failed raises
        here rather than returning a half-answer, because every current caller
        treats the call as a barrier and would otherwise proceed on partial
        state.

        The executor hop is required: the engine call blocks on a queue, and
        running it on the actor's event loop would stall ``generate`` for the
        whole weight sync. ``_generate_blocking`` above uses the same pattern.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")

        def _blocking():
            return self.engine.collective_rpc(
                method,
                timeout=timeout,
                args=tuple(args),
                kwargs=kwargs or {},
                barrier=barrier,
            )

        replies = await asyncio.get_event_loop().run_in_executor(None, _blocking)
        failed = [r for r in replies if not r.ok]
        if failed:
            raise RuntimeError(
                f"collective_rpc({method}) failed on {len(failed)}/{len(replies)} "
                "ranks: " + "; ".join(f"tp{r.tp_rank}: {r.error}" for r in failed[:8])
            )
        return [r.value for r in replies]

    async def get_capabilities(self) -> Any:
        """What this ATOM engine supports, negotiated rather than assumed.

        Lets the caller stop probing with ``hasattr`` and stop hardcoding
        behaviour per backend. Features are intersected across ranks upstream,
        so anything reported here is usable by a collective.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")

        def _blocking():
            return self.engine.get_capabilities()

        return await asyncio.get_event_loop().run_in_executor(None, _blocking)

    def rdma_preflight(self, interface: str, hca: str) -> dict[str, Any]:
        """Fail now, loudly, if this container cannot do RDMA at all.

        Without this the group still forms and the transfer still "works" --
        over TCP, at a fraction of the bandwidth, with nothing in the logs
        saying so. Checked before the rendezvous because a missing device here
        would otherwise surface as all nine ranks hanging for the full timeout.
        """
        from pathlib import Path

        uverbs = sorted(Path("/dev/infiniband").glob("uverbs*"))
        if not uverbs or not (Path("/sys/class/infiniband") / hca).exists():
            raise RuntimeError(
                f"RDMA unavailable in ATOM rollout container: "
                f"uverbs={[p.name for p in uverbs]}, hca={hca!r}"
            )
        if os.environ.get("NCCL_IB_DISABLE", "0") == "1":
            raise RuntimeError("NCCL_IB_DISABLE=1 would force Socket transport")
        return {
            "replica": self.replica_rank,
            "interface": interface,
            "hca": hca,
            "uverbs": len(uverbs),
        }

    async def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join every rank of this replica to the trainer's broadcast group.

        Rank 0 is the trainer; this replica occupies ``base_rank`` upward. The
        per-rank offset is computed worker-side, since only the worker knows its
        own TP rank and local DP rank.
        """
        await self.collective_rpc(
            "init_rdma_weight_group",
            kwargs={
                "master_addr": master_addr,
                "master_port": int(master_port),
                "base_rank": int(base_rank),
                "world_size": int(world_size),
                "group_name": group_name,
                "timeout_s": int(timeout_s),
            },
            timeout=float(timeout_s),
        )
        return True

    async def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
        prequantized_fp8: bool = False,
    ) -> Any:
        """Receive one weight version into every rank of this replica."""
        if prequantized_fp8:
            # Rejected rather than ignored: silently dropping it would leave the
            # trainer quantising and ATOM expecting BF16, which shows up as
            # garbage output rather than an error.
            raise NotImplementedError(
                "ATOM's RDMA receive path is BF16-only; "
                "weight_sync fp8 quantization on the trainer is not supported yet"
            )
        stats = await self.collective_rpc(
            "receive_weights_rdma",
            kwargs={
                "group_name": group_name,
                "version": int(version),
                "verify_full_load": bool(verify_full_load),
            },
            # A weight stream is tens of GB; the default RPC budget is far too
            # short for it.
            timeout=float(os.environ.get("LUMENRL_RDMA_RECV_TIMEOUT_S", "1800")),
        )
        await self.reset_prefix_cache()
        return stats

    async def destroy_rdma_weight_group(self, group_name: str) -> bool:
        if self.engine is None:
            return True
        await self.collective_rpc(
            "destroy_rdma_weight_group", kwargs={"group_name": group_name}
        )
        return True

    async def reset_prefix_cache(self) -> bool:
        if self.engine is not None and hasattr(self.engine, "clear_kv_cache"):
            self.engine.clear_kv_cache()
        elif self.engine is not None and hasattr(self.engine, "core_mgr"):
            self.engine.core_mgr.broadcast_utility_command_sync("clear_kv_cache")
        return True

    async def wait_for_requests_to_drain(self, timeout_s: float = 60.0) -> bool:
        """Block until the engine reports no pending requests.

        Previously ``return True``, which made every caller's drain barrier a
        no-op: a weight sync could begin while responses were still in flight,
        producing rollouts from a half-updated model. ATOM exposes
        ``is_finished()``, so the barrier can be real.
        """
        if self.engine is None:
            return True
        if not hasattr(self.engine, "is_finished"):
            logger.warning(
                "ATOM engine exposes no is_finished(); cannot drain, proceeding"
            )
            return True

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await asyncio.get_event_loop().run_in_executor(
                None, self.engine.is_finished
            ):
                return True
            await asyncio.sleep(0.05)
        logger.warning("ATOM requests did not drain within %.1fs", timeout_s)
        return False

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
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before reload_weights_from_path()."
            )

        import json

        from safetensors.torch import load_file

        from atom.rollout.weight_sync import load_weights_via_shm

        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            files = sorted(
                f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
            )

        def weight_iter():
            for fname in files:
                sd = load_file(os.path.join(weight_dir, fname))
                for name, tensor in sd.items():
                    yield name, tensor

        load_weights_via_shm(self.engine.core_mgr, weight_iter(), bucket_size_mb=2048)
        logger.info(
            "ATOMRayServer[%d]: reloaded weights from %s", self.replica_rank, weight_dir
        )
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
        # Retained rather than recomputed in create(): the RDMA rank striding
        # needs both, and a replica running DP internally contributes
        # tp * dp ranks to the group, not tp.
        self.tensor_parallel_size = int(
            self.engine_kwargs.get("tensor_parallel_size", 1) or 1
        )
        self.data_parallel_size = int(
            self.engine_kwargs.get("data_parallel_size", 1) or 1
        )
        self.rdma_group_name: Optional[str] = None

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
            disable_custom_ar = os.environ.get(
                "LUMENRL_DISABLE_CUSTOM_AR", "1" if atom_tp > 1 else "0"
            )
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids_str,
                "HIP_VISIBLE_DEVICES": gpu_ids_str,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "MASTER_PORT": str(nccl_port),
                "LUMENRL_DISABLE_CUSTOM_AR": disable_custom_ar,
                **(
                    {"ATOM_USE_CUSTOM_ALL_GATHER": "0"}
                    if disable_custom_ar in ("1", "true", "True")
                    else {}
                ),
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
            if true_vocab_size is not None:
                # Belt and braces across two ATOM generations. The pinned build
                # reads LUMENRL_ATOM_TRUE_VOCAB_SIZE from the environment above;
                # newer ones take Config.true_vocab_size and no longer look at
                # the env var. ATOM filters engine kwargs against the Config
                # dataclass fields and drops the rest, so the build that does not
                # know the field ignores this line rather than failing on it --
                # and the mask cannot go quiet just because the pin moved.
                engine_kwargs.setdefault("true_vocab_size", int(true_vocab_size))
            if disable_custom_ar in ("1", "true", "True"):
                engine_kwargs.setdefault(
                    "runner_qualname",
                    "lumenrl.engine.inference.model_runner_nocustomar.NoCustomARModelRunner",
                )

            _dbg = os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True")
            if _dbg:
                logger.info(
                    "[DBG] ATOMReplicaManager: replica %d — gpus=%s disable_ca=%s runner=%s",
                    r,
                    gpu_ids_str,
                    disable_custom_ar,
                    engine_kwargs.get("runner_qualname", "default"),
                )

            server = remote_cls.options(
                num_gpus=0,
                num_cpus=1,
                name=f"lumen-atom-replica-{r}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
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
            logger.info(
                "ATOMReplicaManager: replica %d/%d launched.", i + 1, num_replicas
            )
        logger.info(
            "ATOMReplicaManager: launched %d colocated rollout replicas (atom_tp=%d, workers=%d).",
            num_replicas,
            atom_tp,
            num_workers,
        )

    def _torch_compile_enabled(self) -> bool:
        """True when the ATOM engine will run torch.compile (no-eager or level>0)."""
        comp_cfg = self.engine_kwargs.get("compilation_config") or {}
        level = int(comp_cfg.get("level", 0) or 0)
        return level > 0 or not bool(self.engine_kwargs.get("enforce_eager", True))

    def _get_true_vocab_size(self) -> Optional[int]:
        try:
            from transformers import AutoTokenizer

            vocab_size = len(
                AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            )
            logger.info(
                "ATOMReplicaManager: true tokenizer vocab size = %d", vocab_size
            )
            return int(vocab_size)
        except Exception as exc:
            logger.warning(
                "ATOMReplicaManager: failed to resolve tokenizer vocab size: %s", exc
            )
            return None

    def _engine_kwargs_for_replica(
        self, replica_rank: int, job_id: str
    ) -> dict[str, Any]:
        kwargs = dict(self.engine_kwargs)
        if os.getenv("ATOM_ISOLATE_TORCH_COMPILE_CACHE", "0") not in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }:
            return kwargs

        if not self._torch_compile_enabled():
            return kwargs

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        cache_root = os.getenv(
            "ATOM_TORCH_COMPILE_CACHE_ROOT", "/tmp/atom_torch_compile_cache"
        )
        safe_job_id = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in str(job_id)
        )
        comp_cfg["cache_dir"] = os.path.join(
            cache_root, safe_job_id, f"replica_{replica_rank}"
        )
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

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 600.0,
    ) -> list:
        """Run *method* on every rank of every replica.

        Returns one list per replica, replica-ordered, each holding that
        replica's per-rank values. Kept nested rather than flattened so a caller
        checking coverage can still tell which replica a value came from.
        """
        import ray

        return ray.get(
            [
                s.collective_rpc.remote(method, args, kwargs, barrier, timeout)
                for s in self.servers
            ]
        )

    def get_capabilities(self) -> list:
        """Per-replica capabilities.

        Returned per replica rather than merged: replicas are separate engines,
        and a caller that needs a single answer should decide for itself whether
        to intersect them or to treat a disagreement as a configuration error.
        """
        import ray

        return ray.get([s.get_capabilities.remote() for s in self.servers])

    # ── RDMA weight transfer ──────────────────────────────────────────────
    #
    # Mirrors VLLMReplicaManager's interface exactly, so the backend-agnostic
    # _sync_weights_rdma in the trainer drives either one unchanged.

    @property
    def _ranks_per_replica(self) -> int:
        """Ranks one replica contributes to the weight group.

        A replica running DP internally is several engines behind one actor
        handle, and each of their TP ranks joins separately, so this is tp * dp
        rather than tp. Getting it wrong shifts every later replica's base rank
        and the rendezvous hangs with no useful error.
        """
        return self.tensor_parallel_size * self.data_parallel_size

    def init_rdma_weight_group(
        self,
        actor_wg,
        *,
        interface: str,
        hca: str,
        require_rdma: bool,
        timeout_s: int,
        group_name: str,
    ) -> dict[str, Any]:
        """Build one persistent trainer + all-workers RCCL communicator."""
        import ray

        self._assert_workers_can_receive_rdma()

        if require_rdma:
            # Preflight everyone before anybody rendezvouses: a container
            # missing its verbs device would otherwise park all ranks until the
            # timeout, naming none of them.
            checks = [actor_wg.call_single_async(0, "rdma_preflight", interface, hca)]
            checks.extend(
                server.rdma_preflight.remote(interface, hca) for server in self.servers
            )
            logger.info("ATOM RDMA preflight: %s", ray.get(checks))

        rendezvous = actor_wg.execute_rank_zero_sync("get_rdma_rendezvous", interface)
        master_addr = str(rendezvous["address"])
        master_port = int(rendezvous["port"])
        world_size = 1 + self.num_replicas * self._ranks_per_replica

        # Trainer takes rank 0; replica r takes the block starting at
        # 1 + r * ranks_per_replica. Every rank must call in or the rendezvous
        # blocks, so these go out together and are joined as a set.
        refs = [
            actor_wg.call_single_async(
                0,
                "init_rdma_weight_group",
                master_addr,
                master_port,
                world_size,
                group_name,
                timeout_s,
            )
        ]
        for replica_rank, server in enumerate(self.servers):
            refs.append(
                server.init_rdma_weight_group.remote(
                    master_addr,
                    master_port,
                    1 + replica_rank * self._ranks_per_replica,
                    world_size,
                    group_name,
                    timeout_s,
                )
            )
        ray.get(refs)

        self.rdma_group_name = group_name
        logger.info(
            "ATOM RDMA weight group ready: %s master=%s:%d world=%d "
            "(%d replicas x %d ranks + trainer)",
            group_name,
            master_addr,
            master_port,
            world_size,
            self.num_replicas,
            self._ranks_per_replica,
        )
        return {
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
        }

    def _assert_workers_can_receive_rdma(self) -> None:
        """Refuse to build a group the workers cannot serve.

        Uses ATOM's general capability negotiation rather than a bespoke
        handshake. Checked before the rendezvous for the same reason as
        preflight: afterwards the failure is a hang, not a message.
        """
        import ray

        for replica_rank, server in enumerate(self.servers):
            try:
                caps = ray.get(server.get_capabilities.remote())
            except Exception as exc:
                raise RuntimeError(
                    f"ATOM replica {replica_rank} could not report capabilities; "
                    "the engine predates capability discovery, so its RDMA "
                    "support cannot be confirmed"
                ) from exc
            if not caps.supports("rdma_weight_receive"):
                partial = getattr(caps, "partial", lambda: ())()
                hint = (
                    f" (present on some ranks but not all: {sorted(partial)})"
                    if "rdma_weight_receive" in partial
                    else ""
                )
                raise RuntimeError(
                    f"ATOM replica {replica_rank} does not support "
                    f"rdma_weight_receive{hint}; every rank needs the receiver "
                    "for the group to be usable"
                )

    def start_receive_weights_rdma(
        self,
        *,
        version: int,
        verify_full_load: bool,
        prequantized_fp8: bool = False,
    ) -> list:
        """Arm every replica's receiver and return the refs un-awaited.

        Deliberately not awaited here: all receivers and the trainer's sender
        must sit inside the broadcast at once. Joining them in turn would park
        the first receiver waiting for a sender that is itself waiting for this
        call to return.
        """
        if not self.rdma_group_name:
            raise RuntimeError("ATOM RDMA weight group has not been initialized")
        return [
            server.receive_weights_rdma.remote(
                self.rdma_group_name,
                int(version),
                bool(verify_full_load),
                bool(prequantized_fp8),
            )
            for server in self.servers
        ]

    def destroy_rdma_weight_group(self, actor_wg=None) -> None:
        import ray

        if not self.rdma_group_name:
            return
        refs = [
            server.destroy_rdma_weight_group.remote(self.rdma_group_name)
            for server in self.servers
        ]
        if actor_wg is not None:
            refs.append(actor_wg.call_single_async(0, "destroy_rdma_weight_group"))
        try:
            ray.get(refs)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the cause
            logger.warning("ATOM RDMA group teardown failed: %s", exc)
        self.rdma_group_name = None

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
