"""verl-aligned Ray rollout server: vLLM AsyncLLM inside a Ray actor.

This is LumenRL's port of verl's online rollout replica stack:

* ``VLLMRayServer``  ~ verl ``vLLMHttpServer`` -- a Ray actor that owns one
  vLLM ``AsyncLLM`` engine (TP=1) pinned to a single GPU, optionally serves a
  uvicorn OpenAI-compatible HTTP app, and exposes token-in/token-out
  ``generate`` + ``collective_rpc`` + ``sleep``/``wake_up`` +
  ``update_weights_from_ipc`` over Ray RPC.
* ``VLLMReplicaManager`` ~ verl ``vLLMReplica`` + ``LLMServerManager`` +
  ``CheckpointEngineManager`` -- driver-side helper that colocates one server
  actor on the same GPU as each training ``LumenActorWorker`` (NodeAffinity +
  explicit ``CUDA_VISIBLE_DEVICES``), then fans out generate / sleep / wake /
  weight-update across the replicas.

8 GPUs => 8 replicas, each TP=1 (verl's default DP=8 rollout layout).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class VLLMRayServer:
    """Ray actor hosting one vLLM AsyncLLM engine on a single pinned GPU."""

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any],
        replica_rank: int,
        http_port: int = 0,
        start_http: bool = False,
        base_seed: Optional[int] = None,
    ) -> None:
        # CUDA_VISIBLE_DEVICES / NOSET / replica-rank / job-id are injected via
        # the actor's runtime_env by VLLMReplicaManager before this process
        # starts, so the engine and its ZMQ IPC receiver land on the right GPU.
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.replica_rank = int(replica_rank)
        self.http_port = int(http_port)
        self.start_http = bool(start_http)
        self.base_seed = base_seed
        self.engine = None
        self._http_task = None
        self._http_ready = False

    async def launch(self) -> bool:
        """Build the AsyncLLM engine (and optional HTTP server)."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # verl-aligned per-replica seed: seed = base_seed + replica_rank, so the
        # colocated engines don't all share one RNG (verl: replica_rank+data.seed).
        if self.base_seed is not None and "seed" not in self.engine_kwargs:
            self.engine_kwargs["seed"] = int(self.base_seed) + self.replica_rank

        worker_ext = "lumenrl.engine.inference.vllm_colocate_worker_ext.vLLMColocateWorkerExtension"
        args = AsyncEngineArgs(
            model=self.model_name,
            worker_extension_cls=worker_ext,
            distributed_executor_backend="mp",
            **self.engine_kwargs,
        )
        logger.info("VLLMRayServer[%d]: engine seed=%s",
                    self.replica_rank, self.engine_kwargs.get("seed"))
        self.engine = AsyncLLM.from_engine_args(args)
        logger.info("VLLMRayServer[%d]: AsyncLLM ready.", self.replica_rank)

        # verl-aligned: mask OOV/padded logits at engine init (critical for online
        # FP8 rollout where the requantized lm_head can otherwise emit garbage after
        # a weight update). Best-effort; tokenizer length = true vocab size.
        try:
            from transformers import AutoTokenizer

            vocab_size = len(AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True))
            await self.engine.collective_rpc(
                "monkey_patch_model", kwargs={"vocab_size": vocab_size}
            )
            logger.info("VLLMRayServer[%d]: monkey_patch_model applied (vocab=%d).",
                        self.replica_rank, vocab_size)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("VLLMRayServer[%d]: monkey_patch_model failed: %s",
                           self.replica_rank, exc)

        if self.start_http:
            try:
                await self._start_http()
            except Exception as exc:  # HTTP is best-effort; RPC path still works
                logger.warning("VLLMRayServer[%d]: HTTP server failed: %s",
                               self.replica_rank, exc)
        return True

    async def _start_http(self) -> None:
        import uvicorn
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except Exception:
            from vllm.utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        http_args = parser.parse_args([])
        http_args.model = self.model_name

        app = build_app(http_args)
        import inspect

        sig = inspect.signature(init_app_state)
        vllm_config = None
        if hasattr(self.engine, "get_vllm_config"):
            vllm_config = await self.engine.get_vllm_config()
        if "vllm_config" in sig.parameters and vllm_config is not None:
            await init_app_state(self.engine, vllm_config, app.state, http_args)
        else:
            await init_app_state(self.engine, app.state, http_args)

        cfg = uvicorn.Config(app, host="0.0.0.0", port=self.http_port, log_level="warning")
        server = uvicorn.Server(cfg)
        self._http_task = asyncio.create_task(server.serve())
        self._http_ready = True
        logger.info("VLLMRayServer[%d]: HTTP on :%d", self.replica_rank, self.http_port)

    def ready(self) -> bool:
        return self.engine is not None

    async def generate(
        self,
        prompt: Any,
        sampling_params: dict[str, Any],
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Single-prompt generate. ``prompt`` may be a text string (vLLM tokenizes
        it, matching the FIFO/verl path) or a list of token ids (TokensPrompt)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        want_lp = sampling_params.get("logprobs", None) is not None
        sp = SamplingParams(
            n=1,
            max_tokens=int(sampling_params.get("max_tokens", 512)),
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            logprobs=(0 if want_lp else None),
            seed=sampling_params.get("seed", None),
            stop_token_ids=sampling_params.get("stop_token_ids", None),
        )
        vllm_prompt = prompt if isinstance(prompt, str) else TokensPrompt(prompt_token_ids=list(prompt))
        rid = request_id or uuid4().hex
        final = None
        async for out in self.engine.generate(vllm_prompt, sp, request_id=rid):
            final = out
        comp = final.outputs[0]
        tok = list(comp.token_ids)
        lps = None
        if want_lp and comp.logprobs is not None:
            lps = [float(comp.logprobs[i].get(t).logprob) for i, t in enumerate(tok)]
        out = {
            "text": comp.text,
            "prompt_token_ids": list(final.prompt_token_ids),
            "token_ids": tok,
            "logprobs": lps,
        }
        # Rollout Routing Replay: the expert ids this engine actually selected,
        # uint8 [prompt + response - 1, num_layers, top_k]. Row t is the routing
        # of the forward at position t, i.e. the one that produced token t+1 --
        # the same alignment as `logprobs`. Present only when the engine was
        # built with enable_return_routed_experts.
        routed = getattr(comp, "routed_experts", None)
        if routed is not None:
            out["routed_experts"] = routed
        return out

    async def generate_batch(
        self,
        prompts: list,
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Submit every prompt (text or token ids) as its own concurrent request."""
        tasks = [
            self.generate(p, sampling_params, request_id=uuid4().hex)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    async def collective_rpc(
        self, method: str, args: tuple = (), kwargs: Optional[dict] = None
    ) -> Any:
        return await self.engine.collective_rpc(method, args=args, kwargs=kwargs or {})

    async def update_weights_from_ipc(self, use_shm: bool = False) -> bool:
        """Start the in-worker IPC receiver; blocks until the sender completes."""
        await self.engine.collective_rpc(
            "update_weights_from_ipc", kwargs={"use_shm": use_shm}
        )
        await self.engine.reset_prefix_cache()
        return True

    async def sleep(self, level: int = 2) -> bool:
        await self.engine.sleep(level=level)
        return True

    async def wake_up(self, tags: Optional[list[str]] = None) -> bool:
        await self.engine.wake_up(tags=tags)
        return True

    async def reset_prefix_cache(self) -> bool:
        await self.engine.reset_prefix_cache()
        return True

    async def wait_for_requests_to_drain(self, timeout_s: float = 60.0) -> bool:
        """Best-effort wait until no unfinished requests remain."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            n = 0
            try:
                if hasattr(self.engine, "get_num_unfinished_requests"):
                    n = self.engine.get_num_unfinished_requests()
            except Exception:
                n = 0
            if not n:
                return True
            await asyncio.sleep(0.05)
        return True

    async def shutdown(self) -> bool:
        try:
            if self._http_task is not None:
                self._http_task.cancel()
        except Exception:
            pass
        try:
            if self.engine is not None and hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
        except Exception:
            pass
        self.engine = None
        return True


class VLLMReplicaManager:
    """Driver-side controller for a fleet of colocated ``VLLMRayServer`` actors.

    Mirrors verl's ``vLLMReplica`` placement + ``LLMServerManager`` fan-out +
    ``CheckpointEngineManager`` sleep/wake orchestration, adapted to LumenRL's
    ``RayWorkerGroup``.
    """

    def __init__(
        self,
        actor_wg,
        model_name: str,
        engine_kwargs: dict[str, Any],
        *,
        base_port: int = 8700,
        start_http: bool = False,
        max_concurrency: int = 64,
        base_seed: Optional[int] = None,
    ) -> None:
        self.actor_wg = actor_wg
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.base_port = int(base_port)
        self.start_http = bool(start_http)
        self.max_concurrency = int(max_concurrency)
        self.base_seed = base_seed
        self.num_replicas = actor_wg.num_workers
        self.servers: list = []

    def create(self) -> None:
        """Create + launch one server actor colocated with each training actor."""
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        job_id = ray.get_runtime_context().get_job_id()
        # Ask each training actor where it lives + which physical GPU it owns.
        infos = self.actor_wg.execute_all_sync("get_colocation_info")
        logger.info("VLLMReplicaManager: colocation infos = %s", infos)

        remote_cls = ray.remote(VLLMRayServer)
        for i, info in enumerate(infos):
            node_id = info["node_id"]
            gpu_ids = ",".join(str(g) for g in info["gpu_ids"])
            # ROCm device pinning: select the physical GPU via CUDA/HIP visible
            # devices ONLY. Do NOT also set ROCR_VISIBLE_DEVICES to the physical
            # index -- ROCR filters at a lower level, so ROCR=<phys>+HIP=<phys>
            # double-filters ("No HIP GPUs available"). We still set the NOSET
            # flags (incl. ROCR) so Ray (num_gpus=0) doesn't clear visibility.
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
            server = remote_cls.options(
                num_gpus=0,  # pinned manually via CUDA_VISIBLE_DEVICES; no Ray GPU slot
                num_cpus=1,
                name=f"lumen-vllm-replica-{i}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=self.model_name,
                engine_kwargs=self.engine_kwargs,
                replica_rank=i,
                http_port=self.base_port + i,
                start_http=self.start_http,
                base_seed=self.base_seed,
            )
            self.servers.append(server)

        import ray as _ray
        _ray.get([s.launch.remote() for s in self.servers])
        logger.info("VLLMReplicaManager: launched %d colocated rollout replicas.",
                    len(self.servers))

    # -- fan-out helpers -------------------------------------------------
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
