"""High-level RL training orchestration on the controller process.

Supports three modes:
- **Sync colocate** (default): rollout and training share the same GPUs,
  swapping memory between vLLM and FSDP2 via optimizer offload.
- **Async separated**: Rollouter and Trainer run on separate GPU groups with
  a message queue and periodic parameter sync (see ``AsyncRLTrainer``).
- **Local mode** (single-GPU / testing): all workers run in the controller process.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Type

import torch

import lumenrl.algorithms  # noqa: F401  — populate ALGORITHM_REGISTRY
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.controller import DispatchMode, RayCluster, RayWorkerGroup, create_fused_worker_cls
from lumenrl.controller.dispatch import dispatch_proto
from lumenrl.engine.training.base_engine import BaseEngine, EngineRegistry
from lumenrl.quantization.rollout_correction import apply_rollout_correction
from lumenrl.trainer.callbacks import Callback, LoggingCallback
from lumenrl.utils.metrics import MetricsTracker, compute_kl_divergence
from lumenrl.utils.profiler import DistProfiler
from lumenrl.workers import LumenActorWorker, RefPolicyWorker

logger = logging.getLogger(__name__)


def _algo_num_generations(config: LumenRLConfig) -> int:
    name = config.algorithm.name.lower()
    if name == "ppo":
        return 1
    if name == "dapo":
        return int(config.algorithm.dapo.num_generations)
    return int(config.algorithm.grpo.num_generations)


class RLTrainer:
    """Coordinates rollout, reference, reward, and actor for RL training.

    In local mode, all computation happens in-process without Ray, using
    ``HFEngine`` for generation and ``FSDP2Backend`` for training.
    """

    def __init__(self, config: LumenRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []
        self._algorithm: Any = None
        self._metrics = MetricsTracker()

        self._engine: BaseEngine | None = None
        self._actor_model: torch.nn.Module | None = None
        self._ref_model: torch.nn.Module | None = None
        self._ref_on_cpu: bool = True
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None
        self._atom_engine: Any = None
        self._use_atom: bool = config.policy.generation_backend.lower() == "atom"
        # Ray controller orchestration path is opt-in via config/env/ray address.
        self._use_ray_controller: bool = (
            bool(getattr(config.controller.ray, "enabled", False))
            or
            os.environ.get("LUMENRL_USE_RAY_CONTROLLER", "0") == "1"
            or bool(getattr(config.cluster, "ray_address", None))
        )
        self._critic_worker: Any = None
        self._ray_cluster: RayCluster | None = None
        self._actor_wg: RayWorkerGroup | None = None
        self._ref_wg: RayWorkerGroup | None = None
        self._ray_dispatch_state: dict[str, Any] = {}
        self._profiler: DistProfiler | None = None
        self._prev_step_profile: bool = False
        self._curr_step_profile: bool = False
        self._val_dataset: Any = None
        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = torch.distributed.get_rank() if self._is_distributed else 0
        self._world_size: int = torch.distributed.get_world_size() if self._is_distributed else 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def setup(self) -> None:
        """Initialize models, optimizer, dataset, and algorithm."""
        model_name = self.config.policy.model_name
        if not model_name:
            raise ValueError("config.policy.model_name is required.")

        algo_cls: Type[Any] = ALGORITHM_REGISTRY.get(self.config.algorithm.name)
        self._algorithm = algo_cls(self.config)

        if self._use_ray_controller:
            self._setup_ray_controller()
            return

        quant = {}
        tq = self.config.quantization.training
        if tq.fp8:
            quant["fp8"] = tq.fp8
        quant["fp8_weight_cache"] = tq.fp8_weight_cache
        quant["lumen_norm"] = tq.lumen_norm
        quant["fused_mlp"] = tq.fused_mlp
        quant["fused_rope"] = tq.fused_rope

        optimizer_dtype_str = getattr(self.config.policy.training, "optimizer_dtype", "bf16")
        lr = getattr(self.config.policy, "learning_rate", 1e-6)
        self._base_lr = lr
        self._lr_warmup_steps = getattr(self.config.policy, "lr_warmup_steps", 10)
        self._param_offload = False

        fsdp_cfg_dict: dict = {}
        if hasattr(self.config.policy, "training") and hasattr(self.config.policy.training, "fsdp_cfg"):
            _fc = self.config.policy.training.fsdp_cfg
            if isinstance(_fc, dict):
                fsdp_cfg_dict = _fc
                self._param_offload = _fc.get("param_offload", False)

        backend_str = getattr(self.config.policy, "training_backend", "fsdp2").lower()
        if backend_str in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        elif backend_str == "megatron":
            backend_key = "megatron"
        else:
            backend_key = "fsdp2"

        logger.info("[rank %d] Building actor model via Engine layer: %s (backend=%s, optimizer_dtype=%s)",
                    self._rank, model_name, backend_key, optimizer_dtype_str)

        engine_config = {
            "param_offload": fsdp_cfg_dict.get("param_offload", False),
            "optimizer_offload": fsdp_cfg_dict.get("optimizer_offload", False),
            "grad_offload": fsdp_cfg_dict.get("grad_offload", False),
            "reshard_after_forward": fsdp_cfg_dict.get("reshard_after_forward", True),
            "model_dtype": optimizer_dtype_str,
            "seed": getattr(self.config, "seed", 42),
        }
        optimizer_config = {
            "lr": lr,
            "weight_decay": getattr(self.config.policy, "weight_decay", 0.01),
            "clip_grad": getattr(self.config.policy, "max_grad_norm", 1.0),
            "lr_scheduler_type": "cosine",
            "lr_warmup_steps": self._lr_warmup_steps,
            "lr_warmup_steps_ratio": getattr(self.config.policy, "warmup_ratio", 0.0),
            "total_training_steps": int(self.config.num_training_steps),
        }
        model_config = {
            "local_path": model_name,
            "trust_remote_code": True,
        }

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend=backend_key,
        )
        self._engine = engine_cls(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            model_name=model_name,
            quant_config=quant,
        )
        self._engine.initialize()
        self._actor_model = self._engine.module
        self._optimizer = self._engine.optimizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        kl_coeff = 0.0
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            kl_coeff = self.config.algorithm.dapo.kl_coeff
        elif algo_name == "grpo":
            kl_coeff = self.config.algorithm.grpo.kl_coeff
        elif algo_name == "ppo":
            kl_coeff = self.config.algorithm.ppo.kl_coeff

        if kl_coeff > 0.0:
            logger.info("[rank %d] Loading reference model (kl_coeff=%.4f): %s",
                        self._rank, kl_coeff, model_name)
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
            self._ref_on_cpu = True
        else:
            self._ref_model = None
            self._ref_on_cpu = True
            logger.info("[rank %d] Skipping reference model (kl_coeff=0).", self._rank)

        if self._use_atom:
            from lumenrl.engine.inference.atom_engine import AtomEngine
            atom_cfg = self.config.policy.generation.atom_cfg
            self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
            logger.info("[rank %d] ATOM engine configured (lazy init on first rollout).", self._rank)

        self._load_dataset()

        # ---- Validation dataset ----
        val_path = getattr(self.config, 'val_dataset', '')
        if val_path:
            self._load_val_dataset(val_path)

        if not self.callbacks:
            self.callbacks.append(LoggingCallback(interval=max(1, self.config.logger.log_interval)))

        # ---- Critic worker (value model for PPO/GAE) ----
        if getattr(self.config, 'critic', None) and self.config.critic.enabled:
            from lumenrl.workers import CriticWorker
            critic_config_dict = {
                "critic": {
                    "model_name": self.config.critic.model_name or self.config.policy.model_name,
                    "training_backend": self.config.critic.training_backend,
                    "learning_rate": self.config.critic.learning_rate,
                    "weight_decay": self.config.critic.weight_decay,
                    "max_grad_norm": self.config.critic.max_grad_norm,
                    "value_clip_ratio": self.config.critic.value_clip_ratio,
                },
                "policy": {
                    "model_name": self.config.critic.model_name or self.config.policy.model_name,
                    "training": vars(self.config.policy.training) if hasattr(self.config.policy.training, '__dict__') else {},
                    "seed": getattr(self.config, 'seed', 42),
                },
            }
            self._critic_worker = CriticWorker(self._rank, self._world_size, critic_config_dict)
            self._critic_worker.init_model()
            logger.info("[rank %d] CriticWorker initialized.", self._rank)

        self._resume_step = 0
        if getattr(self.config.checkpointing, "resume", True):
            self._try_resume_checkpoint()

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info("[rank %d] RLTrainer.setup complete: algo=%s, model=%s, world_size=%d, atom=%s, resume_step=%d",
                     self._rank, self.config.algorithm.name, model_name, self._world_size, self._use_atom, self._resume_step)
        self._init_profiler()

    def _setup_ray_controller(self) -> None:
        """Initialize Ray cluster + worker groups for actor/ref orchestration."""
        if RayCluster is None or RayWorkerGroup is None:
            raise RuntimeError("Ray controller modules are unavailable in this environment.")
        if not self._use_atom:
            raise NotImplementedError(
                "Ray controller path currently requires policy.generation_backend=atom."
            )

        # Main path should not depend on torch.distributed collectives.
        self._is_distributed = False
        self._rank = 0
        self._world_size = 1

        model_name = self.config.policy.model_name
        cfg_dict = self._to_plain_dict(self.config)
        default_workers = max(1, self.config.cluster.num_nodes * self.config.cluster.gpus_per_node)
        ray_cfg = self.config.controller.ray

        self._ray_cluster = RayCluster(self.config.cluster)
        self._ray_cluster.init()

        kl_coeff = 0.0
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            kl_coeff = self.config.algorithm.dapo.kl_coeff
        elif algo_name == "grpo":
            kl_coeff = self.config.algorithm.grpo.kl_coeff
        elif algo_name == "ppo":
            kl_coeff = self.config.algorithm.ppo.kl_coeff
        actor_role = ray_cfg.actor
        ref_role = ray_cfg.ref
        actor_workers = actor_role.num_workers if actor_role.num_workers > 0 else default_workers
        ref_workers = ref_role.num_workers if ref_role.num_workers > 0 else default_workers
        actor_pool_name = ray_cfg.topology_map.get("actor", "actor")
        ref_pool_name = ray_cfg.topology_map.get("ref", "ref")

        actor_pool = self._ray_cluster.create_pool(
            actor_pool_name,
            num_gpus=max(1, actor_workers),
            process_on_nodes=actor_role.process_on_nodes,
            max_colocate_count=max(1, actor_role.max_colocate_count),
            detached=actor_role.detached,
            topology_tags=actor_role.topology_tags,
        )

        use_ref = kl_coeff > 0.0
        if ray_cfg.fuse_actor_ref and use_ref:
            if ref_workers != actor_workers:
                raise ValueError("controller.ray.fuse_actor_ref requires actor/ref num_workers to match.")
            fused_cls = create_fused_worker_cls(
                {"actor": LumenActorWorker, "ref": RefPolicyWorker},
                name="ActorRefFusedWorker",
            )
            fused_wg = RayWorkerGroup(
                worker_cls=fused_cls,
                pool=actor_pool,
                num_workers=actor_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(actor_role.dispatch_mode),
                detached=actor_role.detached,
            )
            fused_wg.start()
            spawned = fused_wg.spawn(["actor", "ref"])
            self._actor_wg = spawned["actor"]
            self._ref_wg = spawned["ref"]
            self._actor_wg.call_all("init_model")
            self._ref_wg.call_all("init_model")
        else:
            self._actor_wg = RayWorkerGroup(
                worker_cls=LumenActorWorker,
                pool=actor_pool,
                num_workers=actor_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(actor_role.dispatch_mode),
                detached=actor_role.detached,
            )
            self._actor_wg.start()
            self._actor_wg.call_all("init_model")

        if use_ref and self._ref_wg is None:
            ref_pool = self._ray_cluster.create_pool(
                ref_pool_name,
                num_gpus=max(1, ref_workers),
                process_on_nodes=ref_role.process_on_nodes,
                max_colocate_count=max(1, ref_role.max_colocate_count),
                detached=ref_role.detached,
                topology_tags=ref_role.topology_tags,
            )
            self._ref_wg = RayWorkerGroup(
                worker_cls=RefPolicyWorker,
                pool=ref_pool,
                num_workers=ref_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(ref_role.dispatch_mode),
                detached=ref_role.detached,
            )
            self._ref_wg.start()
            self._ref_wg.call_all("init_model")

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        from lumenrl.engine.inference.atom_engine import AtomEngine
        atom_cfg = self.config.policy.generation.atom_cfg
        self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
        self._load_dataset()

        # ---- Validation dataset ----
        val_path = getattr(self.config, 'val_dataset', '')
        if val_path:
            self._load_val_dataset(val_path)

        if not self.callbacks:
            self.callbacks.append(LoggingCallback(interval=max(1, self.config.logger.log_interval)))
        self._resume_step = 0
        logger.info(
            "RLTrainer.setup (ray-controller) complete: algo=%s, model=%s, actor_workers=%d, ref=%s",
            self.config.algorithm.name,
            model_name,
            actor_workers,
            self._ref_wg is not None,
        )
        self._init_profiler()

    @staticmethod
    def _to_plain_dict(config: Any) -> dict[str, Any]:
        if is_dataclass(config):
            return asdict(config)
        if isinstance(config, dict):
            return dict(config)
        return dict(vars(config))

    def _init_profiler(self) -> None:
        """Initialize trainer-side profiler dispatcher from config."""
        self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)
        self._prev_step_profile = False
        self._curr_step_profile = False

    def _is_profile_step(self, step: int) -> bool:
        if self._profiler is None or not self._profiler.check_enable():
            return False
        steps = self.config.profiler.steps
        return True if steps is None else (step in steps)

    def _maybe_start_profile(self, step: int) -> None:
        curr = self._is_profile_step(step)
        self._curr_step_profile = curr
        if not curr or self._profiler is None:
            return
        if self.config.profiler.profile_continuous_steps:
            if not self._prev_step_profile:
                self._profiler.start(profile_step=step)
        else:
            self._profiler.start(profile_step=step)

    def _maybe_stop_profile(self, step: int) -> None:
        if self._profiler is None:
            return
        next_step_profile = self._is_profile_step(step + 1)
        if self._curr_step_profile:
            if self.config.profiler.profile_continuous_steps:
                if not next_step_profile:
                    self._profiler.stop()
            else:
                self._profiler.stop()
        self._prev_step_profile = self._curr_step_profile
        self._curr_step_profile = next_step_profile

    def _load_dataset(self) -> None:
        """Load the training dataset from config."""
        dataset_path = self.config.reward.dataset
        if not dataset_path:
            logger.warning("No dataset configured; using synthetic prompts.")
            self._dataset = None
            return

        from datasets import load_dataset

        if os.path.isfile(dataset_path) or os.path.isdir(dataset_path):
            if dataset_path.endswith(".parquet"):
                self._dataset = load_dataset("parquet", data_files=dataset_path, split="train")
            elif dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                self._dataset = load_dataset("json", data_files=dataset_path, split="train")
            else:
                self._dataset = load_dataset(dataset_path, split="train")
        else:
            self._dataset = load_dataset(dataset_path, split="train")

        logger.info("Loaded dataset: %d samples from %s", len(self._dataset), dataset_path)

    def _load_val_dataset(self, path: str) -> None:
        """Load validation dataset."""
        from datasets import load_dataset

        if os.path.isfile(path) or os.path.isdir(path):
            if path.endswith(".parquet"):
                self._val_dataset = load_dataset("parquet", data_files=path, split="train")
            elif path.endswith((".jsonl", ".json")):
                self._val_dataset = load_dataset("json", data_files=path, split="train")
            else:
                self._val_dataset = load_dataset(path, split="train")
        else:
            self._val_dataset = load_dataset(path, split="train")

        logger.info("Loaded validation dataset: %d samples from %s", len(self._val_dataset), path)

    def _try_resume_checkpoint(self) -> None:
        """Load model + optimizer state from the latest checkpoint if available."""
        from lumenrl.utils.checkpoint import CheckpointManager

        ckpt_dir = self.config.checkpointing.checkpoint_dir
        latest = CheckpointManager.get_latest(ckpt_dir)
        if latest is None:
            logger.info("[rank %d] No checkpoint found in %s; training from scratch.", self._rank, ckpt_dir)
            return

        logger.info("[rank %d] Resuming from checkpoint: %s", self._rank, latest)
        payload = CheckpointManager.load(latest)
        self._resume_step = int(payload.get("step", 0)) + 1

        # Unwrap nested structure: CheckpointManager.save wraps state in
        # {"step": N, "state_dict": {actual_data}}, so model_state_dict and
        # optimizer_state_dict live one level deeper than expected.
        inner = payload.get("state_dict", {})
        if isinstance(inner, dict) and "model_state_dict" in inner:
            logger.info("[rank %d] Unwrapping nested checkpoint structure.", self._rank)
            payload = inner

        model_sd = payload.get("model_state_dict")
        if model_sd and self._actor_model is not None:
            if self._is_distributed:
                try:
                    from torch.distributed.checkpoint.state_dict import (
                        set_model_state_dict,
                        set_optimizer_state_dict,
                        StateDictOptions,
                    )
                    opts = StateDictOptions(full_state_dict=True)
                    set_model_state_dict(self._actor_model, model_sd, options=opts)
                    logger.info("[rank %d] Restored FSDP2 model state.", self._rank)
                except Exception as exc:
                    logger.warning("[rank %d] FSDP2 set_model_state_dict failed (%s); "
                                   "trying load_state_dict.", self._rank, exc)
                    self._actor_model.load_state_dict(model_sd, strict=False)
            else:
                self._actor_model.load_state_dict(model_sd, strict=False)
                logger.info("[rank %d] Restored model state.", self._rank)

        opt_sd = payload.get("optimizer_state_dict")
        if opt_sd and self._optimizer is not None:
            if self._is_distributed:
                try:
                    from torch.distributed.checkpoint.state_dict import (
                        set_optimizer_state_dict,
                        StateDictOptions,
                    )
                    opts = StateDictOptions(full_state_dict=True)
                    set_optimizer_state_dict(
                        self._actor_model, self._optimizer, opt_sd, options=opts,
                    )
                    logger.info("[rank %d] Restored FSDP2 optimizer state.", self._rank)
                except Exception as exc:
                    logger.warning("[rank %d] FSDP2 set_optimizer_state_dict failed (%s); "
                                   "trying load_state_dict.", self._rank, exc)
                    try:
                        self._optimizer.load_state_dict(opt_sd)
                    except Exception:
                        logger.warning("[rank %d] Optimizer state restore failed; using fresh optimizer.", self._rank)
            else:
                try:
                    self._optimizer.load_state_dict(opt_sd)
                    logger.info("[rank %d] Restored optimizer state.", self._rank)
                except Exception:
                    logger.warning("[rank %d] Optimizer state restore failed; using fresh optimizer.", self._rank)

        del payload
        gc.collect()
        logger.info("[rank %d] Resume complete. Will start from step %d.", self._rank, self._resume_step)

    def _get_batch_prompts(self, step: int) -> tuple[list[str], list[str]]:
        """Get a batch of (prompts, ground_truths) for the current step."""
        import json as _json

        g = _algo_num_generations(self.config)
        num_prompts = max(1, self.config.policy.train_global_batch_size // g)

        if self._dataset is None:
            prompts = [f"What is {i + step * num_prompts} + {i + step * num_prompts + 1}?"
                       for i in range(num_prompts)]
            gts = [str(2 * (i + step * num_prompts) + 1) for i in range(num_prompts)]
            return prompts, gts

        dataset_len = len(self._dataset)
        start = (step * num_prompts) % dataset_len
        indices = [(start + i) % dataset_len for i in range(num_prompts)]
        samples = [self._dataset[idx] for idx in indices]

        prompts = []
        gts = []
        for s in samples:
            prompt_raw = s.get("prompt") or s.get("question") or s.get("input") or ""
            if isinstance(prompt_raw, list):
                text_parts = [m.get("content", "") for m in prompt_raw if isinstance(m, dict)]
                prompt_text = "\n".join(text_parts)
            elif isinstance(prompt_raw, str) and prompt_raw.startswith("["):
                try:
                    msgs = _json.loads(prompt_raw)
                    prompt_text = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
                except (_json.JSONDecodeError, TypeError):
                    prompt_text = prompt_raw
            else:
                prompt_text = str(prompt_raw)

            rm_raw = s.get("reward_model", {})
            if isinstance(rm_raw, str):
                try:
                    rm_raw = _json.loads(rm_raw)
                except (_json.JSONDecodeError, TypeError):
                    rm_raw = {}
            if isinstance(rm_raw, dict):
                gt = rm_raw.get("ground_truth", "")
            else:
                gt = s.get("answer") or s.get("solution") or s.get("target") or ""

            if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
                raw = s.get("prompt") or s.get("question") or s.get("input") or ""
                if isinstance(raw, list):
                    try:
                        prompt_text = self._tokenizer.apply_chat_template(
                            raw, tokenize=False, add_generation_prompt=True,
                        )
                    except Exception:
                        pass

            prompts.append(prompt_text)
            gts.append(gt)

        return prompts, gts

    def _tokenize_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts to input_ids and attention_mask."""
        max_prompt_len = min(self.config.policy.max_total_sequence_length // 2, 1024)
        encoding = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def _log_gpu_mem(self, phase: str, step: int) -> None:
        """Log GPU 0 memory at a critical point (rank 0 only)."""
        if self._rank != 0:
            return
        free, total = torch.cuda.mem_get_info(0)
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(
            "GPU-MEM [step=%d phase=%s] alloc=%.1fGB reserved=%.1fGB free=%.1fGB/%.1fGB",
            step, phase, alloc, reserved, free / 1e9, total / 1e9,
        )

    def _offload_optimizer_to_cpu(self) -> None:
        """Move optimizer state tensors to CPU to free GPU for ATOM rollout.

        Delegates to Engine.to() when available.
        """
        if self._engine is not None:
            self._engine.to(device="cpu", model=False, optimizer=True, grad=False)
            torch.cuda.empty_cache()
            return
        if self._optimizer is None:
            return
        if self._param_offload:
            torch.cuda.empty_cache()
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    state[k] = v.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _reload_optimizer_to_gpu(self) -> None:
        """Move optimizer state tensors back to GPU for the next training step.

        Delegates to Engine.to() when available.
        """
        if self._engine is not None:
            self._engine.to(device="cuda", model=False, optimizer=True, grad=False)
            return
        if self._optimizer is None or self._param_offload:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    state[k] = v.to(self._device, non_blocking=True)

    def _sync_weights_to_atom(self) -> None:
        """Push updated FSDP2 weights to ATOM engine for next rollout.

        Follows verl's approach: extract per-tensor from FSDP2's DTensor
        state_dict, save to /dev/shm as safetensors (HF format), then
        tell the ATOM subprocess to reload from the new path.

        Sequence (GPU 0 memory-safe):
        1. Rank 0: ensure ATOM is already sleeping (done after generation)
        2. All ranks: FSDP2 ``full_tensor()`` all-gather (needs GPU headroom)
        3. Rank 0: save gathered weights, update ``_weight_dir``
        """
        if self._atom_engine is None:
            return

        t0 = time.time()

        if self._rank == 0 and not self._atom_engine._sleeping:
            self._atom_engine.sleep_inprocess()
            logger.info("Weight sync: ATOM engine released (in-process sleep).")

        if self._is_distributed:
            torch.distributed.barrier()
        torch.cuda.empty_cache()

        cpu_state = self._fetch_actor_cpu_state()
        if cpu_state is None:
            return
        torch.cuda.empty_cache()

        sync_dir = Path(os.environ.get(
            "LUMENRL_WEIGHT_SYNC_DIR", "/dev/shm/lumenrl_weight_sync",
        ))

        total_bytes = 0
        if self._rank == 0:
            sync_dir.mkdir(parents=True, exist_ok=True)

            from safetensors.torch import save_file

            max_shard_bytes = 4 * 1024 * 1024 * 1024
            shards: list[dict[str, torch.Tensor]] = [{}]
            current_bytes = 0
            for name, tensor in cpu_state.items():
                t_bytes = tensor.numel() * tensor.element_size()
                if current_bytes + t_bytes > max_shard_bytes and shards[-1]:
                    shards.append({})
                    current_bytes = 0
                shards[-1][name] = tensor
                current_bytes += t_bytes

            weight_map: dict[str, str] = {}
            for i, shard in enumerate(shards, 1):
                fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
                save_file(shard, str(sync_dir / fname))
                for k, v in shard.items():
                    weight_map[k] = fname
                    total_bytes += v.numel() * v.element_size()

            index = {
                "metadata": {"total_size": total_bytes},
                "weight_map": weight_map,
            }
            (sync_dir / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2)
            )

            orig = Path(self.config.policy.model_name)
            for fname in ["config.json", "tokenizer_config.json", "tokenizer.json",
                          "special_tokens_map.json", "generation_config.json",
                          "vocab.json", "merges.txt"]:
                src = orig / fname
                if src.exists():
                    shutil.copy2(str(src), str(sync_dir / fname))

            save_time = time.time() - t0
            logger.info(
                "Weight sync: saved %d params to %s in %.1fs (%.1f GB)",
                len(cpu_state), sync_dir, save_time, total_bytes / 1e9,
            )

            self._atom_engine._weight_dir = str(sync_dir)
            logger.info("Weight sync: ATOM will reload from %s on next generation.", sync_dir)

        del cpu_state
        gc.collect()

        if self._is_distributed:
            torch.distributed.barrier()

    def _fetch_actor_cpu_state(self) -> dict[str, torch.Tensor] | None:
        """Fetch actor weights as CPU tensors for rollout sync."""
        if self._use_ray_controller:
            if self._actor_wg is None:
                return None
            state = self._actor_wg.call_single(0, "get_state_dict")
            return {k: v.detach().cpu().contiguous() for k, v in state.items()}

        if self._actor_model is None:
            return None

        from torch.distributed._tensor import DTensor

        sd = self._actor_model.state_dict()
        cpu_state: dict[str, torch.Tensor] = {}
        for name, param in sd.items():
            full = param.full_tensor() if isinstance(param, DTensor) else param
            cpu_state[name] = full.detach().cpu().contiguous()
        return cpu_state

    def _rollout_with_atom(
        self,
        prompts: list[str],
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate using vLLM engine (PagedAttention, continuous batching).

        Only rank 0 runs generation; results are broadcast to all ranks.
        vLLM ``generate()`` returns response-only text, so we concatenate
        prompt + response and tokenize the full sequence.
        """
        algo_name = self.config.algorithm.name.lower()
        max_resp = self.config.policy.max_response_length
        if max_resp > 0:
            max_tok = max_resp
        else:
            max_tok = max(128, self.config.policy.max_total_sequence_length // 2)
        sp: dict[str, Any] = {"max_tokens": max_tok}
        if algo_name == "dapo":
            sp.update({"temperature": 1.0, "top_p": 0.95})
        elif algo_name == "grpo":
            sp.update({"temperature": 0.7})
        else:
            sp.update({"temperature": 0.0})

        expanded_prompts = []
        for p in prompts:
            for _ in range(num_generations):
                expanded_prompts.append(p)

        if os.environ.get("LUMENRL_DRY_RUN") == "1":
            target_len = int(os.environ.get("LUMENRL_DRY_RUN_RESP_LEN", "100"))
            mock_resp = ("Step: think carefully about this problem. " * (target_len // 7 + 1))[:target_len * 4] + " The answer is \\boxed{42}."
            response_texts = [mock_resp] * len(expanded_prompts)
            full_texts = [p + mock_resp for p in expanded_prompts]
            encoding = self._tokenizer(
                full_texts, padding=True, truncation=True,
                max_length=self.config.policy.max_total_sequence_length,
                return_tensors="pt",
            )
            sequences = encoding["input_ids"].to(self._device)
            seq_mask = encoding["attention_mask"].to(self._device)
            if self._is_distributed:
                torch.distributed.barrier()
                shape_tensor = torch.tensor(list(sequences.shape), device=self._device, dtype=torch.long)
                torch.distributed.broadcast(shape_tensor, src=0)
                torch.distributed.broadcast(sequences, src=0)
                torch.distributed.broadcast(seq_mask, src=0)
            prompt_lengths = []
            for p in expanded_prompts:
                p_enc = self._tokenizer(p, return_tensors="pt")
                prompt_lengths.append(p_enc["input_ids"].shape[1])
            return sequences, seq_mask, prompt_lengths

        if self._rank == 0:
            if self._atom_engine._sleeping:
                model_path = self._atom_engine._weight_dir or self._atom_engine._model_name
                self._atom_engine._send_cmd({
                    "cmd": "wake",
                    "model_path": model_path,
                })
                self._atom_engine._sleeping = False
                logger.info("AtomEngine: woke in-process with %s", model_path)
            elif not getattr(self._atom_engine, '_initialized', False):
                self._atom_engine.wake()
            response_texts = self._atom_engine.generate(expanded_prompts, sampling_params=sp)

            full_texts = [p + r for p, r in zip(expanded_prompts, response_texts)]
            encoding = self._tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.config.policy.max_total_sequence_length,
                return_tensors="pt",
            )
            sequences = encoding["input_ids"].to(self._device)
            seq_mask = encoding["attention_mask"].to(self._device)
        else:
            sequences = torch.zeros(1, 1, dtype=torch.long, device=self._device)
            seq_mask = torch.zeros(1, 1, dtype=torch.long, device=self._device)

        if self._is_distributed:
            torch.distributed.barrier()
            shape_tensor = torch.tensor(list(sequences.shape), device=self._device, dtype=torch.long)
            torch.distributed.broadcast(shape_tensor, src=0)
            if self._rank != 0:
                sequences = torch.zeros(
                    int(shape_tensor[0]), int(shape_tensor[1]),
                    dtype=torch.long, device=self._device,
                )
                seq_mask = torch.zeros_like(sequences)
            torch.distributed.broadcast(sequences, src=0)
            torch.distributed.broadcast(seq_mask, src=0)

        prompt_encoding = self._tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=min(1024, self.config.policy.max_total_sequence_length // 2),
            return_tensors="pt",
        )
        prompt_lengths = prompt_encoding["attention_mask"].sum(dim=1).tolist()

        return sequences, seq_mask, prompt_lengths

    def _set_reshard(self, reshard: bool) -> None:
        """Toggle FSDP2 reshard_after_forward on the actor model."""
        if not self._is_distributed:
            return
        try:
            from lumenrl.engine.training.fsdp_backend import set_reshard_after_forward
            set_reshard_after_forward(self._actor_model, reshard)
        except Exception:
            pass

    def _rollout_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate completions using the actor model in eval mode.

        Optimizations vs naive approach:
        - Phase 1a: disables reshard_after_forward during generate so FSDP2
          keeps parameters all-gathered across decode steps (eliminates
          O(L * tokens) redundant all-gathers).
        - Phase 1b: in distributed mode, shards prompts across ranks so each
          rank generates B/world_size sequences, then all-gathers results.

        Returns (sequences, seq_mask, prompt_lengths) on the current device.
        """
        prompt_lens = attention_mask.sum(dim=1).tolist()

        if num_generations > 1:
            input_ids = input_ids.repeat_interleave(num_generations, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
            prompt_lens = [l for l in prompt_lens for _ in range(num_generations)]

        if self._is_distributed and self._world_size > 1:
            total = input_ids.shape[0]
            chunk = max(1, total // self._world_size)
            start_idx = self._rank * chunk
            end_idx = start_idx + chunk if self._rank < self._world_size - 1 else total
            local_ids = input_ids[start_idx:end_idx]
            local_mask = attention_mask[start_idx:end_idx]
            local_plens = prompt_lens[start_idx:end_idx]
        else:
            local_ids = input_ids
            local_mask = attention_mask
            local_plens = prompt_lens

        max_resp = self.config.policy.max_response_length
        if max_resp > 0:
            max_gen = min(max_resp, self.config.policy.max_total_sequence_length - local_ids.shape[1])
        else:
            max_gen = max(128, self.config.policy.max_total_sequence_length - local_ids.shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_gen,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            gen_kwargs.update({"temperature": 1.0, "top_p": 0.95, "do_sample": True})
        elif algo_name == "grpo":
            gen_kwargs.update({"temperature": 0.7, "do_sample": True})
        else:
            gen_kwargs.update({"do_sample": False})

        self._actor_model.eval()
        had_grad_ckpt = hasattr(self._actor_model, "gradient_checkpointing_disable")
        if had_grad_ckpt:
            try:
                self._actor_model.gradient_checkpointing_disable()
            except Exception:
                pass

        self._set_reshard(False)

        ids_gpu = local_ids.to(self._device)
        mask_gpu = local_mask.to(self._device)

        with torch.no_grad():
            local_seqs = self._actor_model.generate(
                input_ids=ids_gpu,
                attention_mask=mask_gpu,
                **gen_kwargs,
            )

        self._set_reshard(True)

        if had_grad_ckpt:
            try:
                self._actor_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except Exception:
                pass

        if self._is_distributed and self._world_size > 1:
            sequences, seq_mask, prompt_lens = self._allgather_sequences(
                local_seqs, mask_gpu, local_plens,
            )
        else:
            sequences = local_seqs
            seq_mask = torch.ones(sequences.shape, dtype=torch.long, device=sequences.device)
            for i, plen in enumerate(local_plens):
                seq_mask[i, :plen] = local_mask[i, :plen].to(sequences.device)
                pad_id = self._tokenizer.pad_token_id
                if pad_id is not None:
                    seq_mask[i, plen:] = (sequences[i, plen:] != pad_id).long()
            prompt_lens = local_plens

        return sequences, seq_mask, prompt_lens

    def _allgather_sequences(
        self,
        local_seqs: torch.Tensor,
        local_mask: torch.Tensor,
        local_plens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """All-gather variable-length generated sequences across ranks.

        Pads to the global max sequence length, gathers, then trims.
        """
        local_max = torch.tensor([local_seqs.shape[1]], device=self._device)
        global_max = torch.zeros_like(local_max)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
        global_max_len = int(local_max.item())

        if local_seqs.shape[1] < global_max_len:
            pad = torch.full(
                (local_seqs.shape[0], global_max_len - local_seqs.shape[1]),
                self._tokenizer.pad_token_id or 0,
                dtype=local_seqs.dtype, device=local_seqs.device,
            )
            local_seqs = torch.cat([local_seqs, pad], dim=1)

        local_count = torch.tensor([local_seqs.shape[0]], device=self._device, dtype=torch.long)
        counts_list = [torch.zeros_like(local_count) for _ in range(self._world_size)]
        torch.distributed.all_gather(counts_list, local_count)
        max_count = max(c.item() for c in counts_list)

        if local_seqs.shape[0] < max_count:
            pad_rows = torch.zeros(
                (max_count - local_seqs.shape[0], global_max_len),
                dtype=local_seqs.dtype, device=local_seqs.device,
            )
            local_seqs = torch.cat([local_seqs, pad_rows], dim=0)

        gathered = [torch.zeros_like(local_seqs) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, local_seqs)

        all_seqs_list = []
        all_plens = []
        for r, (seqs_r, cnt) in enumerate(zip(gathered, counts_list)):
            n = int(cnt.item())
            all_seqs_list.append(seqs_r[:n])

        plens_tensor = torch.tensor(local_plens, device=self._device, dtype=torch.long)
        if plens_tensor.shape[0] < max_count:
            plens_tensor = torch.nn.functional.pad(plens_tensor, (0, max_count - plens_tensor.shape[0]))
        gathered_plens = [torch.zeros_like(plens_tensor) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered_plens, plens_tensor)
        for r, cnt in enumerate(counts_list):
            n = int(cnt.item())
            all_plens.extend(gathered_plens[r][:n].tolist())

        sequences = torch.cat(all_seqs_list, dim=0)

        pad_id = self._tokenizer.pad_token_id
        seq_mask = torch.ones(sequences.shape, dtype=torch.long, device=sequences.device)
        for i, plen in enumerate(all_plens):
            plen = int(plen)
            if pad_id is not None:
                seq_mask[i, :plen] = 1
                seq_mask[i, plen:] = (sequences[i, plen:] != pad_id).long()

        return sequences, seq_mask, [int(p) for p in all_plens]

    @staticmethod
    def _fused_token_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Per-token log-probs via row-chunked log_softmax to bound peak memory.

        Uses F.log_softmax per sequence row, which avoids promoting the full
        [S, V] logit tensor to float32 and is numerically stable for bf16.
        Matches VERL's ``logprobs_from_logits_v2`` bf16 path.
        """
        logits_shifted = logits[:, :-1]            # [B, S-1, V]  bf16 view
        targets = target_ids[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
        lp_parts = []
        for i in range(logits_shifted.shape[0]):
            row_lp = torch.nn.functional.log_softmax(logits_shifted[i], dim=-1)
            lp_parts.append(row_lp.gather(-1, targets[i]).squeeze(-1))
        return torch.stack(lp_parts, dim=0).float()  # [B, S-1]

    def _compute_log_probs_for_model(
        self,
        model: torch.nn.Module,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        move_to_gpu: bool = False,
    ) -> torch.Tensor:
        """Compute per-token log-probs from model for given sequences.

        Uses the same packed forward path as ``_train_step`` to ensure
        numerical consistency between ``old_log_probs`` and ``log_probs``.
        This is critical: if the two paths differ (e.g. padded SDPA vs
        packed varlen attention), the importance ratio deviates from 1.0
        even with identical weights, causing all ratios to be clipped
        and gradients to vanish.

        When ``move_to_gpu`` is True, moves the model to GPU before forward
        and back to CPU afterward (for CPU-offloaded reference model).
        """
        from lumenrl.engine.training.packing import (
            PackingContext, pack_sequences, packed_token_log_probs,
            unpack_log_probs,
        )

        if move_to_gpu:
            model.to(self._device)

        sequences = sequences.to(self._device)
        attention_mask = attention_mask.to(self._device)

        model.eval()
        S = sequences.shape[1]

        # Use _dynamic_mini_batches-style chunking by actual token count
        max_tok = int(self.config.policy.max_token_len_per_gpu)
        seq_lens = attention_mask.sum(dim=1).long()
        all_log_probs = []

        # Build chunk boundaries by actual token budget
        chunks: list[tuple[int, int]] = []
        start = 0
        n = sequences.shape[0]
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(seq_lens[end].item())
                if tok_count + sl > max_tok and end > start:
                    break
                tok_count += sl
                end += 1
            chunks.append((start, end))
            start = end

        # FSDP2: equalize chunk count across ranks (same fix as _train_step)
        real_chunk_count = len(chunks)
        if self._is_distributed and self._world_size > 1:
            import torch.distributed as dist
            count_t = torch.tensor([real_chunk_count], device=self._device)
            dist.all_reduce(count_t, op=dist.ReduceOp.MAX)
            global_max = int(count_t.item())
            while len(chunks) < global_max:
                chunks.append(chunks[-1])  # dummy: reuse last chunk for FSDP2 collectives

        with torch.no_grad():
            for ci, (cs, ce) in enumerate(chunks):
                ids_chunk = sequences[cs:ce]
                mask_chunk = attention_mask[cs:ce]

                packed = pack_sequences(ids_chunk, mask_chunk)
                with PackingContext(packed.cu_seqlens, packed.max_seqlen):
                    outputs = model(
                        input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    logits = logits.squeeze(0)
                    flat_lp = packed_token_log_probs(
                        logits, packed.input_ids.squeeze(0), packed.cu_seqlens,
                    )
                    token_lp = unpack_log_probs(
                        flat_lp, packed.cu_seqlens, packed.seq_lens, S,
                    )
                    if ci < real_chunk_count:
                        all_log_probs.append(token_lp)
                    del outputs, logits

        if move_to_gpu:
            model.cpu()
            torch.cuda.empty_cache()

        return torch.cat(all_log_probs, dim=0)

    def _compute_rewards(
        self,
        sequences: torch.Tensor,
        prompt_lengths: list[int],
        ground_truths: list[str],
        num_generations: int,
    ) -> tuple[torch.Tensor, list[str]]:
        """Decode responses and compute math rewards.

        Returns rewards on ``self._device`` to avoid a CPU round-trip.
        """
        seq_cpu = sequences.cpu() if sequences.device.type != "cpu" else sequences
        responses = []
        for i in range(seq_cpu.shape[0]):
            plen = prompt_lengths[i]
            response_ids = seq_cpu[i, plen:]
            text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(text)

        expanded_gts = []
        for gt in ground_truths:
            expanded_gts.extend([gt] * num_generations)

        from lumenrl.rewards.math_reward import compute_math_reward

        rewards, details = compute_math_reward(responses, expanded_gts)

        accuracy = sum(1 for d in details if d["acc"]) / max(1, len(details))
        logger.info("Reward: accuracy=%.4f, mean=%.4f", accuracy, rewards.mean().item())

        n_pos = sum(1 for r in rewards if r > 0)
        n_neg = sum(1 for r in rewards if r < 0)
        n_invalid = sum(1 for d in details if d.get("pred") == "[INVALID]")
        logger.info(
            "Reward breakdown: +1=%d, -1=%d, invalid_format=%d / %d total",
            n_pos, n_neg, n_invalid, len(details),
        )
        for idx in range(min(2, len(responses), len(details))):
            tail = responses[idx][-400:]
            pred = details[idx].get("pred", "N/A")
            gt = expanded_gts[idx] if idx < len(expanded_gts) else "?"
            logger.info(
                "Sample[%d] reward=%.1f pred=%s gt=%s tail=...%s",
                idx, rewards[idx].item(), pred, gt, repr(tail[-200:]),
            )

        return rewards.to(self._device), responses

    def _build_response_mask(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Create a mask that is 1 only for response tokens (excluding prompt).

        For left-padded sequences, prompt tokens start at the first non-pad
        position.  We must zero out prompt positions correctly, not just the
        first ``plen`` columns (which are padding for left-padded inputs).
        The returned mask is ``[:, 1:]`` to align with shifted log-probs.
        """
        B, S = attention_mask.shape
        mask = attention_mask.clone()
        for i, plen in enumerate(prompt_lengths):
            # Find where actual tokens start (first 1 in attention_mask)
            actual_start = int((attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item())
            # Prompt spans [actual_start, actual_start + plen)
            mask[i, actual_start:actual_start + plen] = 0
        return mask[:, 1:]

    def _update_lr(self, step: int) -> None:
        """Advance LR scheduler via Engine, falling back to manual warmup."""
        if self._engine is not None and self._engine.lr_scheduler is not None:
            return
        if step < self._lr_warmup_steps:
            warmup_lr = self._base_lr * (step + 1) / self._lr_warmup_steps
        else:
            warmup_lr = self._base_lr
        for pg in self._optimizer.param_groups:
            pg["lr"] = warmup_lr

    def _dynamic_mini_batches(
        self,
        batch: DataProto,
        max_token_len: int,
        fallback_bs: int,
    ) -> list[DataProto]:
        """Split batch into mini-batches capped by total *actual* token count.

        Uses actual sequence lengths (from ``attention_mask``) and sorts by
        length for efficient packing.  With sequence packing, each
        mini-batch contains multiple sequences whose total actual tokens
        fit within ``max_token_len``.
        """
        if max_token_len <= 0:
            return list(batch.mini_batches(fallback_bs))

        # Use actual token lengths for packing (not padded length).
        seq_lens = batch.tensors["attention_mask"].sum(dim=1).long()

        # Sort by length for better packing (similar lengths together).
        sorted_idx = torch.argsort(seq_lens)
        sorted_lens = seq_lens[sorted_idx]
        sorted_tensors = {k: v[sorted_idx] for k, v in batch.tensors.items()}

        batches: list[DataProto] = []
        start = 0
        n = batch.batch_size
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(sorted_lens[end].item())
                if tok_count + sl > max_token_len and end > start:
                    break
                tok_count += sl
                end += 1
            chunk = {k: v[start:end] for k, v in sorted_tensors.items()}
            batches.append(DataProto(tensors=chunk, meta=batch.meta.copy()))
            start = end
        return batches

    def _train_step(
        self,
        batch: DataProto,
        loss_scale: float = 1.0,
        dp_size: int = 1,
    ) -> dict[str, float]:
        """One gradient step on the actor model (with sequence packing)."""
        if self._actor_model is None or self._optimizer is None:
            raise RuntimeError("setup() must be called first.")

        from lumenrl.engine.training.packing import (
            PackingContext, pack_sequences, packed_token_log_probs,
            unpack_log_probs,
        )

        self._actor_model.train()
        sequences = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        # Compute per-mini-batch batch_num_tokens via all-reduce (Verl pattern).
        # Each mini-batch normalizes by its own global token count, not the
        # total across all mini-batches.
        _resp = batch.tensors.get("response_mask")
        if _resp is not None:
            _local_mb_tokens = int(_resp.sum())
        else:
            _local_mb_tokens = int(attention_mask.sum())
        if self._is_distributed:
            _tok_t = torch.tensor(_local_mb_tokens, device=self._device)
            torch.distributed.all_reduce(_tok_t, op=torch.distributed.ReduceOp.SUM)
            mb_batch_num_tokens = int(_tok_t.item())
        else:
            mb_batch_num_tokens = _local_mb_tokens

        # Pack multiple sequences into a single flat tensor for the forward pass
        packed = pack_sequences(sequences, attention_mask)

        # PackingContext stays alive through backward (gradient checkpointing)
        with PackingContext(packed.cu_seqlens, packed.max_seqlen):
            outputs = self._actor_model(
                input_ids=packed.input_ids,
                position_ids=packed.position_ids,
                attention_mask=None,  # varlen attention handles masking
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = logits.squeeze(0)  # (total_tokens, V)

            flat_lp = packed_token_log_probs(
                logits, packed.input_ids.squeeze(0), packed.cu_seqlens,
            )
            del outputs, logits

            # Unpack to [B, S-1] padded format (matches old_log_probs shape)
            token_log_probs = unpack_log_probs(
                flat_lp, packed.cu_seqlens, packed.seq_lens, sequences.shape[1],
            )
            batch.tensors["log_probs"] = token_log_probs

            # Mismatch KL: divergence between rollout (old) and training log_probs.
            # Should be ~0 when both use the same computation path and policy.
            if _resp is not None:
                _diff = (token_log_probs - batch.tensors["old_log_probs"]).detach()
                _rm = _resp.to(dtype=_diff.dtype)
                _denom = _rm.sum().clamp(min=1.0)
                _mismatch_kl = float((_diff * _rm).sum() / _denom)
            else:
                _mismatch_kl = float((token_log_probs - batch.tensors["old_log_probs"]).detach().mean())

            # Attach per-mini-batch global normalization info for Verl-aligned loss
            batch.meta["batch_num_tokens"] = mb_batch_num_tokens
            batch.meta["dp_size"] = dp_size

            loss, metrics = self._algorithm.compute_loss(batch)
            loss = loss.to(self._device)

            if loss.isnan():
                metrics["loss"] = float("nan")
                return metrics

            # dp_size compensation is now inside the loss (via batch_num_tokens),
            # so we only apply loss_scale for gradient accumulation here.
            (loss * loss_scale).backward()

        metrics["loss"] = float(loss.detach())
        metrics["mismatch_kl"] = _mismatch_kl
        return metrics

    def train(self) -> None:
        """Main training loop: rollout → ref → reward → advantages → train → sync."""
        if self._use_ray_controller:
            self._train_with_ray_controller()
            return

        if self._algorithm is None or self._actor_model is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        num_generations = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)
        start_step = self._resume_step
        if start_step > 0:
            logger.info("[rank %d] Skipping steps 0..%d (resuming from checkpoint).", self._rank, start_step - 1)

        for step in range(start_step, total_steps):
            step_start = time.time()
            self.global_step = step
            self._maybe_start_profile(step)
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            prompts, ground_truths = self._get_batch_prompts(step)
            input_ids, attention_mask = self._tokenize_prompts(prompts)

            if self._use_atom and self._atom_engine is not None:
                self._offload_optimizer_to_cpu()

            self._log_gpu_mem("pre_gen", step)
            t0 = time.time()
            if self._use_atom and self._atom_engine is not None:
                sequences, seq_mask, prompt_lengths = self._rollout_with_atom(
                    prompts, num_generations,
                )
            else:
                sequences, seq_mask, prompt_lengths = self._rollout_phase(
                    input_ids, attention_mask, num_generations,
                )
            gen_time = time.time() - t0
            self._log_gpu_mem("post_gen", step)

            if self._use_atom and self._atom_engine is not None and self._rank == 0:
                if not self._atom_engine._sleeping:
                    self._atom_engine.sleep_inprocess()
                    logger.info("ATOM slept after generation — freeing GPU 0 for training.")
            if self._use_atom and self._is_distributed:
                torch.distributed.barrier()
            torch.cuda.empty_cache()
            self._log_gpu_mem("post_atom_sleep", step)

            prompt_tok = int(attention_mask.repeat_interleave(
                num_generations, dim=0).to(seq_mask.device).sum().item()) if num_generations > 0 else 0
            gen_tokens = int(seq_mask.sum().item()) - prompt_tok if num_generations > 0 else 0

            # Shard sequences to per-rank for log-prob computation and training.
            # FSDP2 does gradient all-reduce; each rank only needs its own shard.
            if self._is_distributed and self._world_size > 1:
                _total = sequences.shape[0]
                _chunk = max(1, _total // self._world_size)
                _s = self._rank * _chunk
                _e = _s + _chunk if self._rank < self._world_size - 1 else _total
                sequences = sequences[_s:_e]
                seq_mask = seq_mask[_s:_e]
                if isinstance(prompt_lengths, list):
                    prompt_lengths = prompt_lengths[_s:_e]
                # ground_truths is prompt-level (N_prompts), not sequence-level
                _n_prompts = len(ground_truths) if isinstance(ground_truths, list) else 0
                if _n_prompts > 0:
                    _p_chunk = max(1, _n_prompts // self._world_size)
                    _ps = self._rank * _p_chunk
                    _pe = _ps + _p_chunk if self._rank < self._world_size - 1 else _n_prompts
                    ground_truths = ground_truths[_ps:_pe]

            self._actor_model.eval()
            old_log_probs = self._compute_log_probs_for_model(
                self._actor_model, sequences, seq_mask,
            )

            if step < 3 and self._rank == 0:
                logger.info(
                    "NaN-DEBUG [step=%d post-rollout] old_log_probs: shape=%s nan=%d inf=%d "
                    "min=%.4f max=%.4f mean=%.4f",
                    step, list(old_log_probs.shape),
                    old_log_probs.isnan().sum().item(),
                    old_log_probs.isinf().sum().item(),
                    old_log_probs[~old_log_probs.isnan()].min().item() if not old_log_probs.isnan().all() else float("nan"),
                    old_log_probs[~old_log_probs.isnan()].max().item() if not old_log_probs.isnan().all() else float("nan"),
                    old_log_probs[~old_log_probs.isnan()].mean().item() if not old_log_probs.isnan().all() else float("nan"),
                )
                logger.info(
                    "NaN-DEBUG [step=%d post-rollout] sequences: shape=%s, seq_mask: shape=%s, "
                    "seq_mask sum=%d, prompt_lengths=%s",
                    step, list(sequences.shape), list(seq_mask.shape),
                    seq_mask.sum().item(), prompt_lengths[:4],
                )

            t1 = time.time()
            if self._ref_model is not None:
                ref_log_probs = self._compute_log_probs_for_model(
                    self._ref_model, sequences, seq_mask, move_to_gpu=self._ref_on_cpu,
                )
            else:
                ref_log_probs = torch.zeros_like(old_log_probs)
            ref_time = time.time() - t1

            rewards, responses = self._compute_rewards(
                sequences, prompt_lengths, ground_truths, num_generations,
            )

            response_mask = self._build_response_mask(sequences, seq_mask, prompt_lengths)
            response_lengths = [
                int(response_mask[i].sum().item()) for i in range(response_mask.shape[0])
            ]

            batch = DataProto(
                tensors={
                    "input_ids": sequences,
                    "attention_mask": seq_mask,
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "rewards": rewards,
                    "response_mask": response_mask,
                },
                meta={
                    "algorithm": self.config.algorithm.name,
                    "response_lengths": response_lengths,
                    "responses": responses,
                    "ground_truths": ground_truths * num_generations,
                },
            )

            # --- Critic: compute values (needed for GAE) ---
            if self._critic_worker is not None:
                values_out = self._critic_worker.compute_values(batch)
                batch.tensors["values"] = values_out.tensors["values"]

            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            if self._use_atom and self._atom_engine is not None:
                self._reload_optimizer_to_gpu()

            self._log_gpu_mem("pre_train", step)
            t2 = time.time()
            micro_bs = max(1, int(self.config.policy.train_micro_batch_size))
            max_tok = int(self.config.policy.max_token_len_per_gpu)
            mini_batches = self._dynamic_mini_batches(batch, max_tok, micro_bs)

            # FSDP2: all ranks MUST run the same number of forward/backward passes.
            # Packing by actual length can yield different counts per rank.
            # Synchronize by padding ranks with fewer mini-batches.
            if self._is_distributed and self._world_size > 1:
                import torch.distributed as dist
                my_count = len(mini_batches)
                count_t = torch.tensor([my_count], device=self._device)
                dist.all_reduce(count_t, op=dist.ReduceOp.MAX)
                global_max = int(count_t.item())
                if my_count < global_max:
                    # Pad with dummy mini-batches: reuse last batch but zero response_mask
                    # so loss = 0 while FSDP2 collectives still execute.
                    pad_batch = mini_batches[-1]
                    dummy_tensors = {k: v.clone() for k, v in pad_batch.tensors.items()}
                    dummy_tensors["response_mask"] = torch.zeros_like(dummy_tensors["response_mask"])
                    dummy_mb = DataProto(tensors=dummy_tensors, meta=pad_batch.meta.copy())
                    while len(mini_batches) < global_max:
                        mini_batches.append(dummy_mb)
                    logger.info("[rank %d] Padded mini-batches: %d -> %d for FSDP2 sync",
                                self._rank, my_count, global_max)

            metrics_accum: dict[str, float] = {}
            step_count = 0
            nan_mb_count = 0
            grad_norm = 0.0
            mismatch_kl_initial: float | None = None  # first real mini-batch's mismatch KL
            nan_param_count = 0
            total_param_count = 0
            optimizer_steps = 0

            # Each packed mini-batch is one optimizer step (Verl alignment).
            accum_steps = 1
            _dp_size = self._world_size if self._is_distributed else 1
            # LR warmup (Verl: lr_warmup_steps=10)
            self._update_lr(step)
            _cur_lr = self._optimizer.param_groups[0]["lr"]
            if self._rank == 0:
                logger.info(
                    "[step %d] Training: %d mini-batches, accum_steps=%d, dp_size=%d, lr=%.2e",
                    step, len(mini_batches), accum_steps, _dp_size, _cur_lr,
                )

            # FSDP2 gradient accumulation: disable reduce-scatter on intermediate
            # micro-batches, re-enable on the last micro-batch of each group.
            _fsdp_grad_sync = accum_steps > 1 and self._is_distributed
            if _fsdp_grad_sync:
                from lumenrl.engine.training.fsdp_backend import set_requires_gradient_sync

            _do_engine_step = self._engine is not None

            for i, mini in enumerate(mini_batches):
                if i % accum_steps == 0:
                    if _do_engine_step:
                        self._engine.optimizer_zero_grad()
                    else:
                        self._optimizer.zero_grad(set_to_none=True)
                # Use correct scale for partial final group
                group_start = (i // accum_steps) * accum_steps
                group_size = min(accum_steps, len(mini_batches) - group_start)
                cur_loss_scale = 1.0 / group_size
                # Toggle FSDP2 gradient sync: off for intermediate, on for last in group
                is_last_in_group = (i + 1) % accum_steps == 0 or i == len(mini_batches) - 1
                if _fsdp_grad_sync:
                    set_requires_gradient_sync(self._actor_model, is_last_in_group)
                m = self._train_step(
                    mini, loss_scale=cur_loss_scale, dp_size=_dp_size,
                )
                if m.get("loss") is not None and (m["loss"] != m["loss"]):
                    nan_mb_count += 1
                    for k, v in m.items():
                        if v == v:
                            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                    step_count += 1
                    if (i + 1) % accum_steps == 0 or i == len(mini_batches) - 1:
                        if _do_engine_step:
                            _gn = self._engine.optimizer_step()
                        else:
                            _gn = float(torch.nn.utils.clip_grad_norm_(
                                self._actor_model.parameters(), max_norm=1.0,
                            ))
                            if not torch.isfinite(torch.tensor(_gn)):
                                self._optimizer.zero_grad(set_to_none=True)
                            else:
                                self._optimizer.step()
                        grad_norm = max(grad_norm, _gn)
                        optimizer_steps += 1
                    continue
                # Clean NaN grads before stepping
                _nan_cnt = 0
                _total_cnt = 0
                for p in self._actor_model.parameters():
                    if p.grad is not None:
                        _total_cnt += 1
                        if p.grad.isnan().any():
                            _nan_cnt += 1
                            p.grad = torch.where(
                                p.grad.isnan(), torch.zeros_like(p.grad), p.grad,
                            )
                nan_param_count = max(nan_param_count, _nan_cnt)
                total_param_count = _total_cnt
                # Only clip + step at accumulation boundaries
                if (i + 1) % accum_steps == 0 or i == len(mini_batches) - 1:
                    if _do_engine_step:
                        _gn = self._engine.optimizer_step()
                    else:
                        _gn = float(torch.nn.utils.clip_grad_norm_(
                            self._actor_model.parameters(), max_norm=1.0,
                        ))
                        if not torch.isfinite(torch.tensor(_gn)):
                            self._optimizer.zero_grad(set_to_none=True)
                        else:
                            self._optimizer.step()
                    grad_norm = max(grad_norm, _gn)
                    optimizer_steps += 1
                if mismatch_kl_initial is None and "mismatch_kl" in m:
                    mismatch_kl_initial = m["mismatch_kl"]
                for k, v in m.items():
                    if v == v:
                        metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                step_count += 1
            if _do_engine_step:
                self._engine.optimizer_zero_grad()
                _cur_lr = self._engine.lr_scheduler_step()
            else:
                self._optimizer.zero_grad(set_to_none=True)
            # Restore gradient sync after accumulation loop
            if _fsdp_grad_sync:
                set_requires_gradient_sync(self._actor_model, True)
            if self._rank == 0:
                logger.info("[step %d] Completed %d optimizer steps (from %d mini-batches).",
                            step, optimizer_steps, len(mini_batches))
            train_time = time.time() - t2
            self._log_gpu_mem("post_train", step)

            # --- Critic: update value network ---
            if self._critic_worker is not None:
                for _critic_epoch in range(getattr(self.config.critic, 'num_critic_epochs', 1)):
                    critic_metrics = self._critic_worker.train_step(batch)
                metrics_accum.update(critic_metrics)

            if nan_param_count > 0 and self._rank == 0:
                logger.warning(
                    "[step %d] Zeroed NaN grads in %d/%d params, grad_norm=%.4f",
                    step, nan_param_count, total_param_count, float(grad_norm),
                )

            denom = max(1, step_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}
            metrics["grad_norm"] = float(grad_norm)
            metrics["nan_params"] = nan_param_count
            if mismatch_kl_initial is not None:
                metrics["mismatch_kl"] = mismatch_kl_initial

            step_time = time.time() - step_start
            metrics["timing/step_s"] = step_time
            metrics["timing/gen_s"] = gen_time
            metrics["timing/ref_s"] = ref_time
            metrics["timing/train_s"] = train_time
            if gen_tokens > 0 and gen_time > 0:
                metrics["throughput/gen_tok_per_s"] = gen_tokens / gen_time
            metrics["reward/mean"] = float(rewards.mean().item())
            metrics["reward/accuracy"] = float(
                sum(1 for r in rewards if r > 0) / max(1, len(rewards))
            )
            metrics["seq/max_len"] = int(sequences.shape[1])
            metrics["seq/mean_response_len"] = float(
                sum(response_lengths) / max(1, len(response_lengths))
            )

            if self._is_distributed:
                for k in list(metrics.keys()):
                    t = torch.tensor(metrics[k], dtype=torch.float64, device=self._device)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    metrics[k] = float(t.item())

            # Validation
            val_steps = getattr(self.config, 'val_steps', 0)
            if val_steps > 0 and (step + 1) % val_steps == 0:
                val_metrics = self.run_validation()
                metrics.update(val_metrics)

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            t_sync = time.time()
            self._sync_rollout_weights()
            sync_time = time.time() - t_sync
            if sync_time > 1.0:
                metrics["timing/weight_sync_s"] = sync_time

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            self._maybe_stop_profile(step)

            del sequences, seq_mask, old_log_probs, ref_log_probs
            del rewards, responses, response_mask, batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info("[rank %d] RLTrainer.train finished after %d steps.", self._rank, total_steps)

    def _compute_log_probs_with_worker_group(
        self,
        wg: RayWorkerGroup,
        sequences: torch.Tensor,
        role: str,
    ) -> torch.Tensor:
        req = DataProto(tensors={"input_ids": sequences.detach().cpu()}, meta={})
        role_cfg = self.config.controller.ray.actor if role == "actor" else self.config.controller.ray.ref
        out = wg.dispatch_and_call(
            "compute_log_probs",
            req,
            mode=role_cfg.dispatch_mode,
            mesh_mapping=role_cfg.mesh_mapping,
            lazy_key=role_cfg.lazy_dispatch_key,
        )
        if "log_probs" in out.tensors:
            return out["log_probs"].to(self._device)
        if "ref_log_probs" in out.tensors:
            return out["ref_log_probs"].to(self._device)
        raise KeyError(f"Expected log_probs/ref_log_probs in worker output, got keys={list(out.tensors.keys())}")

    def _update_actor_with_ray(self, batch: DataProto) -> dict[str, float]:
        if self._actor_wg is None:
            raise RuntimeError("Ray actor worker group is not initialized.")
        actor_role_cfg = self.config.controller.ray.actor
        chunks = dispatch_proto(
            batch,
            self._actor_wg.num_workers,
            mode=actor_role_cfg.dispatch_mode,
            mesh_mapping=actor_role_cfg.mesh_mapping,
            lazy_state=self._ray_dispatch_state,
            lazy_key=actor_role_cfg.lazy_dispatch_key,
        )
        if not chunks:
            return {"loss": 0.0}

        if len(chunks) == 1:
            worker_and_chunks = [(0, chunks[0])]
        elif len(chunks) == self._actor_wg.num_workers:
            worker_and_chunks = list(enumerate(chunks))
        else:
            raise ValueError(
                f"actor dispatch produced {len(chunks)} chunks for "
                f"{self._actor_wg.num_workers} workers; expected 1 (rank-zero) "
                "or num_workers."
            )

        outputs: list[dict[str, float]] = []
        for i, chunk in worker_and_chunks:
            if chunk.batch_size == 0:
                continue
            outputs.append(self._actor_wg.call_single(i, "train_step", chunk))
        if not outputs:
            return {"loss": 0.0}
        merged: dict[str, float] = {}
        keys = set().union(*(o.keys() for o in outputs))
        for key in keys:
            vals = [float(o[key]) for o in outputs if key in o]
            merged[key] = float(sum(vals) / max(1, len(vals)))
        return merged

    def _train_with_ray_controller(self) -> None:
        """Ray worker orchestration path (no torch.distributed collectives)."""
        if self._algorithm is None or self._actor_wg is None:
            raise RuntimeError("Call setup() before train().")
        if self._atom_engine is None:
            raise RuntimeError("Ray controller path currently requires ATOM rollout.")

        for cb in self.callbacks:
            cb.on_train_begin(self)

        num_generations = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)
        start_step = self._resume_step

        for step in range(start_step, total_steps):
            step_start = time.time()
            self.global_step = step
            self._maybe_start_profile(step)
            for cb in self.callbacks:
                cb.on_step_begin(self, step)

            prompts, ground_truths = self._get_batch_prompts(step)
            input_ids, attention_mask = self._tokenize_prompts(prompts)

            gen_t0 = time.time()
            sequences, seq_mask, prompt_lengths = self._rollout_with_atom(prompts, num_generations)
            gen_time = time.time() - gen_t0

            old_log_probs = self._compute_log_probs_with_worker_group(self._actor_wg, sequences, role="actor")
            if self._ref_wg is not None:
                ref_log_probs = self._compute_log_probs_with_worker_group(self._ref_wg, sequences, role="ref")
            else:
                ref_log_probs = torch.zeros_like(old_log_probs)

            ref_time = max(0.0, time.time() - (gen_t0 + gen_time))
            rewards, responses = self._compute_rewards(
                sequences, prompt_lengths, ground_truths, num_generations,
            )
            response_mask = self._build_response_mask(sequences, seq_mask, prompt_lengths)
            response_lengths = [int(response_mask[i].sum().item()) for i in range(response_mask.shape[0])]

            batch = DataProto(
                tensors={
                    "input_ids": sequences,
                    "attention_mask": seq_mask,
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "rewards": rewards,
                    "response_mask": response_mask,
                },
                meta={
                    "algorithm": self.config.algorithm.name,
                    "response_lengths": response_lengths,
                    "responses": responses,
                    "ground_truths": ground_truths * num_generations,
                    "algo_config": self._to_plain_dict(self.config.algorithm),
                },
            )

            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            train_t0 = time.time()
            micro_bs = max(1, int(self.config.policy.train_micro_batch_size))
            max_tok = int(self.config.policy.max_token_len_per_gpu)
            mini_batches = self._dynamic_mini_batches(batch, max_tok, micro_bs)

            metrics_accum: dict[str, float] = {}
            for mini in mini_batches:
                m = self._update_actor_with_ray(mini)
                for k, v in m.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
            train_time = time.time() - train_t0
            denom = max(1, len(mini_batches))
            metrics = {k: v / denom for k, v in metrics_accum.items()}

            prompt_tok = int(attention_mask.repeat_interleave(num_generations, dim=0).to(seq_mask.device).sum().item())
            gen_tokens = int(seq_mask.sum().item()) - prompt_tok
            metrics["timing/step_s"] = time.time() - step_start
            metrics["timing/gen_s"] = gen_time
            metrics["timing/ref_s"] = ref_time
            metrics["timing/train_s"] = train_time
            if gen_tokens > 0 and gen_time > 0:
                metrics["throughput/gen_tok_per_s"] = gen_tokens / gen_time
            metrics["reward/mean"] = float(rewards.mean().item())
            metrics["reward/accuracy"] = float(sum(1 for r in rewards if r > 0) / max(1, len(rewards)))
            metrics["seq/max_len"] = int(sequences.shape[1])
            metrics["seq/mean_response_len"] = float(sum(response_lengths) / max(1, len(response_lengths)))

            # Validation
            val_steps = getattr(self.config, 'val_steps', 0)
            if val_steps > 0 and (step + 1) % val_steps == 0:
                val_metrics = self.run_validation()
                metrics.update(val_metrics)

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            t_sync = time.time()
            self._sync_rollout_weights()
            sync_time = time.time() - t_sync
            if sync_time > 1.0:
                metrics["timing/weight_sync_s"] = sync_time

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            self._maybe_stop_profile(step)

            del sequences, seq_mask, old_log_probs, ref_log_probs
            del rewards, responses, response_mask, batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for cb in self.callbacks:
            cb.on_train_end(self)
        logger.info("RLTrainer.train (ray-controller) finished after %d steps.", total_steps)

    def _sync_rollout_weights(self) -> None:
        """Sync actor weights to ATOM rollout engine if configured."""
        if self._use_atom and self._atom_engine is not None:
            self._sync_weights_to_atom()

    def run_validation(self) -> dict[str, float]:
        """Run validation: generate responses, compute rewards, aggregate metrics."""
        if self._val_dataset is None or len(self._val_dataset) == 0:
            return {}

        val_bs = getattr(self.config, 'val_batch_size', 16)
        num_samples = len(self._val_dataset)
        all_scores: list[float] = []
        all_response_lengths: list[int] = []
        all_responses: list[str] = []

        # Iterate through validation dataset in batches
        for start in range(0, num_samples, val_bs):
            end = min(start + val_bs, num_samples)
            indices = list(range(start, end))
            samples = [self._val_dataset[idx] for idx in indices]

            # Extract prompts and ground truths (same logic as _get_batch_prompts)
            import json as _json
            prompts: list[str] = []
            ground_truths: list[str] = []
            for s in samples:
                raw = s.get("prompt") or s.get("question") or s.get("input") or ""
                if isinstance(raw, list):
                    text = "\n".join(m.get("content", "") for m in raw if isinstance(m, dict))
                elif isinstance(raw, str) and raw.startswith("["):
                    try:
                        msgs = _json.loads(raw)
                        text = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
                    except (_json.JSONDecodeError, TypeError):
                        text = raw
                else:
                    text = str(raw)

                if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
                    orig = s.get("prompt") or s.get("question") or s.get("input") or ""
                    if isinstance(orig, list):
                        try:
                            text = self._tokenizer.apply_chat_template(orig, tokenize=False, add_generation_prompt=True)
                        except Exception:
                            pass

                prompts.append(text)
                gt = s.get("answer") or s.get("ground_truth") or s.get("solution") or ""
                ground_truths.append(str(gt))

            input_ids, attention_mask = self._tokenize_prompts(prompts)

            # Generate with greedy decoding for reproducibility
            if self._use_atom and self._atom_engine is not None:
                sequences, seq_mask, prompt_lengths = self._rollout_with_atom(prompts, num_generations=1)
            else:
                sequences, seq_mask, prompt_lengths = self._rollout_phase(input_ids, attention_mask, num_generations=1)

            # Compute rewards
            rewards, responses = self._compute_rewards(sequences, prompt_lengths, ground_truths, num_generations=1)

            # Collect results
            if rewards.dim() > 1:
                scores = rewards.squeeze(-1).tolist()
            else:
                scores = rewards.tolist()
            all_scores.extend(scores)
            all_responses.extend(responses)

            # Response lengths
            response_mask = self._build_response_mask(sequences, seq_mask, prompt_lengths)
            lengths = response_mask.sum(dim=-1).tolist()
            all_response_lengths.extend([int(x) for x in lengths])

        if not all_scores:
            return {}

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        lengths_t = torch.tensor(all_response_lengths, dtype=torch.float32)

        metrics: dict[str, float] = {
            "val/score_mean": float(scores_t.mean()),
            "val/score_max": float(scores_t.max()),
            "val/score_min": float(scores_t.min()),
            "val/score_std": float(scores_t.std()) if len(all_scores) > 1 else 0.0,
            "val/response_length_mean": float(lengths_t.mean()),
            "val/num_samples": float(len(all_scores)),
        }

        # Print sample responses (rank 0 only)
        if self._rank == 0:
            num_print = min(getattr(self.config.logger, 'num_val_samples_to_print', 5), len(all_responses))
            for i in range(num_print):
                logger.info(
                    "Val sample %d: score=%.3f len=%d response=%s",
                    i, all_scores[i], all_response_lengths[i],
                    all_responses[i][:200],
                )

        return metrics

    def cleanup(self) -> None:
        """Release all resources."""
        if self._profiler is not None:
            # Best-effort stop in case an exception interrupted the loop.
            try:
                self._profiler.stop()
            except Exception:
                pass
            self._profiler = None
        if self._actor_wg is not None:
            self._actor_wg.stop()
            self._actor_wg = None
        if self._ref_wg is not None:
            self._ref_wg.stop()
            self._ref_wg = None
        if self._ray_cluster is not None:
            self._ray_cluster.shutdown()
            self._ray_cluster = None
        if self._critic_worker is not None:
            self._critic_worker.cleanup()
            self._critic_worker = None
        if self._atom_engine is not None:
            self._atom_engine.shutdown()
            self._atom_engine = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        if self._actor_model is not None:
            del self._actor_model
        if self._ref_model is not None:
            del self._ref_model
        self._actor_model = None
        self._ref_model = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] RLTrainer.cleanup complete.", self._rank)


__all__ = ["RLTrainer"]
