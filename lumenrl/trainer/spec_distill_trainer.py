"""Speculative Decoding Draft Model Distillation trainer.

Implements the off-policy distillation loop for training
Eagle3 / DFlash draft models:

1. Sample prompts from the dataset.
2. Teacher forward (no grad) -- produce hidden states on dataset sequences.
3. Draft model forward (with grad) -- predict next tokens from teacher hiddens.
4. Loss: cross-entropy / forward KL with position-dependent decay weighting.

Unlike OPD, there is **no student rollout**: the teacher processes dataset
sequences directly, and the draft model learns to match/predict from the
teacher's internal representations.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import time
from typing import Any

import torch
import torch.nn.functional as F

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer.callbacks import Callback, LoggingCallback

logger = logging.getLogger(__name__)


_PREFETCH_SENTINEL = object()


class _TeacherPrefetcher:
    """Prefetch teacher inference in a single background worker thread.

    All Mooncake receives happen on **CPU** to avoid concurrent GPU access
    from the prefetch thread and the training thread (which causes
    VM_L2_PROTECTION_FAULT on ROCm/MI350).  The main thread moves tensors
    to GPU when it calls ``get()``.
    """

    def __init__(self, trainer: "SpecDistillTrainer") -> None:
        self._trainer = trainer
        self._device = trainer._device
        self._req_queue: queue.Queue[int | object] = queue.Queue()
        self._res_queue: queue.Queue[tuple | BaseException] = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while True:
            step = self._req_queue.get()
            if step is _PREFETCH_SENTINEL:
                break
            try:
                ids, mask = self._trainer._get_batch_sequences(step)
                data = self._trainer._teacher_inference_rank0(
                    ids, mask, recv_device=torch.device("cpu"),
                )
                self._res_queue.put((data, ids, mask))
            except Exception as e:
                self._res_queue.put(e)

    def prefetch(self, step: int) -> None:
        """Submit a step for background processing (non-blocking)."""
        self._req_queue.put(step)

    def get(self) -> tuple:
        """Block until the next result is ready; move tensors to GPU."""
        item = self._res_queue.get()
        if isinstance(item, BaseException):
            raise item
        data, ids, mask = item
        if data is not None:
            data = {k: v.to(self._device, non_blocking=True) for k, v in data.items()}
        return data, ids, mask

    def stop(self) -> None:
        self._req_queue.put(_PREFETCH_SENTINEL)
        self._worker.join(timeout=5)


class SpecDistillTrainer:
    """Off-policy draft model distillation trainer.

    Lifecycle: ``setup() -> train() -> cleanup()``.
    """

    def __init__(self, config: LumenRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []

        self._teacher_model: torch.nn.Module | None = None
        self._teacher_engine: Any | None = None  # AtomTeacherEngine, SglangTeacherEngine, or VllmTeacherEngine
        self._mooncake_master: Any | None = None
        self._draft_model: torch.nn.Module | None = None
        self._lm_head_weight: torch.Tensor | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None

        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = (
            torch.distributed.get_rank() if self._is_distributed else 0
        )
        self._world_size: int = (
            torch.distributed.get_world_size() if self._is_distributed else 1
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize teacher, draft model, optimizer, and dataset."""
        teacher_cfg = self.config.algorithm.teacher
        spec_cfg = self.config.algorithm.spec_distill
        draft_cfg = self.config.algorithm.draft

        teacher_name = teacher_cfg.model_name or self.config.policy.model_name
        if not teacher_name:
            raise ValueError(
                "Teacher model name required via algorithm.teacher.model_name "
                "or policy.model_name."
            )

        # ---- Teacher model (frozen) ----
        if teacher_cfg.inference_backend == "sglang":
            self._setup_teacher_sglang(teacher_cfg, teacher_name)
        elif teacher_cfg.inference_backend == "vllm":
            self._setup_teacher_vllm(teacher_cfg, teacher_name)
        elif teacher_cfg.inference_backend == "atom":
            self._setup_teacher_atom(teacher_cfg, teacher_name)
        else:
            self._setup_teacher_hf(teacher_name)

        # Determine hidden dim from teacher
        teacher_hidden_dim = self._lm_head_weight.shape[1]

        # ---- Draft model ----
        draft_type = spec_cfg.draft_type.lower()
        logger.info(
            "[rank %d] Building draft model: type=%s, hidden_dim=%d",
            self._rank,
            draft_type,
            teacher_hidden_dim,
        )

        if draft_type == "eagle3":
            from lumenrl.models.eagle3 import Eagle3Model

            num_layers = draft_cfg.num_layers or 1
            spec_length = self.config.algorithm.spec_distill.spec_length
            num_heads = getattr(draft_cfg, "num_heads", None) or max(1, teacher_hidden_dim // 128)
            ffn_dim = getattr(draft_cfg, "ffn_dim", None)
            head_dim = getattr(draft_cfg, "head_dim", None) or 128
            rms_norm_eps = getattr(draft_cfg, "rms_norm_eps", 1e-6)
            rope_theta = getattr(draft_cfg, "rope_theta", 1000000.0)
            num_kv_heads = getattr(draft_cfg, "num_kv_heads", None)

            rope_scaling = None
            rope_scaling_type = getattr(draft_cfg, "rope_scaling_type", None)
            if rope_scaling_type:
                rope_scaling = {
                    "type": rope_scaling_type,
                    "factor": getattr(draft_cfg, "rope_scaling_factor", 64.0),
                    "original_max_position_embeddings": getattr(draft_cfg, "rope_original_max_pos", 4096),
                    "beta_fast": getattr(draft_cfg, "rope_beta_fast", 32.0),
                    "beta_slow": getattr(draft_cfg, "rope_beta_slow", 1.0),
                    "mscale": getattr(draft_cfg, "rope_mscale", 1.0),
                    "mscale_all_dim": getattr(draft_cfg, "rope_mscale_all_dim", 1.0),
                }

            teacher_vocab_size = self._lm_head_weight.shape[0]
            self._draft_model = Eagle3Model(
                hidden_dim=teacher_hidden_dim,
                vocab_size=teacher_vocab_size,
                num_heads=num_heads,
                num_layers=num_layers,
                length=spec_length,
                ffn_dim=ffn_dim,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                num_kv_heads=num_kv_heads,
                rope_scaling=rope_scaling,
            )
        elif draft_type == "dflash":
            from lumenrl.models.dflash import DFlashModel

            num_layers = draft_cfg.num_layers or 2
            self._draft_model = DFlashModel(
                hidden_dim=teacher_hidden_dim,
                num_target_layers=spec_cfg.num_target_layers,
                num_heads=max(1, teacher_hidden_dim // 128),
                num_layers=num_layers,
                block_size=8,
            )
        else:
            raise ValueError(
                f"Unknown draft_type: {draft_type!r}. Use 'eagle3' or 'dflash'."
            )

        resume_from = getattr(draft_cfg, "resume_from", None)
        if resume_from and not getattr(draft_cfg, "from_scratch", False):
            import glob as _glob
            ckpt_files = sorted(_glob.glob(os.path.join(resume_from, "checkpoint_*.pt")))
            if ckpt_files:
                ckpt_path = ckpt_files[-1]
                logger.info("[rank %d] Loading draft weights from %s", self._rank, ckpt_path)
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = payload.get("model_state_dict", payload)
                self._draft_model.load_state_dict(sd, strict=False)
            else:
                logger.warning("[rank %d] resume_from=%s but no checkpoints found", self._rank, resume_from)

        draft_dtype_str = getattr(draft_cfg, "dtype", "bfloat16")
        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        draft_dtype = _dtype_map.get(draft_dtype_str, torch.bfloat16)
        self._draft_model.to(device=self._device, dtype=draft_dtype)
        self._grad_scaler = None

        # Apply Lumen AITER kernel optimizations before FSDP2 sharding
        training_quant = {}
        if hasattr(self.config, "quantization") and hasattr(self.config.quantization, "training"):
            tq = self.config.quantization.training
            if tq is not None:
                training_quant = tq if isinstance(tq, dict) else {
                    k: getattr(tq, k, None)
                    for k in ["fp8", "lumen_norm", "lumen_linear",
                              "hf_attn_patch", "fp8_param_manager"]
                }
        lumen_norm = training_quant.get("lumen_norm", False)
        lumen_linear = training_quant.get("lumen_linear", False)
        hf_attn_patch = training_quant.get("hf_attn_patch", False)
        if lumen_norm or lumen_linear or hf_attn_patch:
            try:
                from lumen.config import LumenConfig
                lumen_cfg = LumenConfig(
                    lumen_norm=bool(lumen_norm),
                    lumen_linear=bool(lumen_linear),
                    hf_attn_patch=bool(hf_attn_patch),
                    scaling="none",
                )
                _manager, self._draft_model = lumen_cfg.enable(self._draft_model)
            except ImportError:
                logger.warning("Lumen not available; using default kernels")

        if self._is_distributed:
            from lumenrl.engine.training.fsdp_backend import FSDP2Backend

            self._draft_model = FSDP2Backend.apply_fsdp2(
                self._draft_model, {"enabled": True}
            )

        self._optimizer = torch.optim.AdamW(
            [p for p in self._draft_model.parameters() if p.requires_grad],
            lr=self.config.policy.learning_rate,
            weight_decay=self.config.policy.weight_decay,
        )

        total_steps = self.config.num_training_steps
        warmup_ratio = self.config.policy.warmup_ratio
        warmup_steps = int(total_steps * warmup_ratio)
        if warmup_steps > 0 or warmup_ratio > 0:
            from torch.optim.lr_scheduler import LambdaLR
            import math as _math

            def _cosine_with_warmup(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / max(1, warmup_steps)
                progress = (current_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                return max(0.0, 0.5 * (1.0 + _math.cos(_math.pi * progress)))

            self._scheduler = LambdaLR(self._optimizer, _cosine_with_warmup)
        else:
            self._scheduler = None

        # ---- Tokenizer ----
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            teacher_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        # ---- Dataset ----
        self._load_dataset()

        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info(
            "[rank %d] SpecDistillTrainer.setup complete: teacher=%s, draft=%s",
            self._rank,
            teacher_name,
            draft_type,
        )

    def _setup_teacher_hf(self, teacher_name: str) -> None:
        """Load teacher model via HuggingFace (original path)."""
        from transformers import AutoModelForCausalLM

        logger.info("[rank %d] Loading teacher model (HF): %s", self._rank, teacher_name)
        self._teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._teacher_model.eval()
        for p in self._teacher_model.parameters():
            p.requires_grad_(False)
        self._teacher_model.to(self._device)

        lm_head_weight = None
        for name, param in self._teacher_model.named_parameters():
            if "lm_head" in name and "weight" in name:
                lm_head_weight = param.detach().clone()
                break
        if lm_head_weight is None:
            raise RuntimeError("Could not find lm_head.weight in teacher model.")
        self._lm_head_weight = lm_head_weight

    def _setup_teacher_atom(self, teacher_cfg: Any, teacher_name: str) -> None:
        """Load teacher model via ATOM engine (subprocess, TP, quantization).

        Hidden states are transferred via MORI-IO P2P RDMA (GPU Direct).
        """
        from lumenrl.engine.inference.atom_teacher_engine import AtomTeacherEngine

        gpu_ids = teacher_cfg.gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        logger.info(
            "[rank %d] Loading teacher model (ATOM): %s (tp=%d, gpus=%s)",
            self._rank, teacher_name,
            teacher_cfg.tensor_parallel_size, gpu_ids,
        )
        # Only rank 0 starts the ATOM subprocess
        if self._rank == 0:
            max_bs = max(1, int(self.config.policy.train_global_batch_size))
            max_seq = int(self.config.policy.max_total_sequence_length)
            self._teacher_engine = AtomTeacherEngine(
                model_name=teacher_name,
                tensor_parallel_size=teacher_cfg.tensor_parallel_size,
                gpu_ids=gpu_ids,
                mori_io_host=getattr(teacher_cfg, "mori_io_host", "127.0.0.1"),
                mori_io_port=getattr(teacher_cfg, "mori_io_port", 0),
                mori_io_qp_per_transfer=getattr(teacher_cfg, "mori_io_qp_per_transfer", 2),
                max_batch_size=max_bs,
                max_seq_len=max_seq,
                local_device=self._device,
            )
            self._teacher_engine.start()
            self._lm_head_weight = self._teacher_engine.get_lm_head_weight()
        else:
            self._teacher_engine = None
            self._lm_head_weight = None

        # Broadcast lm_head_weight to all ranks
        if self._is_distributed:
            if self._rank == 0:
                shape = torch.tensor(
                    list(self._lm_head_weight.shape),
                    dtype=torch.long, device=self._device,
                )
            else:
                shape = torch.zeros(2, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(shape, src=0)

            if self._rank == 0:
                weight = self._lm_head_weight.to(self._device)
            else:
                weight = torch.zeros(
                    shape[0].item(), shape[1].item(),
                    dtype=torch.bfloat16, device=self._device,
                )
            torch.distributed.broadcast(weight, src=0)
            self._lm_head_weight = weight.cpu()

    def _setup_teacher_sglang(self, teacher_cfg: Any, teacher_name: str) -> None:
        """Load teacher via SGLang Engine + ATOM plugin + Mooncake RDMA."""
        from lumenrl.engine.inference.sglang_teacher_engine import SglangTeacherEngine

        gpu_ids = teacher_cfg.gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        atom_plugin = getattr(teacher_cfg, "atom_plugin", False)
        logger.info(
            "[rank %d] Loading teacher (SGLang+ATOM): %s (tp=%d, gpus=%s, atom=%s)",
            self._rank, teacher_name,
            teacher_cfg.tensor_parallel_size, gpu_ids, atom_plugin,
        )

        mc_config = getattr(self.config, "mooncake", None)

        # Launch Mooncake master if needed (rank 0 only)
        if self._rank == 0 and mc_config is not None:
            if not mc_config.master_server_address:
                from lumenrl.transfer.mooncake_master import launch_mooncake_master
                self._mooncake_master = launch_mooncake_master(mc_config)
                logger.info(
                    "Mooncake master started: %s", mc_config.master_server_address
                )

        # Only rank 0 starts the subprocess
        if self._rank == 0:
            self._teacher_engine = SglangTeacherEngine(
                model_name=teacher_name,
                gpu_ids=gpu_ids,
                mooncake_config=mc_config,
                tensor_parallel_size=teacher_cfg.tensor_parallel_size,
                atom_plugin=atom_plugin,
                quantization=getattr(teacher_cfg, "quantization", ""),
                max_batch_size=max(1, int(self.config.policy.train_global_batch_size)),
                max_seq_len=int(self.config.policy.max_total_sequence_length),
                local_device=self._device,
            )
            self._teacher_engine.start()
            self._lm_head_weight = self._teacher_engine.get_lm_head_weight()
        else:
            self._teacher_engine = None
            self._lm_head_weight = None

        # Broadcast lm_head_weight to all ranks
        if self._is_distributed:
            if self._rank == 0:
                shape = torch.tensor(
                    list(self._lm_head_weight.shape),
                    dtype=torch.long, device=self._device,
                )
            else:
                shape = torch.zeros(2, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(shape, src=0)

            if self._rank == 0:
                weight = self._lm_head_weight.to(self._device)
            else:
                weight = torch.zeros(
                    shape[0].item(), shape[1].item(),
                    dtype=torch.bfloat16, device=self._device,
                )
            torch.distributed.broadcast(weight, src=0)
            self._lm_head_weight = weight.cpu()

    def _setup_teacher_vllm(self, teacher_cfg: Any, teacher_name: str) -> None:
        """Load teacher via vLLM + ATOM plugin + Mooncake."""
        from lumenrl.engine.inference.vllm_teacher_engine import VllmTeacherEngine

        gpu_ids = teacher_cfg.gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        logger.info(
            "[rank %d] Loading teacher (vLLM+ATOM): %s (tp=%d, gpus=%s)",
            self._rank, teacher_name,
            teacher_cfg.tensor_parallel_size, gpu_ids,
        )

        mc_config = getattr(self.config, "mooncake", None)

        if self._rank == 0 and mc_config is not None:
            if not mc_config.master_server_address:
                from lumenrl.transfer.mooncake_master import launch_mooncake_master
                self._mooncake_master = launch_mooncake_master(mc_config)
                logger.info(
                    "Mooncake master started: %s", mc_config.master_server_address
                )

        if self._rank == 0:
            self._teacher_engine = VllmTeacherEngine(
                model_name=teacher_name,
                gpu_ids=gpu_ids,
                mooncake_config=mc_config,
                tensor_parallel_size=teacher_cfg.tensor_parallel_size,
                quantization=getattr(teacher_cfg, "quantization", ""),
                max_batch_size=max(1, int(self.config.policy.train_global_batch_size)),
                max_seq_len=int(self.config.policy.max_total_sequence_length),
                local_device=self._device,
            )
            self._teacher_engine.start()
            self._lm_head_weight = self._teacher_engine.get_lm_head_weight()
            self._embed_weight = self._teacher_engine.get_embed_weight()
            self._norm_weight, self._norm_eps = self._teacher_engine.get_norm_weight()
        else:
            self._teacher_engine = None
            self._lm_head_weight = None
            self._embed_weight = None
            self._norm_weight = None
            self._norm_eps = 1e-6

        if self._is_distributed:
            for attr in ("_lm_head_weight", "_embed_weight", "_norm_weight"):
                w = getattr(self, attr)
                if self._rank == 0:
                    shape = torch.tensor(
                        list(w.shape), dtype=torch.long, device=self._device,
                    )
                else:
                    shape = torch.zeros(2, dtype=torch.long, device=self._device)
                ndim_t = torch.tensor([len(w.shape) if w is not None else 0],
                                      dtype=torch.long, device=self._device)
                torch.distributed.broadcast(ndim_t, src=0)
                ndim = ndim_t.item()
                if self._rank != 0:
                    shape = torch.zeros(ndim, dtype=torch.long, device=self._device)
                else:
                    shape = torch.tensor(
                        list(w.shape), dtype=torch.long, device=self._device,
                    )
                torch.distributed.broadcast(shape, src=0)
                dims = [shape[i].item() for i in range(ndim)]
                if self._rank == 0:
                    weight = w.to(self._device)
                else:
                    weight = torch.zeros(
                        *dims,
                        dtype=torch.bfloat16, device=self._device,
                    )
                torch.distributed.broadcast(weight, src=0)
                setattr(self, attr, weight.cpu())

            eps_t = torch.tensor([self._norm_eps], dtype=torch.float64, device=self._device)
            torch.distributed.broadcast(eps_t, src=0)
            self._norm_eps = eps_t.item()

    def _teacher_inference_rank0(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recv_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Run teacher inference on rank 0 only (no NCCL). Thread-safe."""
        if self._rank != 0:
            return None
        return self._teacher_engine.extract_hidden_states(
            input_ids, attention_mask, recv_device=recv_device,
        )

    def _teacher_broadcast(
        self,
        rank0_data: dict[str, torch.Tensor] | None,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """NCCL broadcast teacher data from rank 0 to all ranks."""
        if self._rank == 0:
            hidden_states = rank0_data["hidden_states"]
            token_embeds = rank0_data["token_embeds"]
            last_hidden_states = rank0_data["last_hidden_states"]
        else:
            hidden_states = None
            token_embeds = None
            last_hidden_states = None

        if self._is_distributed:
            if self._rank == 0:
                shape = torch.tensor(
                    [hidden_states.shape[0], hidden_states.shape[1],
                     hidden_states.shape[2], token_embeds.shape[2],
                     last_hidden_states.shape[2]],
                    dtype=torch.long, device=self._device,
                )
            else:
                shape = torch.zeros(5, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(shape, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST shape OK", self._rank)

            B, T = shape[0].item(), shape[1].item()
            D_hidden, D_embed, D_last = shape[2].item(), shape[3].item(), shape[4].item()
            if self._rank == 0:
                h = hidden_states.to(self._device)
                e = token_embeds.to(self._device)
                lhs = last_hidden_states.to(self._device)
            else:
                h = torch.zeros(B, T, D_hidden, dtype=torch.bfloat16, device=self._device)
                e = torch.zeros(B, T, D_embed, dtype=torch.bfloat16, device=self._device)
                lhs = torch.zeros(B, T, D_last, dtype=torch.bfloat16, device=self._device)
            torch.distributed.broadcast(h, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST hidden_states OK (B=%d T=%d D=%d, %.1fG)", self._rank, B, T, D_hidden, h.element_size()*h.nelement()/1e9)
            torch.distributed.broadcast(e, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST token_embeds OK", self._rank)
            torch.distributed.broadcast(lhs, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST last_hidden_states OK", self._rank)
            hidden_states = h
            token_embeds = e
            last_hidden_states = lhs

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": input_ids,
            "last_hidden_states": last_hidden_states,
        }

    def _teacher_forward_vllm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via vLLM+ATOM engine (rank 0 only, then broadcast)."""
        rank0_data = self._teacher_inference_rank0(input_ids, attention_mask)
        return self._teacher_broadcast(rank0_data, input_ids)

    def _load_dataset(self) -> None:
        """Load training dataset."""
        dataset_path = self.config.reward.dataset
        if not dataset_path:
            logger.warning("No dataset configured; using synthetic prompts.")
            self._dataset = None
            return

        from datasets import load_dataset

        if os.path.isfile(dataset_path) or os.path.isdir(dataset_path):
            if dataset_path.endswith(".parquet"):
                self._dataset = load_dataset(
                    "parquet", data_files=dataset_path, split="train"
                )
            elif dataset_path.endswith((".jsonl", ".json")):
                self._dataset = load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
            else:
                self._dataset = load_dataset(dataset_path, split="train")
        else:
            self._dataset = load_dataset(dataset_path, split="train")

        self._dataset = self._dataset.shuffle(seed=self.config.seed)
        logger.info(
            "Loaded dataset: %d samples (shuffled, seed=%d) from %s",
            len(self._dataset), self.config.seed, dataset_path,
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _get_batch_sequences(
        self, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize dataset samples into (input_ids, attention_mask)."""
        import json as _json

        bs = max(1, self.config.policy.train_global_batch_size)

        if self._dataset is None:
            texts = [
                f"What is {i + step * bs} + {i + step * bs + 1}? The answer is {2 * (i + step * bs) + 1}."
                for i in range(bs)
            ]
        else:
            dataset_len = len(self._dataset)
            start = (step * bs) % dataset_len
            indices = [(start + i) % dataset_len for i in range(bs)]
            samples = [self._dataset[idx] for idx in indices]
            texts = []
            for s in samples:
                convs = s.get("conversations")
                if convs and isinstance(convs, list) and isinstance(convs[0], dict):
                    if hasattr(self._tokenizer, "apply_chat_template"):
                        ROLE_MAP = {"human": "user", "gpt": "assistant"}
                        normalized = []
                        for m in convs:
                            if "role" in m and "content" in m:
                                normalized.append(m)
                            elif "from" in m and "value" in m:
                                normalized.append({
                                    "role": ROLE_MAP.get(m["from"], m["from"]),
                                    "content": m["value"],
                                })
                            else:
                                normalized.append(m)
                        text = self._tokenizer.apply_chat_template(
                            normalized, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = "\n".join(
                            m.get("content", "") for m in convs if isinstance(m, dict)
                        )
                    texts.append(text)
                    continue
                raw = (
                    s.get("prompt") or s.get("question") or s.get("input")
                    or s.get("problem") or ""
                )
                answer = (
                    s.get("answer")
                    or s.get("solution")
                    or s.get("generated_solution")
                    or s.get("target")
                    or s.get("output")
                    or ""
                )
                if isinstance(raw, list):
                    text = "\n".join(
                        m.get("content", "") for m in raw if isinstance(m, dict)
                    )
                elif isinstance(raw, str) and raw.startswith("["):
                    try:
                        msgs = _json.loads(raw)
                        text = "\n".join(
                            m.get("content", "") for m in msgs if isinstance(m, dict)
                        )
                    except (_json.JSONDecodeError, TypeError):
                        text = raw
                else:
                    text = str(raw)
                if answer:
                    text = text + " " + str(answer)
                texts.append(text)

        max_len = self.config.policy.max_total_sequence_length
        self._tokenizer.padding_side = "right"
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len - 1,  # vLLM requires prompt + 1 < max_model_len
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # ------------------------------------------------------------------
    # Teacher forward
    # ------------------------------------------------------------------

    def _teacher_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run teacher on dataset sequences, return hidden states.

        Returns a dict with:
        - ``"hidden_states"``: 3 concatenated aux hidden states ``[B, T, 3*D]``
          from layers [1, N//2-1, N-4] (Eagle3 convention)
        - ``"token_embeds"``: input embeddings ``[B, T, D]``
        - ``"input_ids"``: the original input ids
        """
        backend = self.config.algorithm.teacher.inference_backend
        if backend == "vllm":
            return self._teacher_forward_vllm(input_ids, attention_mask)
        if backend in ("sglang", "atom"):
            return self._teacher_forward_atom(input_ids, attention_mask)
        return self._teacher_forward_hf(input_ids, attention_mask)

    def _teacher_forward_atom(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via ATOM engine (rank 0 only, then broadcast).

        Hidden states arrive on the training GPU via MORI-IO RDMA
        (GPU Direct), avoiding CPU round-trips.

        NOTE: ATOM engine currently returns only the final hidden state
        ``[B, T, D]``, not 3 aux layers.  For Eagle3 training, use the
        ``vllm`` or ``hf`` backend instead until ATOM is updated.
        """
        if self._rank == 0:
            # MORI-IO returns tensors already on the training GPU
            result = self._teacher_engine.extract_hidden_states(
                input_ids, attention_mask,
            )
            hidden_states = result["hidden_states"]  # [B, T, D] or [B, T, 3*D]
            token_embeds = result["token_embeds"]     # [B, T, D] on GPU
        else:
            hidden_states = None
            token_embeds = None

        # Broadcast from rank 0 to all training ranks
        if self._is_distributed:
            if self._rank == 0:
                shape = torch.tensor(
                    [hidden_states.shape[0], hidden_states.shape[1],
                     hidden_states.shape[2], token_embeds.shape[2]],
                    dtype=torch.long, device=self._device,
                )
            else:
                shape = torch.zeros(4, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(shape, src=0)

            B, T = shape[0].item(), shape[1].item()
            D_hidden, D_embed = shape[2].item(), shape[3].item()
            if self._rank == 0:
                h = hidden_states.to(self._device)
                e = token_embeds.to(self._device)
            else:
                h = torch.zeros(B, T, D_hidden, dtype=torch.bfloat16, device=self._device)
                e = torch.zeros(B, T, D_embed, dtype=torch.bfloat16, device=self._device)
            torch.distributed.broadcast(h, src=0)
            torch.distributed.broadcast(e, src=0)
            hidden_states = h
            token_embeds = e

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": input_ids,
        }

    def _teacher_forward_hf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via HuggingFace (original path).

        Extracts 3 aux hidden states from layers [1, N//2-1, N-4] and
        concatenates them to [B, T, 3*H].  HF ``output_hidden_states``
        returns index 0 = embedding, index i+1 = layer i output, so the
        HF indices are [2, N//2, N-3].
        """
        if self._teacher_model is None:
            raise RuntimeError("Teacher model not loaded.")

        micro_bs = max(1, self.config.algorithm.opd.teacher_micro_batch_size)

        all_hidden: list[torch.Tensor] = []
        all_embeds: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, input_ids.shape[0], micro_bs):
                end = min(start + micro_bs, input_ids.shape[0])
                ids_chunk = input_ids[start:end].to(self._device)
                mask_chunk = attention_mask[start:end].to(self._device)

                outputs = self._teacher_model(
                    input_ids=ids_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=True,
                )

                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    all_embeds.append(outputs.hidden_states[0].cpu())
                    # HF hidden_states: [0]=embed, [i+1]=layer_i output
                    # Aux layers: [1, N//2-1, N-4]
                    # HF indices: [2, N//2, N-3]
                    num_layers = len(outputs.hidden_states) - 1
                    hf_aux_ids = [2, num_layers // 2, num_layers - 3]
                    aux_concat = torch.cat(
                        [outputs.hidden_states[i].cpu() for i in hf_aux_ids],
                        dim=-1,
                    )  # [micro_B, T, 3*H]
                    all_hidden.append(aux_concat)
                else:
                    raise RuntimeError(
                        "Teacher model did not return hidden_states."
                    )

                del outputs

        return {
            "hidden_states": torch.cat(all_hidden, dim=0),
            "token_embeds": torch.cat(all_embeds, dim=0),
            "input_ids": input_ids,
        }

    # ------------------------------------------------------------------
    # Draft model train step
    # ------------------------------------------------------------------

    def _debug_sync(self, tag: str, step: int) -> None:
        """Sync GPU and log checkpoint for crash debugging."""
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_res = torch.cuda.memory_reserved() / 1e9
        logger.warning("[rank %d] DEBUG step=%d checkpoint: %s OK (%.1fG/%.1fG)", self._rank, step, tag, mem_alloc, mem_res)

    def _install_backward_hooks(self) -> None:
        """Register backward hooks on each submodule of the draft model for crash debugging."""
        if getattr(self, "_bwd_hooks_installed", False):
            return
        self._bwd_hooks_installed = True
        rank = self._rank

        def _make_hook(name):
            def hook(module, grad_input, grad_output):
                torch.cuda.synchronize()
                mem = torch.cuda.memory_allocated() / 1e9
                logger.warning("[rank %d] BWD hook: %s OK (%.1fG)", rank, name, mem)
            return hook

        model = self._draft_model
        for name, mod in model.named_modules():
            if name == "":
                continue
            mod.register_full_backward_hook(_make_hook(name))
        logger.warning("[rank %d] Installed backward debug hooks on draft model", rank)

    def _train_step_eagle3(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Eagle3 draft model training step (single micro-batch)."""
        step = self.global_step
        draft_dtype = next(self._draft_model.parameters()).dtype
        aux_hidden = teacher_data["hidden_states"].to(device=self._device, dtype=draft_dtype)  # [B, T, 3*H]
        input_ids = teacher_data["input_ids"].to(self._device)
        lm_head_w = self._lm_head_weight.to(device=self._device, dtype=draft_dtype)
        embed_w = self._embed_weight.to(device=self._device, dtype=draft_dtype)
        token_embeds = torch.nn.functional.embedding(input_ids, embed_w)
        loss_mask = attention_mask.to(self._device)
        spec_cfg = self.config.algorithm.spec_distill
        self._debug_sync("data_to_gpu", step)

        target_hs = teacher_data.get("last_hidden_states")
        if target_hs is not None:
            target_hs = target_hs.to(device=self._device, dtype=draft_dtype)
            # Mask out positions where teacher produced NaN hidden states
            nan_mask = torch.isnan(target_hs).any(dim=-1)  # [B, T]
            if nan_mask.any():
                nan_count = nan_mask.sum().item()
                loss_mask = loss_mask.clone()
                loss_mask[:, :nan_mask.shape[1]][nan_mask] = 0
                target_hs = target_hs.nan_to_num_(0.0)
                logger.warning("Masked %d teacher NaN positions from loss", nan_count)
            # vLLM returns pre-norm hidden states; apply teacher's final RMSNorm
            if self._norm_weight is not None:
                nw = self._norm_weight.to(device=self._device, dtype=draft_dtype)
                ths_f32 = target_hs.float()
                variance = ths_f32.pow(2).mean(-1, keepdim=True)
                target_hs = (ths_f32 * torch.rsqrt(variance + self._norm_eps)).to(draft_dtype) * nw
        self._debug_sync("rmsnorm_target", step)

        result = self._draft_model(
            token_embeds=token_embeds,
            aux_hidden_states=aux_hidden,
            teacher_lm_head_weight=lm_head_w,
            embed_weight=embed_w,
            loss_mask=loss_mask,
            target_ids=input_ids,
            loss_type=spec_cfg.loss_type,
            target_hidden_states=target_hs,
        )
        self._debug_sync("draft_forward", step)

        total_loss = torch.tensor(0.0, device=self._device)
        metrics: dict[str, float] = {}

        for i, (step_loss, step_acc) in enumerate(
            zip(result["losses"], result["accuracies"])
        ):
            weight = spec_cfg.position_decay ** i
            total_loss = total_loss + weight * step_loss
            metrics[f"step_{i}_loss"] = float(step_loss.detach())
            metrics[f"step_{i}_acc"] = float(step_acc.detach())

        num_accum = getattr(self, "_grad_accum_steps", 1)
        scaled_loss = total_loss / num_accum
        if self._grad_scaler is not None:
            self._grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._debug_sync("backward", step)
        metrics["loss"] = float(total_loss.detach())
        return metrics

    def _train_step_dflash(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        """DFlash draft model training step (single micro-batch)."""
        teacher_hidden = teacher_data["hidden_states"].to(self._device)
        input_ids = teacher_data["input_ids"].to(self._device)
        lm_head_w = self._lm_head_weight.to(self._device)
        loss_mask = attention_mask.to(self._device)

        spec_cfg = self.config.algorithm.spec_distill

        result = self._draft_model(
            teacher_hidden_states=teacher_hidden,
            lm_head_weight=lm_head_w,
            loss_mask=loss_mask,
            target_ids=input_ids,
            loss_decay_gamma=spec_cfg.loss_decay_gamma,
        )

        loss = result.get("loss")
        if loss is None:
            return {"loss": 0.0}

        num_accum = getattr(self, "_grad_accum_steps", 1)
        (loss / num_accum).backward()

        metrics: dict[str, float] = {"loss": float(loss.detach())}
        if "accuracy" in result:
            metrics["accuracy"] = float(result["accuracy"].detach())
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Spec Distill loop: teacher forward on dataset -> draft model train."""
        if self._draft_model is None and self._teacher_model is None and self._teacher_engine is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        total_steps = int(self.config.num_training_steps)
        logger.info("[rank %d] SpecDistill train loop: total_steps=%d", self._rank, total_steps)
        spec_cfg = self.config.algorithm.spec_distill
        draft_type = spec_cfg.draft_type.lower()

        use_vllm = self.config.algorithm.teacher.inference_backend == "vllm"
        prefetcher = _TeacherPrefetcher(self) if (use_vllm and self._rank == 0) else None

        if prefetcher is not None:
            prefetcher.prefetch(0)
            if total_steps > 1:
                prefetcher.prefetch(1)

        self._install_backward_hooks()

        for step in range(total_steps):
            step_start = time.time()
            self.global_step = step
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            t0 = time.time()
            if prefetcher is not None:
                rank0_data, input_ids, attention_mask = prefetcher.get()
                teacher_data = self._teacher_broadcast(rank0_data, input_ids)
            else:
                input_ids, attention_mask = self._get_batch_sequences(step)
                teacher_data = self._teacher_forward(input_ids, attention_mask)
            teacher_time = time.time() - t0
            self._debug_sync("teacher_broadcast", step)

            if prefetcher is not None and step + 2 < total_steps:
                prefetcher.prefetch(step + 2)

            t1 = time.time()
            B = teacher_data["input_ids"].shape[0]
            T = teacher_data["input_ids"].shape[1]
            adaptive_mb = max(1, 16384 // max(T, 1))
            mb = max(1, min(adaptive_mb, B))
            num_accum = (B + mb - 1) // mb
            self._grad_accum_steps = num_accum
            self._optimizer.zero_grad(set_to_none=True)

            metrics_accum: dict[str, float] = {}
            accum_count = 0
            self._draft_model.train()

            for mb_start in range(0, B, mb):
                mb_end = min(mb_start + mb, B)
                mb_teacher = {k: v[mb_start:mb_end] for k, v in teacher_data.items()}
                mb_mask = attention_mask[mb_start:mb_end]

                if draft_type == "eagle3":
                    m = self._train_step_eagle3(mb_teacher, mb_mask)
                else:
                    m = self._train_step_dflash(mb_teacher, mb_mask)

                for k, v in m.items():
                    if isinstance(v, float) and v == v:
                        metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                accum_count += 1

            metrics = {k: v / accum_count for k, v in metrics_accum.items()}

            if self._grad_scaler is not None:
                self._grad_scaler.unscale_(self._optimizer)

            nan_count = 0
            nan_names = []
            for name, p in self._draft_model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    nan_frac = torch.isnan(p.grad).float().mean().item()
                    nan_names.append(f"{name}({nan_frac:.3f})")
                    p.grad.zero_()
                    nan_count += 1
            if nan_count > 0:
                logger.warning("Zeroed NaN grads in %d params: %s", nan_count, ", ".join(nan_names[:5]))

            self._debug_sync("pre_grad_clip", step)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._draft_model.parameters(),
                max_norm=self.config.policy.max_grad_norm,
            )
            self._debug_sync("grad_clip", step)
            if not torch.isfinite(grad_norm):
                logger.warning("Skipping optimizer step (grad_norm=%s)", grad_norm)
            elif self._grad_scaler is not None:
                self._grad_scaler.step(self._optimizer)
            else:
                self._optimizer.step()
            self._debug_sync("optimizer_step", step)
            if self._grad_scaler is not None:
                self._grad_scaler.update()
            if self._scheduler is not None:
                self._scheduler.step()
            self._optimizer.zero_grad(set_to_none=True)
            train_time = time.time() - t1

            metrics["grad_norm"] = float(grad_norm)
            metrics["lr"] = self._optimizer.param_groups[0]["lr"]
            metrics["timing/step_s"] = time.time() - step_start
            metrics["timing/teacher_s"] = teacher_time
            metrics["timing/train_s"] = train_time
            metrics["seq/max_len"] = int(input_ids.shape[1])

            if self._is_distributed:
                for k in list(metrics.keys()):
                    t = torch.tensor(
                        metrics[k], dtype=torch.float64, device=self._device
                    )
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.AVG
                    )
                    metrics[k] = float(t.item())
            self._debug_sync("all_reduce", step)

            self.last_metrics = metrics

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            del teacher_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info(
            "[rank %d] SpecDistillTrainer.train finished after %d steps.",
            self._rank,
            total_steps,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources."""
        if self._teacher_engine is not None:
            self._teacher_engine.shutdown()
            self._teacher_engine = None
        if self._mooncake_master is not None:
            self._mooncake_master.shutdown()
            self._mooncake_master = None
        del self._teacher_model, self._draft_model
        self._teacher_model = None
        self._draft_model = None
        self._lm_head_weight = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] SpecDistillTrainer.cleanup complete.", self._rank)


__all__ = ["SpecDistillTrainer"]
