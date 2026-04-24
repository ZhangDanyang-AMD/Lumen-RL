"""On-Policy Distillation (OPD) trainer.

Implements the DeepSeek-V4 style OPD training loop:

1. Student rollout (no grad) -- generate sequences.
2. Teacher forward (no grad) -- produce logits on student sequences.
3. Student second forward (with grad) -- produce student logits.
4. Minimise KL(student || teacher) via :func:`opd_kl_divergence`.

The student model is also the actor; the teacher is a separate frozen model.
Optionally supports *lazy logits* (teacher returns hidden states, logits are
reconstructed via ``TeacherLMHead``).
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any

import torch

import lumenrl.algorithms  # noqa: F401 -- populate ALGORITHM_REGISTRY
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.trainer.callbacks import Callback, LoggingCallback

logger = logging.getLogger(__name__)


class OPDTrainer:
    """Coordinates student rollout, teacher forward, and OPD training.

    Lifecycle: ``setup() -> train() -> cleanup()``.
    """

    def __init__(self, config: LumenRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []
        self._algorithm: Any = None

        self._student_model: torch.nn.Module | None = None
        self._teacher_model: torch.nn.Module | None = None
        self._teacher_lm_head: Any = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None
        self._atom_engine: Any = None
        self._use_atom: bool = config.policy.generation_backend.lower() == "atom"
        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = torch.distributed.get_rank() if self._is_distributed else 0
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
        """Initialize student, teacher, optimizer, dataset, and algorithm."""
        model_name = self.config.policy.model_name
        if not model_name:
            raise ValueError("config.policy.model_name is required (student model).")

        algo_cls = ALGORITHM_REGISTRY.get(self.config.algorithm.name)
        self._algorithm = algo_cls(self.config)

        from lumenrl.engine.training.fsdp_backend import FSDP2Backend

        # ---- Student (actor) model ----
        logger.info("[rank %d] Building student model: %s", self._rank, model_name)
        self._student_model = FSDP2Backend.build_model(model_name)
        quant: dict[str, Any] = {}
        tq = self.config.quantization.training
        if tq.fp8:
            quant["fp8"] = tq.fp8
        quant["fp8_weight_cache"] = tq.fp8_weight_cache
        self._student_model = FSDP2Backend.apply_lumen_optimizations(
            self._student_model, quant
        )
        if hasattr(self._student_model, "gradient_checkpointing_enable"):
            self._student_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        if self._is_distributed:
            fsdp_cfg: dict[str, Any] = {"enabled": True}
            if (
                hasattr(self.config.policy, "training")
                and hasattr(self.config.policy.training, "fsdp_cfg")
                and self.config.policy.training.fsdp_cfg
            ):
                fsdp_cfg.update(self.config.policy.training.fsdp_cfg)
            self._student_model = FSDP2Backend.apply_fsdp2(
                self._student_model, fsdp_cfg
            )
        else:
            self._student_model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            [p for p in self._student_model.parameters() if p.requires_grad],
            lr=1e-6,
            weight_decay=0.01,
        )

        # ---- Teacher model ----
        teacher_cfg = self.config.algorithm.teacher
        teacher_name = teacher_cfg.model_name or model_name
        opd_cfg = self.config.algorithm.opd

        if opd_cfg.lazy_logits:
            self._load_teacher_lazy(teacher_name, teacher_cfg)
        else:
            self._load_teacher_full(teacher_name)

        # ---- Tokenizer ----
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        # ---- Dataset ----
        self._load_dataset()

        # ---- ATOM engine for rollout ----
        if self._use_atom:
            from lumenrl.engine.inference.atom_engine import AtomEngine

            atom_cfg = self.config.policy.generation.atom_cfg
            self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
            logger.info("[rank %d] ATOM engine configured.", self._rank)

        # ---- Callbacks ----
        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info(
            "[rank %d] OPDTrainer.setup complete: model=%s, teacher=%s, lazy=%s",
            self._rank,
            model_name,
            teacher_name,
            opd_cfg.lazy_logits,
        )

    def _load_teacher_full(self, teacher_name: str) -> None:
        """Load the full teacher model for logits-mode OPD."""
        from transformers import AutoModelForCausalLM

        logger.info("[rank %d] Loading full teacher model: %s", self._rank, teacher_name)
        self._teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._teacher_model.eval()
        for p in self._teacher_model.parameters():
            p.requires_grad_(False)

    def _load_teacher_lazy(self, teacher_name: str, teacher_cfg: Any) -> None:
        """Load only teacher lm_head (+ optional norm) for lazy-logits OPD.

        In lazy mode we still need the full teacher for hidden states, but we
        also cache the TeacherLMHead for logit reconstruction.
        """
        from transformers import AutoModelForCausalLM
        from lumenrl.core.teacher_lm_head import TeacherLMHead

        logger.info(
            "[rank %d] Loading teacher (lazy logits mode): %s", self._rank, teacher_name
        )
        self._teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._teacher_model.eval()
        for p in self._teacher_model.parameters():
            p.requires_grad_(False)

        # Build TeacherLMHead from the loaded model's lm_head
        lm_head_weight = None
        for name, param in self._teacher_model.named_parameters():
            if "lm_head" in name and "weight" in name:
                lm_head_weight = param.detach().clone()
                break
        if lm_head_weight is not None:
            self._teacher_lm_head = TeacherLMHead(lm_head_weight)
            self._teacher_lm_head.to(self._device)
            logger.info("[rank %d] TeacherLMHead ready.", self._rank)

    def _load_dataset(self) -> None:
        """Load training dataset (same logic as RLTrainer)."""
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

        logger.info("Loaded dataset: %d samples from %s", len(self._dataset), dataset_path)

    # ------------------------------------------------------------------
    # Prompt helpers (shared with RLTrainer pattern)
    # ------------------------------------------------------------------

    def _get_batch_prompts(self, step: int) -> list[str]:
        """Return a list of prompt strings for the current step."""
        import json as _json

        bs = max(1, self.config.policy.train_global_batch_size)

        if self._dataset is None:
            return [
                f"What is {i + step * bs} + {i + step * bs + 1}?" for i in range(bs)
            ]

        dataset_len = len(self._dataset)
        start = (step * bs) % dataset_len
        indices = [(start + i) % dataset_len for i in range(bs)]
        samples = [self._dataset[idx] for idx in indices]

        prompts: list[str] = []
        for s in samples:
            raw = s.get("prompt") or s.get("question") or s.get("input") or ""
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

            if self._tokenizer is not None and hasattr(
                self._tokenizer, "apply_chat_template"
            ):
                orig = s.get("prompt") or s.get("question") or s.get("input") or ""
                if isinstance(orig, list):
                    try:
                        text = self._tokenizer.apply_chat_template(
                            orig, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        pass

            prompts.append(text)
        return prompts

    def _tokenize_prompts(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_prompt_len = min(self.config.policy.max_total_sequence_length // 2, 1024)
        enc = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate sequences from the student model (no grad)."""
        prompt_lens = attention_mask.sum(dim=1).tolist()

        max_resp = self.config.policy.max_response_length
        if max_resp > 0:
            max_gen = min(
                max_resp,
                self.config.policy.max_total_sequence_length - input_ids.shape[1],
            )
        else:
            max_gen = max(
                128,
                self.config.policy.max_total_sequence_length - input_ids.shape[1],
            )

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_gen,
            "pad_token_id": self._tokenizer.pad_token_id,
            "temperature": 0.7,
            "do_sample": True,
        }

        self._student_model.eval()
        ids_gpu = input_ids.to(self._device)
        mask_gpu = attention_mask.to(self._device)

        with torch.no_grad():
            sequences = self._student_model.generate(
                input_ids=ids_gpu, attention_mask=mask_gpu, **gen_kwargs
            )

        seq_mask = torch.ones(
            sequences.shape, dtype=torch.long, device=sequences.device
        )
        pad_id = self._tokenizer.pad_token_id
        if pad_id is not None:
            seq_mask = (sequences != pad_id).long()

        return sequences, seq_mask, prompt_lens

    # ------------------------------------------------------------------
    # Teacher forward
    # ------------------------------------------------------------------

    def _teacher_forward(
        self,
        sequences: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run teacher on student-generated sequences, return logits [B, T-1, V].

        In lazy mode: teacher returns hidden states, then TeacherLMHead
        reconstructs logits from them.
        """
        if self._teacher_model is None:
            raise RuntimeError("Teacher model not loaded.")

        opd_cfg = self.config.algorithm.opd
        micro_bs = max(1, opd_cfg.teacher_micro_batch_size)
        teacher_device = self._device

        # Move teacher to GPU if on CPU
        teacher_on_cpu = next(self._teacher_model.parameters()).device.type == "cpu"
        if teacher_on_cpu:
            self._teacher_model.to(teacher_device)

        all_logits: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, sequences.shape[0], micro_bs):
                end = min(start + micro_bs, sequences.shape[0])
                ids_chunk = sequences[start:end].to(teacher_device)
                mask_chunk = seq_mask[start:end].to(teacher_device)

                if opd_cfg.lazy_logits and self._teacher_lm_head is not None:
                    outputs = self._teacher_model(
                        input_ids=ids_chunk,
                        attention_mask=mask_chunk,
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[-1][:, :-1]
                    chunk_logits = self._teacher_lm_head(hidden)
                else:
                    outputs = self._teacher_model(
                        input_ids=ids_chunk, attention_mask=mask_chunk
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    chunk_logits = logits[:, :-1]

                all_logits.append(chunk_logits.cpu())
                del outputs

        if teacher_on_cpu:
            self._teacher_model.cpu()
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=0)  # [B, T-1, V]

    # ------------------------------------------------------------------
    # Student train step
    # ------------------------------------------------------------------

    @staticmethod
    def _fused_token_log_probs(
        logits: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token log-probs (row-chunked for bf16 stability)."""
        logits_shifted = logits[:, :-1]
        targets = target_ids[:, 1:].unsqueeze(-1)
        parts = []
        for i in range(logits_shifted.shape[0]):
            row_lp = torch.nn.functional.log_softmax(logits_shifted[i], dim=-1)
            parts.append(row_lp.gather(-1, targets[i]).squeeze(-1))
        return torch.stack(parts, dim=0).float()

    def _train_step(self, batch: DataProto) -> dict[str, float]:
        """One gradient accumulation step: student forward + OPD loss."""
        if self._student_model is None or self._optimizer is None:
            raise RuntimeError("setup() must be called first.")

        self._student_model.train()
        sequences = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        outputs = self._student_model(
            input_ids=sequences, attention_mask=attention_mask
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        student_logits = logits[:, :-1]  # [B, T-1, V]
        del outputs, logits

        batch.tensors["student_logits"] = student_logits
        # teacher_logits already on batch

        loss, metrics = self._algorithm.compute_loss(batch)
        loss = loss.to(self._device)

        if loss.isnan():
            metrics["loss"] = float("nan")
            return metrics

        num_accum = getattr(self, "_grad_accum_steps", 1)
        (loss / num_accum).backward()

        metrics["loss"] = float(loss.detach())
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """OPD loop: rollout -> teacher forward -> student train."""
        if self._algorithm is None or self._student_model is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        total_steps = int(self.config.num_training_steps)
        micro_bs = max(1, int(self.config.policy.train_micro_batch_size))

        for step in range(total_steps):
            step_start = time.time()
            self.global_step = step
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            # Phase 1: Get prompts
            prompts = self._get_batch_prompts(step)
            input_ids, attention_mask = self._tokenize_prompts(prompts)

            # Phase 2: Student rollout (no grad)
            t0 = time.time()
            sequences, seq_mask, prompt_lengths = self._rollout_phase(
                input_ids, attention_mask
            )
            gen_time = time.time() - t0

            # Phase 3: Teacher forward on student sequences
            t1 = time.time()
            teacher_logits = self._teacher_forward(sequences, seq_mask)
            teacher_time = time.time() - t1

            # Build response mask (loss only on response tokens)
            response_mask = seq_mask.clone()
            for i, plen in enumerate(prompt_lengths):
                response_mask[i, :plen] = 0
            response_mask = response_mask[:, 1:]  # align with shifted logits

            # Phase 4: Student training
            t2 = time.time()
            batch = DataProto(
                tensors={
                    "input_ids": sequences,
                    "attention_mask": seq_mask,
                    "teacher_logits": teacher_logits,
                    "response_mask": response_mask,
                },
                meta={"algorithm": self.config.algorithm.name},
            )

            mini_batches = list(batch.mini_batches(micro_bs))
            self._grad_accum_steps = len(mini_batches)
            self._optimizer.zero_grad(set_to_none=True)
            metrics_accum: dict[str, float] = {}
            step_count = 0

            for mini in mini_batches:
                m = self._train_step(mini)
                if m.get("loss") is not None and m["loss"] != m["loss"]:
                    continue
                for k, v in m.items():
                    if v == v:
                        metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                step_count += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._student_model.parameters(), max_norm=1.0
            )
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            train_time = time.time() - t2

            # Metrics
            denom = max(1, step_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}
            metrics["grad_norm"] = float(grad_norm)
            metrics["timing/step_s"] = time.time() - step_start
            metrics["timing/gen_s"] = gen_time
            metrics["timing/teacher_s"] = teacher_time
            metrics["timing/train_s"] = train_time
            metrics["seq/max_len"] = int(sequences.shape[1])

            if self._is_distributed:
                for k in list(metrics.keys()):
                    t = torch.tensor(
                        metrics[k], dtype=torch.float64, device=self._device
                    )
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    metrics[k] = float(t.item())

            self.last_metrics = metrics

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            del sequences, seq_mask, teacher_logits, batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info(
            "[rank %d] OPDTrainer.train finished after %d steps.",
            self._rank,
            total_steps,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources."""
        if self._atom_engine is not None:
            self._atom_engine.shutdown()
            self._atom_engine = None
        del self._student_model, self._teacher_model, self._teacher_lm_head
        self._student_model = None
        self._teacher_model = None
        self._teacher_lm_head = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] OPDTrainer.cleanup complete.", self._rank)


__all__ = ["OPDTrainer"]
