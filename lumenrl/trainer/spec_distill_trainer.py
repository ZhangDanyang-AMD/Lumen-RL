"""Speculative Decoding Draft Model Distillation trainer.

Implements the TorchSpec-style off-policy distillation loop for training
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
import time
from typing import Any

import torch
import torch.nn.functional as F

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.trainer.callbacks import Callback, LoggingCallback

logger = logging.getLogger(__name__)


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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[rank %d] Loading teacher model: %s", self._rank, teacher_name)
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

        # Extract lm_head weight for draft model's logit projection
        lm_head_weight = None
        for name, param in self._teacher_model.named_parameters():
            if "lm_head" in name and "weight" in name:
                lm_head_weight = param.detach().clone()
                break
        if lm_head_weight is None:
            raise RuntimeError("Could not find lm_head.weight in teacher model.")
        self._lm_head_weight = lm_head_weight

        # Determine hidden dim from teacher
        teacher_hidden_dim = lm_head_weight.shape[1]

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
            self._draft_model = Eagle3Model(
                hidden_dim=teacher_hidden_dim,
                num_heads=max(1, teacher_hidden_dim // 128),
                num_layers=num_layers,
                length=5,
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

        self._draft_model.to(self._device)

        if self._is_distributed:
            from lumenrl.engine.training.fsdp_backend import FSDP2Backend

            self._draft_model = FSDP2Backend.apply_fsdp2(
                self._draft_model, {"enabled": True}
            )

        self._optimizer = torch.optim.AdamW(
            [p for p in self._draft_model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=0.01,
        )

        # ---- Tokenizer ----
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

        logger.info(
            "Loaded dataset: %d samples from %s", len(self._dataset), dataset_path
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
                raw = s.get("prompt") or s.get("question") or s.get("input") or ""
                answer = (
                    s.get("answer")
                    or s.get("solution")
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
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
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
        - ``"hidden_states"``: last-layer hidden states ``[B, T, D]``
        - ``"token_embeds"``: input embeddings ``[B, T, D]`` (for Eagle3)
        - ``"input_ids"``: the original input ids
        """
        if self._teacher_model is None:
            raise RuntimeError("Teacher model not loaded.")

        spec_cfg = self.config.algorithm.spec_distill
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

                # Get embeddings (first hidden state = after embedding layer)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    all_embeds.append(outputs.hidden_states[0].cpu())
                    all_hidden.append(outputs.hidden_states[-1].cpu())
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

    def _train_step_eagle3(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Eagle3 draft model training step."""
        self._draft_model.train()

        token_embeds = teacher_data["token_embeds"].to(self._device)
        teacher_hidden = teacher_data["hidden_states"].to(self._device)
        input_ids = teacher_data["input_ids"].to(self._device)
        lm_head_w = self._lm_head_weight.to(self._device)
        loss_mask = attention_mask.to(self._device)

        result = self._draft_model(
            token_embeds=token_embeds,
            teacher_hidden=teacher_hidden,
            lm_head_weight=lm_head_w,
            loss_mask=loss_mask,
            target_ids=input_ids,
        )

        if "losses" not in result or not result["losses"]:
            return {"loss": 0.0}

        spec_cfg = self.config.algorithm.spec_distill
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
        (total_loss / num_accum).backward()
        metrics["loss"] = float(total_loss.detach())
        return metrics

    def _train_step_dflash(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        """DFlash draft model training step."""
        self._draft_model.train()

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
        if self._draft_model is None or self._teacher_model is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        total_steps = int(self.config.num_training_steps)
        spec_cfg = self.config.algorithm.spec_distill
        draft_type = spec_cfg.draft_type.lower()
        micro_bs = max(1, int(self.config.policy.train_micro_batch_size))

        for step in range(total_steps):
            step_start = time.time()
            self.global_step = step
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            # Phase 1: Get dataset sequences
            input_ids, attention_mask = self._get_batch_sequences(step)

            # Phase 2: Teacher forward (no grad) -> hidden states
            t0 = time.time()
            teacher_data = self._teacher_forward(input_ids, attention_mask)
            teacher_time = time.time() - t0

            # Phase 3: Draft model training
            t1 = time.time()
            self._grad_accum_steps = 1
            self._optimizer.zero_grad(set_to_none=True)

            if draft_type == "eagle3":
                metrics = self._train_step_eagle3(teacher_data, attention_mask)
            else:
                metrics = self._train_step_dflash(teacher_data, attention_mask)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._draft_model.parameters(), max_norm=1.0
            )
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            train_time = time.time() - t1

            metrics["grad_norm"] = float(grad_norm)
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
