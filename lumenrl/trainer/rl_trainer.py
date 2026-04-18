"""High-level RL training orchestration on the controller process.

Supports three modes:
- **Sync colocate** (default): rollout and training share the same GPUs,
  swapping memory between vLLM and FSDP2 via optimizer offload.
- **Async separated**: Rollouter and Trainer run on separate GPU groups with
  a message queue and periodic parameter sync (see ``AsyncRLTrainer``).
- **Local mode** (single-GPU / testing): all workers run in the controller process.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Type

import torch

import lumenrl.algorithms  # noqa: F401  — populate ALGORITHM_REGISTRY
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.quantization.rollout_correction import apply_rollout_correction
from lumenrl.trainer.callbacks import Callback, LoggingCallback
from lumenrl.utils.metrics import MetricsTracker, compute_kl_divergence

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

        self._actor_model: torch.nn.Module | None = None
        self._ref_model: torch.nn.Module | None = None
        self._ref_on_cpu: bool = True
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None
        self._atom_engine: Any = None
        self._use_atom: bool = config.policy.generation_backend.lower() == "atom"
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

        quant = {}
        tq = self.config.quantization.training
        if tq.fp8:
            quant["fp8"] = tq.fp8
        quant["fp8_weight_cache"] = tq.fp8_weight_cache

        from lumenrl.engine.training.fsdp_backend import FSDP2Backend

        logger.info("[rank %d] Building actor model: %s", self._rank, model_name)
        self._actor_model = FSDP2Backend.build_model(model_name)
        self._actor_model = FSDP2Backend.apply_lumen_optimizations(self._actor_model, quant)

        if self._is_distributed:
            fsdp_cfg = {"enabled": True}
            if hasattr(self.config.policy, "training") and hasattr(self.config.policy.training, "fsdp_cfg") and self.config.policy.training.fsdp_cfg:
                fsdp_cfg.update(self.config.policy.training.fsdp_cfg)
            self._actor_model = FSDP2Backend.apply_fsdp2(self._actor_model, fsdp_cfg)
        else:
            self._actor_model.to(self._device)
        lr = 1e-6
        self._optimizer = torch.optim.AdamW(
            [p for p in self._actor_model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

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

        if not self.callbacks:
            self.callbacks.append(LoggingCallback(interval=max(1, self.config.logger.log_interval)))

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info("[rank %d] RLTrainer.setup complete: algo=%s, model=%s, world_size=%d, atom=%s",
                     self._rank, self.config.algorithm.name, model_name, self._world_size, self._use_atom)

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

    def _offload_optimizer_to_cpu(self) -> None:
        """Move optimizer state tensors to CPU to free GPU memory for rollout."""
        if self._optimizer is None:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

    def _reload_optimizer_to_gpu(self) -> None:
        """Move optimizer state tensors back to GPU for the next training step."""
        if self._optimizer is None:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    state[k] = v.to(self._device, non_blocking=True)

    def _sync_weights_to_atom(self) -> None:
        """Extract FSDP actor weights and push to ATOM engine for next rollout.

        TODO: Currently skipped because safetensors has issues with FSDP2
        lazy storage pointers.  The vLLM subprocess is always spawned fresh
        with the original model weights.  Weight sync will be re-enabled
        once we implement proper FSDP2 state_dict consolidation.
        """
        if self._atom_engine is None or self._actor_model is None:
            return
        logger.debug("Weight sync to ATOM skipped (FSDP2 lazy storage workaround).")

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

        if self._rank == 0:
            self._atom_engine.wake()
            response_texts = self._atom_engine.generate(expanded_prompts, sampling_params=sp)
            self._atom_engine.sleep()

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
        """Per-token log-probs via row-chunked logsumexp to bound peak memory.

        Processes one sequence at a time so the float32 promotion for
        numerical stability only needs ``[S, V]`` instead of ``[B, S, V]``.
        """
        logits_shifted = logits[:, :-1]            # [B, S-1, V]  bf16 view
        targets = target_ids[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
        chosen = logits_shifted.gather(-1, targets).squeeze(-1).float()  # [B, S-1]
        log_z_parts = []
        for i in range(logits_shifted.shape[0]):
            log_z_parts.append(logits_shifted[i].float().logsumexp(dim=-1))
        log_z = torch.stack(log_z_parts, dim=0)    # [B, S-1]
        return chosen - log_z

    def _compute_log_probs_for_model(
        self,
        model: torch.nn.Module,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        move_to_gpu: bool = False,
    ) -> torch.Tensor:
        """Compute per-token log-probs from model for given sequences.

        Uses a memory-efficient fused gather+logsumexp instead of full
        log_softmax over the vocabulary.

        When ``move_to_gpu`` is True, moves the model to GPU before forward
        and back to CPU afterward (for CPU-offloaded reference model).
        """
        if move_to_gpu:
            model.to(self._device)

        sequences = sequences.to(self._device)
        attention_mask = attention_mask.to(self._device)

        model.eval()
        micro_bs = max(1, int(self.config.policy.train_micro_batch_size))
        all_log_probs = []

        with torch.no_grad():
            for start in range(0, sequences.shape[0], micro_bs):
                end = min(start + micro_bs, sequences.shape[0])
                ids_chunk = sequences[start:end]
                mask_chunk = attention_mask[start:end]
                outputs = model(input_ids=ids_chunk, attention_mask=mask_chunk)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                token_lp = self._fused_token_log_probs(logits, ids_chunk)
                all_log_probs.append(token_lp)

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

        return rewards.to(self._device), responses

    def _build_response_mask(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Create a mask that is 1 only for response tokens (excluding prompt)."""
        mask = attention_mask.clone()
        for i, plen in enumerate(prompt_lengths):
            mask[i, :plen] = 0
        return mask[:, 1:]

    def _dynamic_mini_batches(
        self,
        batch: DataProto,
        max_token_len: int,
        fallback_bs: int,
    ) -> list[DataProto]:
        """Split batch into mini-batches capped by total token count.

        Each mini-batch has at most ``max_token_len`` total tokens
        (sum of sequence lengths across all rows). Falls back to
        fixed ``fallback_bs`` if ``max_token_len <= 0``.
        """
        if max_token_len <= 0:
            return list(batch.mini_batches(fallback_bs))

        seq_lens = batch.tensors["attention_mask"].sum(dim=1)
        batches: list[DataProto] = []
        start = 0
        n = batch.batch_size
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(seq_lens[end].item())
                if tok_count + sl > max_token_len and end > start:
                    break
                tok_count += sl
                end += 1
            chunk = {k: v[start:end] for k, v in batch.tensors.items()}
            batches.append(DataProto(tensors=chunk, meta=batch.meta.copy()))
            start = end
        return batches

    def _train_step(self, batch: DataProto) -> dict[str, float]:
        """One gradient step on the actor model."""
        if self._actor_model is None or self._optimizer is None:
            raise RuntimeError("setup() must be called first.")

        self._actor_model.train()
        sequences = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        outputs = self._actor_model(input_ids=sequences, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        token_log_probs = self._fused_token_log_probs(logits, sequences)

        batch.tensors["log_probs"] = token_log_probs

        if self.global_step < 3 and self._rank == 0:
            mb_idx = getattr(self, "_nan_debug_mb_idx", 0)
            self._nan_debug_mb_idx = mb_idx + 1
            olp = batch.tensors["old_log_probs"]
            nlp = token_log_probs
            rmask = batch.tensors.get("response_mask")
            adv = batch.tensors.get("advantages")
            ratio_dbg = torch.exp(nlp - olp)
            logger.info(
                "NaN-DEBUG mb=%d: B=%d, adv min=%.4f max=%.4f zero=%d/%d, "
                "ratio nan=%d inf=%d min=%.4f max=%.4f, "
                "logp nan=%d oldlogp nan=%d logits nan=%d",
                mb_idx, sequences.shape[0],
                adv.min().item() if adv is not None else 0,
                adv.max().item() if adv is not None else 0,
                (adv.abs() < 1e-8).sum().item() if adv is not None else -1,
                adv.numel() if adv is not None else -1,
                ratio_dbg.isnan().sum().item(), ratio_dbg.isinf().sum().item(),
                ratio_dbg.min().item(), ratio_dbg.max().item(),
                nlp.isnan().sum().item(), olp.isnan().sum().item(),
                logits.isnan().sum().item(),
            )

        loss, metrics = self._algorithm.compute_loss(batch)
        loss = loss.to(self._device)

        if self.global_step < 3 and self._rank == 0:
            mb_idx = getattr(self, "_nan_debug_mb_idx", 1) - 1
            logger.info(
                "NaN-DEBUG step=%d mb=%d loss: loss=%.6f loss_pg=%.6f loss_total=%.6f "
                "loss_isnan=%s",
                self.global_step, mb_idx, float(loss.detach()),
                metrics.get("loss_pg", float("nan")),
                metrics.get("loss_total", float("nan")),
                loss.isnan().item(),
            )

        if loss.isnan():
            metrics["loss"] = float("nan")
            return metrics

        num_accum = getattr(self, "_grad_accum_steps", 1)
        scaled_loss = loss / num_accum
        scaled_loss.backward()

        metrics["loss"] = float(loss.detach())
        return metrics

    def train(self) -> None:
        """Main training loop: rollout → ref → reward → advantages → train → sync."""
        if self._algorithm is None or self._actor_model is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        num_generations = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)

        for step in range(total_steps):
            step_start = time.time()
            self.global_step = step
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            prompts, ground_truths = self._get_batch_prompts(step)
            input_ids, attention_mask = self._tokenize_prompts(prompts)

            if self._use_atom and self._atom_engine is not None and step > 0:
                self._offload_optimizer_to_cpu()

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

            if self._use_atom and self._atom_engine is not None and step > 0:
                self._reload_optimizer_to_gpu()

            prompt_tok = int(attention_mask.repeat_interleave(
                num_generations, dim=0).to(seq_mask.device).sum().item()) if num_generations > 0 else 0
            gen_tokens = int(seq_mask.sum().item()) - prompt_tok if num_generations > 0 else 0

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

            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            t2 = time.time()
            micro_bs = max(1, int(self.config.policy.train_micro_batch_size))
            max_tok = int(self.config.policy.max_token_len_per_gpu)
            mini_batches = self._dynamic_mini_batches(batch, max_tok, micro_bs)
            self._grad_accum_steps = len(mini_batches)
            self._optimizer.zero_grad(set_to_none=True)
            metrics_accum: dict[str, float] = {}
            step_count = 0
            for mini in mini_batches:
                m = self._train_step(mini)
                for k, v in m.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                step_count += 1
            torch.nn.utils.clip_grad_norm_(self._actor_model.parameters(), max_norm=1.0)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            train_time = time.time() - t2

            denom = max(1, step_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}

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

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            self._sync_rollout_weights()

            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_end(self, step, metrics)

            if self._is_distributed:
                torch.distributed.barrier()

            del sequences, seq_mask, old_log_probs, ref_log_probs
            del rewards, responses, response_mask, batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info("[rank %d] RLTrainer.train finished after %d steps.", self._rank, total_steps)

    def _sync_rollout_weights(self) -> None:
        """Sync actor weights to ATOM rollout engine if configured."""
        if self._use_atom and self._atom_engine is not None:
            self._sync_weights_to_atom()

    def run_validation(self) -> dict[str, float]:
        """Run a lightweight validation pass."""
        return {"val/kl_proxy": 0.0}

    def cleanup(self) -> None:
        """Release all resources."""
        if self._atom_engine is not None:
            self._atom_engine.shutdown()
            self._atom_engine = None
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
