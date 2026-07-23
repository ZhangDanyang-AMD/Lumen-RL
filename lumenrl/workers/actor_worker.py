"""Policy (actor) worker: training backends and log-prob computation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lumenrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    kl_penalty,
    policy_gradient_loss,
)
from lumenrl.core.protocol import DataProto
from lumenrl.core.types import AlgorithmName, TrainingBackend
from lumenrl.engine.training.base_engine import BaseEngine, EngineRegistry
from lumenrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class LumenActorWorker(BaseWorker):
    """Trainable policy worker using the Engine abstraction layer.

    Delegates model construction, optimizer, LR scheduling, and offload
    management to the Engine layer (FSDP2 or Megatron).
    """

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._engine: BaseEngine | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Build policy network via EngineRegistry."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        backend_raw = str(
            policy.get("training_backend", TrainingBackend.FSDP2.value)
        ).lower()

        if backend_raw in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        elif backend_raw == "megatron":
            backend_key = "megatron"
        elif backend_raw in ("megatron_native", "megatron-native"):
            backend_key = "megatron_native"
        else:
            raise ValueError(f"Unknown policy.training_backend: {backend_raw}")

        model_name = str(policy.get("model_name", ""))
        training_cfg = policy.get("training", {}) or {}
        quant = get_nested_config(self.config, "quantization", "training", default={}) or {}

        engine_config = self._build_engine_config(backend_key, training_cfg, policy)
        optimizer_config = self._build_optimizer_config(policy)
        model_config = self._build_model_config(policy)

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend=backend_key,
        )

        if backend_key in ("fsdp", "fsdp2"):
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
                quant_config=quant,
            )
        else:
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
            )

        self._engine.initialize()
        self._log.info("LumenActorWorker: initialized %s engine.", backend_key)

    def _probe_forward_entropy(self, model_name: str) -> None:
        """One-time sanity probe: entropy of the FSDP forward on a fixed clean
        sequence. A correct base-model forward gives ~1-1.5; ~4+ indicates the
        sharded forward is degenerate."""
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            base = tk("Question: What is 12*13? Answer: 12*13 = 156.", add_special_tokens=False)["input_ids"]

            def _probe(ids, npad, tag):
                real = list(ids)
                seq = torch.tensor([[0] * npad + real], device=self._device)
                am = torch.tensor([[0] * npad + [1] * len(real)], device=self._device)
                data = {"input_ids": seq, "attention_mask": am,
                        "position_ids": (am.long().cumsum(-1) - 1).clamp(min=0),
                        "meta": {"calculate_entropy": True}}
                with self._engine.eval_mode():
                    out = self._engine.infer_batch(data)
                em = am[:, 1:].float()
                e = out["model_output"]["entropy"].float()
                ent = float((e * em).sum() / em.sum().clamp(min=1))
                self._log.info("PROBE_FWD %s entropy=%.3f (len=%d pad=%d)", tag, ent, len(real), npad)

            _probe(base, 0, "short_nopad")
            _probe(base, 512, "short_pad512")
            _probe(base * 40, 512, "long_pad512")
        except Exception as exc:
            self._log.warning("PROBE_FWD failed: %s", exc)

    def get_mp_info(self) -> dict[str, Any]:
        """Report this actor's Megatron model-parallel coordinates.

        Used by the controller to build the DP x (TP,PP,CP) data ``mesh_mapping`` and
        the loss-normalizing DP size. Falls back to pure-DP (rank/world) when
        Megatron model-parallel state is not initialized (FSDP / legacy backends).
        """
        try:
            from megatron.core import parallel_state as mpu
            if mpu.is_initialized():
                return {
                    # Exclude CP: all CP ranks holding different token chunks of
                    # one sequence must receive the same controller data shard.
                    "dp_rank": int(mpu.get_data_parallel_rank(with_context_parallel=False)),
                    "dp_size": int(mpu.get_data_parallel_world_size(with_context_parallel=False)),
                    "tp_rank": int(mpu.get_tensor_model_parallel_rank()),
                    "pp_rank": int(mpu.get_pipeline_model_parallel_rank()),
                    "cp_rank": int(mpu.get_context_parallel_rank()),
                    "cp_size": int(mpu.get_context_parallel_world_size()),
                    "is_last_stage": bool(mpu.is_pipeline_last_stage()),
                }
        except Exception:
            pass
        return {
            "dp_rank": int(self.rank), "dp_size": int(self.world_size),
            "tp_rank": 0, "pp_rank": 0, "cp_rank": 0, "cp_size": 1,
            "is_last_stage": True,
        }

    def _build_engine_config(
        self, backend: str, training_cfg: dict, policy: dict,
    ) -> dict[str, Any]:
        if backend in ("fsdp", "fsdp2"):
            fsdp_cfg = training_cfg.get("fsdp_cfg") or {}
            if not isinstance(fsdp_cfg, dict):
                from dataclasses import asdict, is_dataclass
                fsdp_cfg = asdict(fsdp_cfg) if is_dataclass(fsdp_cfg) else dict(vars(fsdp_cfg))
            # Mixed-precision (verl-aligned): keep FP32 master weights so small
            # Adam updates (lr ~1e-6 ~= 5e-7/step) accumulate. Storing the master
            # in bf16 (eps ~8e-3) silently rounds these updates away -> the policy
            # never changes (grad_norm looks normal but entropy/reward stay flat).
            # Compute/all-gather in bf16 via FSDP2 MixedPrecisionPolicy.
            compute_dtype = training_cfg.get("optimizer_dtype", "bf16")
            return {
                "param_offload": fsdp_cfg.get("param_offload", False),
                "optimizer_offload": fsdp_cfg.get("optimizer_offload", False),
                "grad_offload": fsdp_cfg.get("grad_offload", False),
                "reshard_after_forward": fsdp_cfg.get("reshard_after_forward", True),
                "model_dtype": "fp32",
                "mixed_precision": {
                    "param_dtype": compute_dtype,
                    "reduce_dtype": "fp32",
                },
                "seed": int(policy.get("seed", 42)),
            }
        elif backend in ("megatron", "megatron_native"):
            meg_cfg = training_cfg.get("megatron_cfg") or policy.get("megatron_cfg") or {}
            if not isinstance(meg_cfg, dict):
                from dataclasses import asdict, is_dataclass
                meg_cfg = asdict(meg_cfg) if is_dataclass(meg_cfg) else dict(vars(meg_cfg))
            return {
                "tensor_model_parallel_size": meg_cfg.get("tensor_model_parallel_size", 1),
                "pipeline_model_parallel_size": meg_cfg.get("pipeline_model_parallel_size", 1),
                "context_parallel_size": meg_cfg.get("context_parallel_size", 1),
                "expert_model_parallel_size": meg_cfg.get("expert_model_parallel_size", 1),
                "sequence_parallel": meg_cfg.get("sequence_parallel", False),
                "param_offload": meg_cfg.get("param_offload", False),
                "optimizer_offload": meg_cfg.get("optimizer_offload", False),
                "grad_offload": meg_cfg.get("grad_offload", False),
                "seed": int(policy.get("seed", 42)),
                "dtype": meg_cfg.get("dtype", "bf16"),
                "use_distributed_optimizer": meg_cfg.get("use_distributed_optimizer", False),
                # Activation recomputation (needed for long sequences: Megatron
                # local-spec attention is O(seq^2) in memory without TE flash).
                "recompute_granularity": meg_cfg.get("recompute_granularity", None),
                "recompute_method": meg_cfg.get("recompute_method", None),
                "recompute_num_layers": meg_cfg.get("recompute_num_layers", None),
                # Long-sequence memory: flash attn (O(L)) + chunked/fused log-prob.
                "attention_backend": meg_cfg.get("attention_backend", "unfused"),
                "log_probs_chunk_size": meg_cfg.get("log_probs_chunk_size", 0),
                # Dynamic-batch packing (flash_attn_varlen) for GEMM efficiency.
                "enable_dynamic_batch": meg_cfg.get("enable_dynamic_batch", False),
                "max_tokens_per_gpu": meg_cfg.get("max_tokens_per_gpu", 0),
            }
        return {}

    def _build_optimizer_config(self, policy: dict) -> dict[str, Any]:
        lr = float(policy.get("learning_rate", policy.get("lr", 1e-6)))
        return {
            "lr": lr,
            "weight_decay": float(policy.get("weight_decay", 0.01)),
            "clip_grad": float(policy.get("max_grad_norm", 1.0)),
            "lr_warmup_steps": int(policy.get("lr_warmup_steps", 10)),
            "lr_warmup_steps_ratio": float(policy.get("warmup_ratio", 0.0)),
        }

    def _build_model_config(self, policy: dict) -> dict[str, Any]:
        return {
            "local_path": str(policy.get("model_name", "")),
            "trust_remote_code": True,
        }

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return per-token log-probs (and optionally entropy) for the policy.

        Uses the SAME packed-varlen forward as the training step and as
        ``RLTrainer._compute_log_probs_for_model`` (remove-padding + AITER
        ``flash_attn_varlen_func``), which is the path validated to match verl
        (``use_remove_padding=True``) to 1e-4 on entropy / log-prob / loss.

        The earlier padded left-pad + SDPA forward here produced ~7x inflated
        entropy: left-pad tokens were fed to attention with plain arange
        positions, flattening the logit distribution. Packing removes all pad
        tokens and gives every real token its exact varlen position.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before compute_log_probs().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        # Megatron backend owns its own (GPTModel) forward + logprob path.
        if hasattr(self._engine, "engine_compute_log_probs"):
            out = self._engine.engine_compute_log_probs(batch)
            # Every TP/PP/CP member ran the collective forward on the same data.
            # The engine designates one rank after CP reconstruction so the
            # controller's DP merge does not duplicate this group's rows.
            src = getattr(self._engine, "is_mp_src_rank_with_outputs", None)
            if src is not None and not src():
                return DataProto(meta=dict(batch.meta))
            return out

        from lumenrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_entropy,
            packed_token_log_probs,
            unpack_log_probs,
        )

        sequences = batch["input_ids"].to(self._device)
        B, S = sequences.shape
        am = batch.tensors.get("attention_mask")
        if am is not None:
            am = am.to(self._device)
        else:
            # No mask supplied -> treat every token as real. Only safe for
            # already-unpadded input; the Ray controller always sends the mask.
            am = torch.ones_like(sequences)

        want_entropy = bool(batch.meta.get("calculate_entropy", False))
        # verl divides logits by the sampling temperature before log_prob/entropy
        # (logits.div_(temperature)); apply the same so old/train/ref stay unbiased.
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        # Pack ONE sequence per forward: dropping the left-pad and running plain
        # causal attention on a single segment is correct regardless of whether
        # the varlen (packing) attention patch is installed. Packing multiple
        # sequences here would require that patch for cross-sequence isolation
        # (which is disabled under LUMEN_DISABLE_HF_ATTN_PATCH=1 / pure SDPA).
        import torch.distributed as dist

        n_rows = B
        if dist.is_initialized() and dist.get_world_size() > 1:
            # FSDP2 all-gathers per forward must run in lockstep across ranks;
            # equalize the number of forwards (pad with a dummy last row).
            cnt = torch.tensor([n_rows], device=self._device)
            dist.all_reduce(cnt, op=dist.ReduceOp.MAX)
            n_iters = int(cnt.item())
        else:
            n_iters = n_rows

        lp_parts: list[torch.Tensor] = []
        ent_parts: list[torch.Tensor] = []
        module = self._engine.module

        with self._engine.eval_mode():
            with torch.no_grad():
                for i in range(n_iters):
                    row = min(i, B - 1)  # dummy rows reuse the last real row
                    ids_row = sequences[row:row + 1]
                    mask_row = am[row:row + 1]
                    packed = pack_sequences(ids_row, mask_row)
                    with PackingContext(packed.cu_seqlens, packed.max_seqlen):
                        outputs = module(
                            input_ids=packed.input_ids,
                            position_ids=packed.position_ids,
                            attention_mask=None,
                            use_cache=False,
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        logits = logits.squeeze(0)
                        flat_lp = packed_token_log_probs(
                            logits, packed.input_ids.squeeze(0), packed.cu_seqlens,
                            temperature=temperature,
                        )
                        token_lp = unpack_log_probs(
                            flat_lp, packed.cu_seqlens, packed.seq_lens, S,
                        )
                        if want_entropy:
                            flat_ent = packed_token_entropy(
                                logits, packed.cu_seqlens,
                                temperature=temperature, upcast=True,
                            )
                            token_ent = unpack_log_probs(
                                flat_ent, packed.cu_seqlens, packed.seq_lens, S,
                            )
                        if i < n_rows:
                            lp_parts.append(token_lp)
                            if want_entropy:
                                ent_parts.append(token_ent)
                        del outputs, logits

        log_probs = torch.cat(lp_parts, dim=0)
        tensors = {"log_probs": log_probs.cpu(), "input_ids": batch["input_ids"]}
        if want_entropy and ent_parts:
            tensors["entropy"] = torch.cat(ent_parts, dim=0).cpu()
        out = DataProto(tensors=tensors, meta=dict(batch.meta))
        return out

    def train_step(self, batch: DataProto) -> dict[str, float]:
        """Compute the RL surrogate loss, backward, and step the optimizer.

        The worker recomputes the forward pass locally so gradients flow
        through the model parameters.  The algorithm name and hyperparameters
        are passed via ``batch.meta``.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        def loss_fn(model_output, data):
            token_log_probs = model_output["log_probs"]
            input_ids = data["input_ids"]

            old_logp = batch.tensors.get("old_log_probs")
            adv = batch.tensors.get("advantages")
            algo_name = str(batch.meta.get("algorithm", "grpo")).lower()

            if old_logp is not None and adv is not None:
                old_logp = old_logp.to(token_log_probs.device)
                adv = adv.to(token_log_probs.device)

                if adv.dim() == 1:
                    adv = adv.unsqueeze(-1).expand_as(token_log_probs)
                elif adv.shape[-1] != token_log_probs.shape[-1]:
                    adv = adv[..., :token_log_probs.shape[-1]]

                if old_logp.shape[-1] != token_log_probs.shape[-1]:
                    old_logp = old_logp[..., :token_log_probs.shape[-1]]

                # Prefer response_mask (PPO loss is over response tokens only).
                # log_probs[:, j] predicts target token j+1, so shift the mask by
                # one to align (verl-aligned); fall back to attention_mask.
                L = token_log_probs.shape[-1]
                resp_mask = batch.tensors.get("response_mask")
                if resp_mask is not None:
                    mask = resp_mask.to(token_log_probs.device).float()
                    if mask.shape[-1] == L + 1:
                        mask = mask[:, 1:]
                    mask = mask[..., :L]
                else:
                    mask = batch.tensors.get("attention_mask")
                    if mask is not None:
                        mask = mask.to(token_log_probs.device)[..., :L].float()

                # Algo hyperparams live under the per-algo sub-config
                # (algo_config[algo_name]); the flat top-level keys are None.
                algo_cfg_full = batch.meta.get("algo_config", {}) or {}
                _sub = algo_cfg_full.get(algo_name)
                _sub = _sub if isinstance(_sub, dict) else {}

                def _cfg(key, default):
                    v = _sub.get(key, algo_cfg_full.get(key, default))
                    return default if v is None else v

                if algo_name == AlgorithmName.DAPO.value:
                    clip_low = float(_cfg("clip_ratio_low", 0.2))
                    clip_high = float(_cfg("clip_ratio_high", 0.28))
                    loss = asymmetric_clip_loss(
                        token_log_probs, old_logp, adv, clip_low, clip_high, mask=mask,
                    )
                else:
                    clip = float(_cfg("clip_ratio", 0.2))
                    loss = policy_gradient_loss(
                        token_log_probs, old_logp, adv, clip, mask=mask,
                    )

                kl_c = float(_cfg("kl_coeff", 0.0))
                ref_logp = batch.tensors.get("ref_log_probs")
                if kl_c > 0.0 and ref_logp is not None:
                    ref_logp = ref_logp.to(token_log_probs.device)
                    if ref_logp.shape[-1] != token_log_probs.shape[-1]:
                        ref_logp = ref_logp[..., :token_log_probs.shape[-1]]
                    kl = kl_penalty(token_log_probs, ref_logp, mask=mask)
                    loss = loss + kl_c * kl
            else:
                shift_labels = input_ids[:, 1:].contiguous().to(token_log_probs.device)
                loss = F.cross_entropy(
                    token_log_probs.view(-1),
                    shift_labels.view(-1),
                )

            return loss, {"loss": float(loss.detach())}

        data = {"input_ids": batch["input_ids"].to(self._device)}
        if "attention_mask" in batch:
            data["attention_mask"] = batch["attention_mask"].to(self._device)

        with self._engine.train_mode():
            output = self._engine.train_batch(data, loss_fn)

        metrics: dict[str, float] = {}
        if "metrics" in output:
            for k, v in output["metrics"].items():
                if isinstance(v, list):
                    metrics[k] = sum(v) / max(len(v), 1)
                else:
                    metrics[k] = float(v)

        if "loss" in output:
            if isinstance(output["loss"], list):
                metrics["loss"] = sum(output["loss"]) / max(len(output["loss"]), 1)
            else:
                metrics["loss"] = float(output["loss"])

        lr = self._engine.lr_scheduler_step()
        metrics["lr"] = lr

        algo_name = str(batch.meta.get("algorithm", "grpo")).lower()
        self._log.debug("train_step loss=%f (algo=%s)", metrics.get("loss", 0.0), algo_name)
        return metrics

    def update_policy(self, batch: DataProto) -> dict[str, float]:
        """verl-aligned worker-side PPO update over this actor's DP shard.

        The controller dispatches the full advantage-augmented batch across all
        actors (each gets ``global / num_workers`` rows) with ``batch_num_tokens``
        (GLOBAL response-token count) and ``dp_size`` in meta. This runs ONE
        optimizer step; the engine micro-batches internally (grad accumulation),
        deferring the FSDP2 reduce-scatter to the last micro. The per-token PG
        loss is normalized by the global token count and dp_size-compensated, so
        summing shard gradients (FSDP averages across ranks) yields the correct
        global token-mean -- matching the validated torchrun path.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before update_policy().")
        if batch.batch_size == 0:
            return {"loss": 0.0}

        # Megatron backend owns its own (GPTModel) forward-backward + DAPO loss.
        if hasattr(self._engine, "engine_update_policy"):
            return self._engine.engine_update_policy(batch)

        data: dict[str, Any] = {k: v for k, v in batch.tensors.items()}
        # verl-aligned packed (remove-padding + varlen) training forward: the
        # engine packs each micro, dropping all pad tokens. This is the forward
        # that matches verl (use_remove_padding=True) and keeps old_log_probs
        # (also packed) consistent with the train forward (ratio ~= 1). The old
        # padded SDPA path flattened logits (AITER attn does not mask left-pad),
        # inflating entropy ~7x.
        data["use_packed_forward"] = True
        data["temperature"] = float(batch.meta.get("temperature", 1.0) or 1.0)
        # 1 sequence per micro (memory); grads accumulate across micros.
        data["micro_batch_size"] = 1
        self._pol_meta = dict(batch.meta)
        if not getattr(self, "_logged_norm", False):
            self._logged_norm = True
            self._log.info(
                "update_policy norm: batch_num_tokens=%s dp_size=%s rows=%d has_ris=%s",
                self._pol_meta.get("batch_num_tokens"), self._pol_meta.get("dp_size"),
                batch.batch_size, "rollout_is_weights" in batch.tensors,
            )

        with self._engine.train_mode():
            output = self._engine.train_batch(data, self._policy_loss_fn)

        metrics: dict[str, float] = {}
        for k, v in output.get("metrics", {}).items():
            metrics[k] = (sum(v) / len(v)) if isinstance(v, list) and v else float(v)
        if "loss" in output:
            lv = output["loss"]
            metrics["loss"] = (sum(lv) / len(lv)) if isinstance(lv, list) and lv else float(lv)
        metrics["lr"] = self._engine.lr_scheduler_step()
        return metrics

    def reset_memory_stats(self) -> bool:
        """Reset per-step CUDA/HIP peak memory counters for this actor rank."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        return True

    def get_memory_stats(self) -> dict[str, float]:
        """Return current-step peak memory counters for this actor rank."""
        if not torch.cuda.is_available():
            return {
                "max_reserved_bytes": 0.0,
                "max_allocated_bytes": 0.0,
                "reserved_bytes": 0.0,
                "allocated_bytes": 0.0,
            }
        return {
            "max_reserved_bytes": float(torch.cuda.max_memory_reserved()),
            "max_allocated_bytes": float(torch.cuda.max_memory_allocated()),
            "reserved_bytes": float(torch.cuda.memory_reserved()),
            "allocated_bytes": float(torch.cuda.memory_allocated()),
        }

    def _policy_loss_fn(self, model_output, data):
        """DAPO/GRPO surrogate on ONE micro, using the micro's own tensors.

        Reads old_log_probs/advantages/response_mask/rollout_is_weights from the
        micro (``data``) -- NOT a closed-over full batch -- so shapes always
        match ``model_output``. Applies dual-clip + TIS + global-token
        normalization (batch_num_tokens / dp_size from meta), matching verl.
        """
        token_log_probs = model_output["log_probs"]
        L = token_log_probs.shape[-1]
        dev = token_log_probs.device
        meta = getattr(self, "_pol_meta", {}) or {}
        algo_name = str(meta.get("algorithm", "dapo")).lower()

        old_logp = data.get("old_log_probs")
        adv = data.get("advantages")
        if old_logp is None or adv is None:
            shift_labels = data["input_ids"][:, 1:].contiguous().to(dev)
            loss = F.cross_entropy(token_log_probs.reshape(-1, token_log_probs.shape[-1]) if token_log_probs.dim() == 3 else token_log_probs.reshape(-1), shift_labels.reshape(-1))
            return loss, {"loss": float(loss.detach())}

        # response mask aligned to [., L]
        resp_mask = data.get("response_mask")
        if resp_mask is not None:
            mask = resp_mask.to(dev).float()
            if mask.shape[-1] == L + 1:
                mask = mask[:, 1:]
            mask = mask[..., :L]
        else:
            am = data.get("attention_mask")
            mask = am.to(dev)[..., :L].float() if am is not None else None

        old_logp = old_logp.to(dev)[..., :L]
        adv = adv.to(dev)
        if adv.dim() == 1:
            adv = adv.unsqueeze(-1).expand_as(token_log_probs)
        elif adv.shape[-1] != L:
            adv = adv[..., :L]

        algo_cfg_full = meta.get("algo_config", {}) or {}
        _sub = algo_cfg_full.get(algo_name)
        _sub = _sub if isinstance(_sub, dict) else {}

        def _cfg(key, default):
            v = _sub.get(key, algo_cfg_full.get(key, default))
            return default if v is None else v

        bnt = meta.get("batch_num_tokens")
        dp = int(meta.get("dp_size", 1) or 1)
        ris = data.get("rollout_is_weights")
        if ris is not None:
            ris = ris.to(dev)
            if ris.dim() > 1 and ris.shape[-1] != L:
                ris = ris[..., :L]

        if algo_name == AlgorithmName.DAPO.value:
            loss = asymmetric_clip_loss(
                token_log_probs, old_logp, adv,
                float(_cfg("clip_ratio_low", 0.2)), float(_cfg("clip_ratio_high", 0.28)),
                mask=mask, clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
            )
        else:
            loss = policy_gradient_loss(
                token_log_probs, old_logp, adv, float(_cfg("clip_ratio", 0.2)), mask=mask,
            )

        kl_c = float(_cfg("kl_coeff", 0.0))
        ref_logp = data.get("ref_log_probs")
        if kl_c > 0.0 and ref_logp is not None:
            ref_logp = ref_logp.to(dev)[..., :L]
            loss = loss + kl_c * kl_penalty(token_log_probs, ref_logp, mask=mask)

        # KL diagnostics (verl-aligned, detached, over response tokens). Return
        # per-micro SUM + token COUNT (not a per-sequence mean): the controller
        # aggregates as a GLOBAL token-weighted mean (verl masked_mean over the
        # whole batch). A mean-of-per-sequence-means would over-weight short/
        # outlier sequences and make core/kl look much noisier than verl.
        #   ppo_kl          = Σ(old_logp - new_logp) / Σtok  -> verl actor/ppo_kl
        #   rollout_corr/kl = Σ(rollout_logp - new_logp) / Σtok -> verl rollout_corr/kl
        out_metrics = {"loss": float(loss.detach())}
        if mask is not None:
            with torch.no_grad():
                tok = float(mask.sum())
                out_metrics["ppo_kl_sum"] = float(((old_logp - token_log_probs) * mask).sum())
                out_metrics["ppo_kl_tok"] = tok
                rlp = data.get("rollout_log_probs")
                if rlp is not None:
                    rlp = rlp.to(dev)[..., :L]
                    out_metrics["rollout_corr_kl_sum"] = float(((rlp - token_log_probs) * mask).sum())
                    out_metrics["rollout_corr_kl_tok"] = tok
        return loss, out_metrics

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU full state dict for weight sync to rollout engines.

        FSDP2 shards parameters as ``DTensor`` across the actor process group,
        so ``full_tensor()`` all-gathers the unsharded tensor. This is a
        COLLECTIVE op: every actor must call ``get_state_dict`` concurrently or
        the all-gather deadlocks (see ``_fetch_actor_cpu_state``).
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before get_state_dict().")
        from torch.distributed.tensor import DTensor

        params, _ = self._engine.get_per_tensor_param()
        out: dict[str, torch.Tensor] = {}
        for name, param in params:
            full = param.full_tensor() if isinstance(param, DTensor) else param
            out[name] = full.detach().cpu()
        return out

    def save_checkpoint(self, local_path: str, global_step: int = 0) -> bool:
        """Save this actor rank's sharded training state.

        This mirrors verl's Ray worker checkpoint contract: every actor writes
        its own model/optimizer/scheduler shard under the same step directory.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before save_checkpoint().")
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        # Native Megatron engine: use sharded dist-checkpoint (no DP duplication,
        # resharding-capable) instead of per-rank torch.save of full weights.
        if hasattr(self._engine, "save_dist_checkpoint"):
            self._engine.save_dist_checkpoint(str(path), global_step=global_step)
            return True
        rank = int(self.rank)
        world = int(self.world_size)
        module = getattr(self._engine, "module", None)
        optimizer = getattr(self._engine, "optimizer", None)
        scheduler = getattr(self._engine, "lr_scheduler", None)
        if module is None:
            raise RuntimeError("Engine has no module to checkpoint.")

        torch.save(module.state_dict(), path / f"model_world_size_{world}_rank_{rank}.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), path / f"optim_world_size_{world}_rank_{rank}.pt")
        extra = {
            "global_step": int(global_step),
            "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        torch.save(extra, path / f"extra_state_world_size_{world}_rank_{rank}.pt")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return True

    def load_checkpoint(self, local_path: str) -> int:
        """Load this actor rank's sharded training state and return global_step."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before load_checkpoint().")
        path = Path(local_path)
        if hasattr(self._engine, "load_dist_checkpoint"):
            return self._engine.load_dist_checkpoint(str(path))
        rank = int(self.rank)
        world = int(self.world_size)
        module = getattr(self._engine, "module", None)
        optimizer = getattr(self._engine, "optimizer", None)
        scheduler = getattr(self._engine, "lr_scheduler", None)
        if module is None:
            raise RuntimeError("Engine has no module to restore.")

        model_path = path / f"model_world_size_{world}_rank_{rank}.pt"
        optim_path = path / f"optim_world_size_{world}_rank_{rank}.pt"
        extra_path = path / f"extra_state_world_size_{world}_rank_{rank}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint shard: {model_path}")

        module.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        if optimizer is not None and optim_path.exists():
            optimizer.load_state_dict(torch.load(optim_path, map_location="cpu", weights_only=False))
        global_step = 0
        if extra_path.exists():
            extra = torch.load(extra_path, map_location="cpu", weights_only=False)
            global_step = int(extra.get("global_step", 0))
            sched_state = extra.get("lr_scheduler")
            if scheduler is not None and sched_state is not None:
                scheduler.load_state_dict(sched_state)
            rng = extra.get("rng") or {}
            if rng.get("cpu") is not None:
                torch.set_rng_state(rng["cpu"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state(rng["cuda"])
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return global_step

    def update_weights_ipc_send(
        self, bucket_size_mb: int = 512, use_shm: bool = False,
    ) -> bool:
        """Stream full (all-gathered) BF16 weights to the colocated vLLM replica.

        verl-aligned ZMQ CUDA-IPC weight sync. ``full_tensor()`` is an FSDP
        all-gather COLLECTIVE, so every actor must call this concurrently (the
        trainer dispatches it to all actors at once). Each actor sends to its
        own replica's socket keyed by ``{job_id, replica_rank=self.rank,
        local_rank=0}`` -- matching the receiver's ``_get_zmq_handle``. The
        receiver is started separately (server.update_weights_from_ipc) and
        both run concurrently.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before update_weights_ipc_send().")

        import asyncio

        import ray
        from torch.distributed.tensor import DTensor

        from lumenrl.engine.inference.bucketed_weight_transfer import BucketedWeightSender

        job_id = ray.get_runtime_context().get_job_id()
        handle = f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{self.rank}-rank-0.sock"
        keep_fp32 = os.environ.get("LUMENRL_SYNC_FP32") == "1"

        params, _ = self._engine.get_per_tensor_param()

        def _gen():
            for name, param in params:
                full = param.full_tensor() if isinstance(param, DTensor) else param
                full = full.detach()
                if full.dtype == torch.float32 and not keep_fp32:
                    full = full.to(torch.bfloat16)
                if not full.is_cuda:
                    full = full.to("cuda", non_blocking=True)
                yield name, full

        sender = BucketedWeightSender(
            zmq_handle=handle, bucket_size_mb=int(bucket_size_mb), use_shm=bool(use_shm)
        )
        asyncio.run(sender.async_send_weights(_gen()))
        return True

    def cleanup(self) -> None:
        self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
