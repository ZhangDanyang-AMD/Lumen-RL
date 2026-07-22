# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""VIME-style Megatron-Core training engine for LumenRL (BF16, TP=1/PP=1/DP=N).

Builds a real Megatron-Core ``GPTModel`` (Qwen3), loads HF weights, and runs
the DAPO/GRPO RL step through Megatron modules -- the same training stack VIME
uses (GPTModel + Megatron forward) -- while plugging into LumenRL's Ray
controller via the ``BaseEngine`` interface.

Scope: tensor/pipeline parallel = 1, data parallel = world size (the LumenRL
BF16 smoke). Mixed precision uses BF16 compute with FP32 master weights; data
parallel gradient sync is a manual mean all-reduce over the world group.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lumenrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    kl_penalty,
    policy_gradient_loss,
)
from lumenrl.core.protocol import DataProto
from lumenrl.core.types import AlgorithmName
from lumenrl.engine.training.base_engine import BaseEngine, EngineRegistry
from lumenrl.engine.training.qwen3_megatron_bridge import (
    Qwen3Dims,
    hf_to_megatron,
    load_hf_safetensors,
    megatron_to_hf,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "INFO"))


class MegatronEngine(BaseEngine):
    """Megatron-Core GPTModel engine (Qwen3, BF16, TP=PP=1, DP=world)."""

    def __init__(self, model_config, engine_config, optimizer_config, model_name: str = ""):
        super().__init__()
        self.model_config = model_config if isinstance(model_config, dict) else vars(model_config)
        self.engine_config = engine_config if isinstance(engine_config, dict) else vars(engine_config)
        self.optimizer_config = (
            optimizer_config if isinstance(optimizer_config, dict) else vars(optimizer_config)
        )
        self.model_name = model_name or self.model_config.get("local_path", "")
        self.module: torch.nn.Module | None = None   # unwrapped GPTModel (eval fwd, save/load)
        self._ddp: Any = None                          # Megatron DistributedDataParallel wrapper
        self.optimizer: Any = None                     # Megatron distributed optimizer
        self.lr_scheduler: Any = None                  # Megatron OptimizerParamScheduler
        self._dims: Qwen3Dims | None = None
        self._step = 0
        self.mode: str | None = None

    # -- offload (Ray path: never offload) --
    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def train_mode(self, **kwargs):
        return nullcontext()

    def eval_mode(self, **kwargs):
        return nullcontext()

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        import megatron.core.transformer.transformer_block as tb
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.torch_norm import WrappedTorchNorm as WTN
        from megatron.core.transformer.transformer_config import TransformerConfig

        ec = self.engine_config
        tp = int(ec.get("tensor_model_parallel_size", 1))
        pp = int(ec.get("pipeline_model_parallel_size", 1))
        cp = int(ec.get("context_parallel_size", 1))
        ep = int(ec.get("expert_model_parallel_size", 1))
        seed = int(ec.get("seed", 42))

        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                expert_model_parallel_size=ep,
            )
        model_parallel_cuda_manual_seed(seed)

        # ---- HF config -> Qwen3 dims / TransformerConfig ----
        cfg_path = os.path.join(self.model_name, "config.json")
        with open(cfg_path) as fh:
            hf = json.load(fh)
        head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
        self._dims = Qwen3Dims(
            num_layers=hf["num_hidden_layers"], hidden=hf["hidden_size"],
            num_heads=hf["num_attention_heads"], num_kv_groups=hf["num_key_value_heads"],
            head_dim=head_dim, ffn=hf["intermediate_size"], vocab=hf["vocab_size"],
        )
        tb.LayerNormImpl = WTN  # force torch RMSNorm (apex FusedLayerNorm lacks RMSNorm)
        tfcfg = TransformerConfig(
            num_layers=hf["num_hidden_layers"], hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_query_groups=hf["num_key_value_heads"], kv_channels=head_dim,
            ffn_hidden_size=hf["intermediate_size"], gated_linear_unit=True,
            activation_func=F.silu, add_bias_linear=False,
            add_qkv_bias=bool(hf.get("attention_bias", False)),
            normalization="RMSNorm", layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
            qk_layernorm=True, hidden_dropout=0.0, attention_dropout=0.0,
            bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec(qk_layernorm=True)
        spec.submodules.input_layernorm = WTN
        spec.submodules.pre_mlp_layernorm = WTN
        spec.submodules.self_attention.submodules.q_layernorm = WTN
        spec.submodules.self_attention.submodules.k_layernorm = WTN

        model = GPTModel(
            config=tfcfg, transformer_layer_spec=spec, vocab_size=hf["vocab_size"],
            max_sequence_length=hf.get("max_position_embeddings", 32768),
            pre_process=True, post_process=True, position_embedding_type="rope",
            rotary_base=hf.get("rope_theta", 1000000.0),
            share_embeddings_and_output_weights=bool(hf.get("tie_word_embeddings", False)),
        )

        # ---- load HF weights ----
        logger.info("MegatronEngine[%d]: loading HF weights from %s", self._rank(), self.model_name)
        hf_state = load_hf_safetensors(self.model_name)
        meg_state = hf_to_megatron(hf_state, self._dims)
        del hf_state
        missing = model.load_state_dict(meg_state, strict=False)
        real_missing = [k for k in missing.missing_keys if "_extra_state" not in k]
        if real_missing:
            raise RuntimeError(f"Megatron load missing keys: {real_missing[:6]} ...")
        del meg_state
        self.module = model.cuda().bfloat16()
        self._tfcfg = tfcfg

        # ---- Megatron DistributedDataParallel: shards optimizer state across DP ----
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        oc = self.optimizer_config
        self._clip = float(oc.get("clip_grad", 1.0))
        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
            average_in_collective=True,
            bucket_size=None,
        )
        self._ddp = DDP(config=tfcfg, ddp_config=ddp_cfg, module=self.module)

        opt_cfg = OptimizerConfig(
            optimizer="adam", lr=float(oc.get("lr", 1e-6)),
            weight_decay=float(oc.get("weight_decay", 0.1)),
            adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
            clip_grad=self._clip, bf16=True, fp16=False,
            params_dtype=torch.bfloat16, use_distributed_optimizer=True,
        )
        self.optimizer = get_megatron_optimizer(opt_cfg, model_chunks=[self._ddp])

        warmup = int(oc.get("lr_warmup_steps", 10))
        base_lr = float(oc.get("lr", 1e-6))
        wd = float(oc.get("weight_decay", 0.1))
        self.lr_scheduler = OptimizerParamScheduler(
            self.optimizer, init_lr=0.0, max_lr=base_lr, min_lr=base_lr,
            lr_warmup_steps=warmup, lr_decay_steps=max(warmup + 1, 1000),
            lr_decay_style="constant", start_wd=wd, end_wd=wd,
            wd_incr_steps=0, wd_incr_style="constant",
        )
        if self._rank() == 0:
            n = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
            logger.info(
                "MegatronEngine: model+distributed-optimizer ready, %d params, dp_size=%d",
                n, self.get_data_parallel_size(),
            )

    # ------------------------------------------------------------------
    def _rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def get_data_parallel_size(self) -> int:
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_world_size()
        except Exception:
            return dist.get_world_size() if dist.is_initialized() else 1

    def get_data_parallel_rank(self) -> int:
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_rank()
        except Exception:
            return self._rank()

    def get_data_parallel_group(self):
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_group()
        except Exception:
            return dist.group.WORLD if dist.is_initialized() else None

    def is_mp_src_rank_with_outputs(self) -> bool:
        return True

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True) -> None:
        return

    # ------------------------------------------------------------------
    def _forward_logits(self, ids: torch.Tensor, model=None) -> torch.Tensor:
        """Run the model on a single unpadded sequence -> logits [L, V] (float).

        ``model`` defaults to the unwrapped GPTModel (eval); pass ``self._ddp``
        during training so DDP grad hooks fire and grads land in the buffer.
        """
        m = model if model is not None else self.module
        L = ids.numel()
        inp = ids.view(1, L)
        pos = torch.arange(L, device=ids.device).view(1, L)
        out = m(input_ids=inp, position_ids=pos, attention_mask=None)
        logits = out.logits if hasattr(out, "logits") else out
        return logits.view(L, -1).float()

    @staticmethod
    def _real_block(mask_row: torch.Tensor) -> tuple[int, int]:
        idx = mask_row.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 0, 0
        return int(idx[0].item()), int(idx.numel())

    # ---- engine-level compute_log_probs (actor delegates here) ----
    def engine_compute_log_probs(self, batch: DataProto) -> DataProto:
        seqs = batch["input_ids"]
        B, S = seqs.shape
        am = batch.tensors.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        want_ent = bool(batch.meta.get("calculate_entropy", False))
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        lp_out = torch.zeros(B, S, dtype=torch.float32)
        ent_out = torch.zeros(B, S, dtype=torch.float32) if want_ent else None
        self.module.eval()
        with torch.no_grad():
            for r in range(B):
                start, L = self._real_block(am[r])
                if L < 2:
                    continue
                ids = seqs[r, start:start + L].to("cuda")
                logits = self._forward_logits(ids) / temperature  # [L,V]
                lp = torch.log_softmax(logits[:-1], dim=-1)
                tok_lp = lp.gather(-1, ids[1:].view(-1, 1)).squeeze(-1)  # [L-1]
                lp_out[r, start:start + L - 1] = tok_lp.cpu()
                if want_ent:
                    p = torch.softmax(logits[:-1], dim=-1)
                    ent = -(p * torch.log_softmax(logits[:-1], dim=-1)).sum(-1)
                    ent_out[r, start:start + L - 1] = ent.cpu()
        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    # ---- engine-level DAPO/GRPO update (actor delegates here) ----
    def engine_update_policy(self, batch: DataProto) -> dict[str, float]:
        if batch.batch_size == 0:
            return {"loss": 0.0, "lr": self._cur_lr(), "grad_norm": 0.0}
        meta = dict(batch.meta)
        algo_name = str(meta.get("algorithm", "dapo")).lower()
        temperature = float(meta.get("temperature", 1.0) or 1.0)
        bnt = meta.get("batch_num_tokens")
        dp = int(meta.get("dp_size", self.get_data_parallel_size()) or 1)
        algo_cfg_full = meta.get("algo_config", {}) or {}
        _sub = algo_cfg_full.get(algo_name)
        _sub = _sub if isinstance(_sub, dict) else {}

        def _cfg(key, default):
            v = _sub.get(key, algo_cfg_full.get(key, default))
            return default if v is None else v

        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        B, S = seqs.shape

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        loss_accum = 0.0
        ppo_kl_sum = 0.0
        ppo_kl_tok = 0.0
        rc_kl_sum = 0.0
        rc_kl_tok = 0.0
        n_rows = 0

        for r in range(B):
            start, L = self._real_block(am[r])
            if L < 2:
                continue
            ids = seqs[r, start:start + L].to("cuda")
            logits = self._forward_logits(ids, model=self._ddp) / temperature  # [L,V] (grad)
            lp = torch.log_softmax(logits[:-1], dim=-1)
            token_lp = lp.gather(-1, ids[1:].view(-1, 1)).squeeze(-1).view(1, -1)  # [1,L-1]
            Lm = token_lp.shape[-1]
            dev = token_lp.device

            def _col(name, shift):
                x = t.get(name)
                if x is None:
                    return None
                x = x[r].to(dev)
                s0 = start + (1 if shift else 0)
                return x[s0:].reshape(1, -1)

            old_lp = _col("old_log_probs", shift=False)
            if old_lp is None:
                continue
            resp_mask = _col("response_mask", shift=True)
            adv_t = t.get("advantages")
            if adv_t is None:
                continue
            if adv_t.dim() == 1:
                adv = adv_t[r].to(dev).view(1, 1).expand(1, Lm).float()
            else:
                adv = adv_t[r].to(dev)[start + 1:].reshape(1, -1).float()
            ris = _col("rollout_is_weights", shift=False)
            ref_lp0 = _col("ref_log_probs", shift=False)
            rlp0 = _col("rollout_log_probs", shift=False)

            # Align every per-token tensor + token_lp to their common min length
            # (rollout tensors can differ by one at the sequence boundary).
            cand = [token_lp, old_lp, adv]
            for v in (resp_mask, ris, ref_lp0, rlp0):
                if v is not None:
                    cand.append(v)
            Le = min(v.shape[-1] for v in cand)
            token_lp = token_lp[..., :Le]
            old_lp = old_lp[..., :Le]
            adv = adv[..., :Le]
            mask = resp_mask[..., :Le].float() if resp_mask is not None else None
            ris = ris[..., :Le] if ris is not None else None
            ref_lp = ref_lp0[..., :Le] if ref_lp0 is not None else None
            rlp = rlp0[..., :Le] if rlp0 is not None else None

            if algo_name == AlgorithmName.DAPO.value:
                loss = asymmetric_clip_loss(
                    token_lp, old_lp, adv,
                    float(_cfg("clip_ratio_low", 0.2)), float(_cfg("clip_ratio_high", 0.28)),
                    mask=mask, clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                    batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
                )
            else:
                loss = policy_gradient_loss(
                    token_lp, old_lp, adv, float(_cfg("clip_ratio", 0.2)), mask=mask,
                )
            kl_c = float(_cfg("kl_coeff", 0.0))
            if kl_c > 0.0 and ref_lp is not None:
                loss = loss + kl_c * kl_penalty(token_lp, ref_lp, mask=mask)

            loss.backward()
            loss_accum += float(loss.detach())
            n_rows += 1
            if mask is not None:
                with torch.no_grad():
                    tok = float(mask.sum())
                    ppo_kl_sum += float(((old_lp - token_lp) * mask).sum())
                    ppo_kl_tok += tok
                    if rlp is not None:
                        rc_kl_sum += float(((rlp - token_lp) * mask).sum())
                        rc_kl_tok += tok

        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        metrics = {
            "loss": loss_accum / max(1, n_rows),
            "lr": lr,
            "grad_norm": grad_norm,
        }
        if ppo_kl_tok > 0:
            metrics["ppo_kl_sum"] = ppo_kl_sum
            metrics["ppo_kl_tok"] = ppo_kl_tok
        if rc_kl_tok > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl_sum
            metrics["rollout_corr_kl_tok"] = rc_kl_tok
        return metrics

    def _optimizer_step(self) -> float:
        """Reduce grads across DP (+reduce-scatter for the distributed optimizer),
        then step the Megatron distributed optimizer."""
        from megatron.core.distributed import finalize_model_grads
        finalize_model_grads([self._ddp])
        update_successful, grad_norm, _num_zeros = self.optimizer.step()
        if not update_successful:
            logger.warning("optimizer.step reported update_successful=False")
        return float(grad_norm) if grad_norm is not None else 0.0

    def _cur_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return 0.0

    def _sched_step(self) -> float:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(increment=1)
        return self._cur_lr()

    def lr_scheduler_step(self) -> float:
        return self._cur_lr()

    # ---- weight sync: Megatron -> HF named tensors ----
    def get_per_tensor_param(self, **kwargs):
        assert self.module is not None
        named = [(n, p.detach()) for n, p in self.module.named_parameters()]
        gen = megatron_to_hf(named, self._dims)
        return gen, None


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngine):
    pass
