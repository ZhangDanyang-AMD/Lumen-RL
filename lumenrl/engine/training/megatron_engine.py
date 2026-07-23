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

import math  # noqa: E402

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:  # pragma: no cover - flash_attn optional
    _flash_attn_func = None
    _flash_attn_varlen_func = None


class FlashSelfAttentionCore(torch.nn.Module):
    """Flash-attention drop-in for Megatron's local-spec ``DotProductAttention``.

    The local (non-TE) core attention materializes the full ``[b, np, sq, sk]``
    score matrix -> **O(L^2)** memory, which OOMs at long RL response lengths
    (resp=20480). This replacement calls ``flash_attn_func`` (O(L) memory) and is
    swapped into the GPT layer spec's ``self_attention.submodules.core_attention``
    when ``megatron_cfg.attention_backend == "flash"``.

    Assumes causal self-attention on a single unpadded sequence (LumenRL's
    per-sequence forward). GQA (num_query_groups < num_heads) is handled natively
    by flash-attn, so we skip the KV ``repeat_interleave`` the local path does.
    """

    def __init__(self, config, layer_number: int = 1, attn_mask_type=None,
                 attention_type=None, cp_comm_type=None, softmax_scale=None, **kwargs):
        super().__init__()
        if _flash_attn_func is None:
            raise ImportError(
                "megatron_cfg.attention_backend='flash' requires the flash_attn "
                "package (import failed). Install flash-attn or set attention_backend='unfused'."
            )
        self.config = config
        self.layer_number = max(1, layer_number or 1)
        head_dim = getattr(config, "kv_channels", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # ---- packed varlen (thd): multiple concatenated sequences, one forward ----
        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            if _flash_attn_varlen_func is None:
                raise ImportError("dynamic-batch packing needs flash_attn.flash_attn_varlen_func")
            cu = packed_seq_params.cu_seqlens_q
            mx = int(packed_seq_params.max_seqlen_q)
            # Megatron sbhd with b=1: [T, 1, h, d] -> thd [T, h, d]
            q = query.squeeze(1)
            k = key.squeeze(1)
            v = value.squeeze(1)
            out = _flash_attn_varlen_func(
                q, k, v, cu, cu, mx, mx, causal=True, softmax_scale=self.softmax_scale,
            )  # [T, np, hn]
            return out.reshape(out.shape[0], 1, -1)  # [T, 1, np*hn]
        # ---- single unpadded sequence: Megatron layout [s,b,h,d] -> flash [b,s,h,d] ----
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        out = _flash_attn_func(q, k, v, causal=True, softmax_scale=self.softmax_scale)
        # [b, sq, np, hn] -> [sq, b, np*hn]
        out = out.transpose(0, 1).contiguous()
        return out.reshape(out.shape[0], out.shape[1], -1)


class _FusedTokenLogProb(torch.autograd.Function):
    """Memory-efficient per-token log-prob: ``log p(target) = logit_target - logsumexp``.

    Retains a single ``[L, V]`` softmax buffer for backward instead of the
    several ``[L, V]`` tensors that ``log_softmax(logits).gather(...)`` keeps
    alive (the full log_softmax output plus its gradient). Values/gradients are
    exact. Backward uses ``grad_logits = (onehot(target) - softmax) * grad_lp``.
    """

    @staticmethod
    def forward(ctx, logits, target):
        logits = logits.float()
        m = logits.max(dim=-1, keepdim=True).values          # [L,1]
        shifted = logits.sub(m)                               # new [L,V]
        exp = shifted.exp_()                                  # in-place -> exp
        Z = exp.sum(dim=-1, keepdim=True)                     # [L,1]
        softmax = exp.div_(Z)                                 # in-place -> softmax
        logZ = Z.log_().add_(m)                               # logsumexp [L,1]
        tgt_logit = logits.gather(-1, target.unsqueeze(-1))   # [L,1]
        log_prob = (tgt_logit - logZ).squeeze(-1)             # [L]
        ctx.save_for_backward(softmax, target)
        return log_prob

    @staticmethod
    def backward(ctx, grad_lp):
        softmax, target = ctx.saved_tensors                   # softmax [L,V]
        grad = softmax.neg_()                                 # -softmax (reuse buffer)
        grad.scatter_add_(-1, target.unsqueeze(-1), torch.ones_like(grad[:, :1]))
        grad.mul_(grad_lp.unsqueeze(-1))
        return grad, None


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
        # Long-sequence memory knobs (default off -> unchanged smoke behavior):
        #   attention_backend="flash"  -> O(L) flash attn instead of O(L^2) local core
        #   log_probs_chunk_size>0     -> memory-efficient fused/chunked token log-prob
        self._attention_backend = str(ec.get("attention_backend") or "unfused").lower()
        self._logprob_chunk_size = int(ec.get("log_probs_chunk_size") or 0)
        # Dynamic-batch packing: concat multiple sequences into one varlen forward
        # (flash_attn_varlen + cu_seqlens) to keep GEMMs full on short sequences.
        self._dynamic_batch = bool(ec.get("enable_dynamic_batch") or False)
        self._max_tokens_per_gpu = int(ec.get("max_tokens_per_gpu") or 0)
        if self._dynamic_batch and self._attention_backend != "flash":
            # varlen packing requires the flash core; auto-enable it.
            self._attention_backend = "flash"

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
        # Activation recomputation: Megatron local-spec attention (no TE flash) keeps
        # the full O(seq^2) score matrix, so long-sequence training (resp=20480) OOMs
        # without recompute. Off by default (smoke, short seq); enable via megatron_cfg.
        recompute_kwargs: dict = {}
        rc_gran = ec.get("recompute_granularity") or None
        if rc_gran:
            recompute_kwargs["recompute_granularity"] = rc_gran
            recompute_kwargs["recompute_method"] = ec.get("recompute_method") or "uniform"
            recompute_kwargs["recompute_num_layers"] = int(ec.get("recompute_num_layers") or 1)
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
            **recompute_kwargs,
        )
        spec = get_gpt_layer_local_spec(qk_layernorm=True)
        spec.submodules.input_layernorm = WTN
        spec.submodules.pre_mlp_layernorm = WTN
        spec.submodules.self_attention.submodules.q_layernorm = WTN
        spec.submodules.self_attention.submodules.k_layernorm = WTN
        if self._attention_backend == "flash":
            # Swap the O(L^2) local core attention for flash-attn (O(L) memory).
            spec.submodules.self_attention.submodules.core_attention = FlashSelfAttentionCore
            logger.info("MegatronEngine[%d]: using flash-attention core (O(L) memory)", self._rank())

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

    def _forward_logits_packed(self, ids_list, model=None) -> tuple[torch.Tensor, list[int]]:
        """Packed varlen forward: concat ``ids_list`` (per-sequence 1D token tensors)
        into one [1,T] stream and run a single GPTModel forward with thd
        ``PackedSeqParams``. Returns ``(logits [T,V] float, offsets)`` where
        ``offsets[i]:offsets[i+1]`` slices sequence ``i``'s logits.

        Rotary is applied per-segment by Megatron's thd path (via cu_seqlens);
        attention is isolated per-segment by flash_attn_varlen. So each sequence's
        logits are identical to a standalone forward (up to bf16 nondeterminism)."""
        from megatron.core.packed_seq_params import PackedSeqParams
        m = model if model is not None else self.module
        lens = [int(t.numel()) for t in ids_list]
        offsets = [0]
        for L in lens:
            offsets.append(offsets[-1] + L)
        T = offsets[-1]
        tokens = torch.cat([t.view(-1) for t in ids_list], dim=0).view(1, T)
        # per-segment position ids (0..L_i-1); ignored by thd rotary but kept correct.
        pos = torch.cat([torch.arange(L, device=tokens.device) for L in lens], dim=0).view(1, T)
        cu = torch.tensor(offsets, dtype=torch.int32, device=tokens.device)
        max_seqlen = max(lens) if lens else 0
        pp = PackedSeqParams(
            cu_seqlens_q=cu, cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen, qkv_format="thd",
        )
        out = m(input_ids=tokens, position_ids=pos, attention_mask=None, packed_seq_params=pp)
        logits = out.logits if hasattr(out, "logits") else out
        return logits.view(T, -1).float(), offsets

    def _build_bins(self, lengths: list[int], budget: int) -> list[list[int]]:
        """Greedy bin-packing of row indices into groups whose summed token length
        stays <= ``budget`` (a row longer than budget forms its own bin)."""
        if budget <= 0:
            budget = max(lengths) if lengths else 1
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        bins: list[list[int]] = []
        bin_tokens: list[int] = []
        for i in order:
            Li = lengths[i]
            placed = False
            for b in range(len(bins)):
                if bin_tokens[b] + Li <= budget:
                    bins[b].append(i)
                    bin_tokens[b] += Li
                    placed = True
                    break
            if not placed:
                bins.append([i])
                bin_tokens.append(Li)
        return bins

    @staticmethod
    def _real_block(mask_row: torch.Tensor) -> tuple[int, int]:
        idx = mask_row.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 0, 0
        return int(idx[0].item()), int(idx.numel())

    # ---- memory-efficient log-prob helpers (see FlashSelfAttentionCore/#2) ----
    def _token_logprob_train(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-token log-prob with grad. Uses the fused single-buffer CE (optionally
        chunked over the sequence) when ``log_probs_chunk_size>0``; otherwise the
        original ``log_softmax(...).gather(...)`` path (kept for the smoke config)."""
        cs = self._logprob_chunk_size
        if cs and cs > 0:
            outs = []
            for s in range(0, logits.shape[0], cs):
                outs.append(_FusedTokenLogProb.apply(logits[s:s + cs], targets[s:s + cs]))
            return torch.cat(outs, dim=0)
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, targets.view(-1, 1)).squeeze(-1)

    def _logprob_entropy_nograd(
        self, logits: torch.Tensor, targets: torch.Tensor, want_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """No-grad per-token log-prob (+ optional entropy), chunked over the
        sequence to bound the ``[chunk, V]`` softmax memory."""
        cs = self._logprob_chunk_size if (self._logprob_chunk_size and self._logprob_chunk_size > 0) else logits.shape[0]
        cs = max(1, cs)
        lps, ents = [], []
        for s in range(0, logits.shape[0], cs):
            lg = logits[s:s + cs]
            lsm = torch.log_softmax(lg, dim=-1)
            lps.append(lsm.gather(-1, targets[s:s + cs].view(-1, 1)).squeeze(-1))
            if want_entropy:
                ents.append(-(lsm.exp() * lsm).sum(-1))
        lp = torch.cat(lps, dim=0)
        ent = torch.cat(ents, dim=0) if want_entropy else None
        return lp, ent

    def _row_policy_loss(self, t, r, start, token_lp, algo_name, cfg_fn, bnt, dp):
        """DAPO/PG loss + PPO-KL metrics for one sequence, given its (grad-carrying)
        per-token log-prob ``token_lp`` [1, Lm]. Returns ``(loss_tensor|None, stats|None)``.
        Shared by the packed and per-row training paths."""
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
            return None, None
        resp_mask = _col("response_mask", shift=True)
        adv_t = t.get("advantages")
        if adv_t is None:
            return None, None
        if adv_t.dim() == 1:
            adv = adv_t[r].to(dev).view(1, 1).expand(1, Lm).float()
        else:
            adv = adv_t[r].to(dev)[start + 1:].reshape(1, -1).float()
        ris = _col("rollout_is_weights", shift=False)
        ref_lp0 = _col("ref_log_probs", shift=False)
        rlp0 = _col("rollout_log_probs", shift=False)

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
                float(cfg_fn("clip_ratio_low", 0.2)), float(cfg_fn("clip_ratio_high", 0.28)),
                mask=mask, clip_ratio_c=float(cfg_fn("clip_ratio_c", 0.0)),
                batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
            )
        else:
            loss = policy_gradient_loss(
                token_lp, old_lp, adv, float(cfg_fn("clip_ratio", 0.2)), mask=mask,
            )
        kl_c = float(cfg_fn("kl_coeff", 0.0))
        if kl_c > 0.0 and ref_lp is not None:
            loss = loss + kl_c * kl_penalty(token_lp, ref_lp, mask=mask)

        stats = {"loss": float(loss.detach()), "ppo_kl_sum": 0.0, "ppo_kl_tok": 0.0,
                 "rc_kl_sum": 0.0, "rc_kl_tok": 0.0}
        if mask is not None:
            with torch.no_grad():
                tok = float(mask.sum())
                stats["ppo_kl_sum"] = float(((old_lp - token_lp) * mask).sum())
                stats["ppo_kl_tok"] = tok
                if rlp is not None:
                    stats["rc_kl_sum"] = float(((rlp - token_lp) * mask).sum())
                    stats["rc_kl_tok"] = tok
        return loss, stats

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

        def _emit(r, start, L, seg_logits, ids_row):
            tok_lp, ent = self._logprob_entropy_nograd(seg_logits[:-1], ids_row[1:], want_ent)  # [L-1]
            lp_out[r, start:start + L - 1] = tok_lp.cpu()
            if want_ent and ent is not None:
                ent_out[r, start:start + L - 1] = ent.cpu()

        with torch.no_grad():
            rows = []
            for r in range(B):
                start, L = self._real_block(am[r])
                if L >= 2:
                    rows.append((r, start, L))
            if self._dynamic_batch:
                budget = self._max_tokens_per_gpu if self._max_tokens_per_gpu > 0 else 21504
                lengths = [L for (_, _, L) in rows]
                for bin_rows in self._build_bins(lengths, budget):
                    ids_list = [seqs[rows[j][0], rows[j][1]:rows[j][1] + rows[j][2]].to("cuda") for j in bin_rows]
                    logits_packed, offsets = self._forward_logits_packed(ids_list, model=self.module)
                    logits_packed = logits_packed / temperature
                    for k, j in enumerate(bin_rows):
                        r, start, L = rows[j]
                        _emit(r, start, L, logits_packed[offsets[k]:offsets[k + 1]], ids_list[k])
            else:
                for (r, start, L) in rows:
                    ids = seqs[r, start:start + L].to("cuda")
                    logits = self._forward_logits(ids) / temperature  # [L,V]
                    _emit(r, start, L, logits, ids)
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

        def _accum(stats):
            nonlocal loss_accum, ppo_kl_sum, ppo_kl_tok, rc_kl_sum, rc_kl_tok, n_rows
            loss_accum += stats["loss"]
            n_rows += 1
            ppo_kl_sum += stats["ppo_kl_sum"]
            ppo_kl_tok += stats["ppo_kl_tok"]
            rc_kl_sum += stats["rc_kl_sum"]
            rc_kl_tok += stats["rc_kl_tok"]

        # Collect valid (non-empty) rows.
        rows = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L >= 2:
                rows.append((r, start, L))

        if self._dynamic_batch:
            # ---- dynamic-batch packing: concat rows into varlen forwards ----
            budget = self._max_tokens_per_gpu if self._max_tokens_per_gpu > 0 else 21504
            lengths = [L for (_, _, L) in rows]
            for bin_rows in self._build_bins(lengths, budget):
                ids_list = [seqs[rows[j][0], rows[j][1]:rows[j][1] + rows[j][2]].to("cuda") for j in bin_rows]
                logits_packed, offsets = self._forward_logits_packed(ids_list, model=self._ddp)
                logits_packed = logits_packed / temperature  # [T,V] (grad)
                bin_loss = None
                for k, j in enumerate(bin_rows):
                    r, start, _L = rows[j]
                    seg = logits_packed[offsets[k]:offsets[k + 1]]           # [L,V]
                    token_lp = self._token_logprob_train(seg[:-1], ids_list[k][1:]).view(1, -1)
                    loss, stats = self._row_policy_loss(t, r, start, token_lp, algo_name, _cfg, bnt, dp)
                    if loss is None:
                        continue
                    bin_loss = loss if bin_loss is None else bin_loss + loss
                    _accum(stats)
                if bin_loss is not None:
                    bin_loss.backward()
        else:
            # ---- per-sequence forward (original path) ----
            for (r, start, L) in rows:
                ids = seqs[r, start:start + L].to("cuda")
                logits = self._forward_logits(ids, model=self._ddp) / temperature  # [L,V] (grad)
                token_lp = self._token_logprob_train(logits[:-1], ids[1:]).view(1, -1)  # [1,L-1]
                loss, stats = self._row_policy_loss(t, r, start, token_lp, algo_name, _cfg, bnt, dp)
                if loss is None:
                    continue
                loss.backward()
                _accum(stats)

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
