# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Megatron-Core training engine (TransformerEngine spec) for LumenRL.

Builds Qwen3 ``GPTModel`` with the **TransformerEngine layer spec** (fused
attention + fused LayerNorm-Linear) and Megatron-Core's native TP/PP/CP process
groups, pipeline schedule, distributed optimizer, and distributed checkpointing.
Context parallelism uses TE's packed ``thd`` zigzag path: every sequence is split
into ``2*CP`` chunks, each rank owns its symmetric pair, and response log-probs
are reconstructed with a differentiable CP all-reduce.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lumenrl.core.protocol import DataProto
from lumenrl.engine.training import dsv4_megatron_bridge as dsv4
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine
from lumenrl.engine.training.qwen3_megatron_bridge import (
    Qwen3Dims,
    hf_to_megatron,
    load_hf_safetensors,
    megatron_to_hf,
)
from lumenrl.engine.training.qwen3moe_megatron_bridge import (
    _expert_local_index,
    _non_expert_hf_to_megatron,
    build_moe_dims,
    hf_expert_fc1,
    hf_expert_fc2,
    megatron_to_hf_moe,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "INFO"))


def _mem_diag_on() -> bool:
    return os.environ.get("LUMENRL_MEM_DIAG") == "1"


def _mem_diag_begin(phase: str, mbs: list) -> None:
    """Per-rank memory and microbatch shape at the start of a phase.

    Exists because a standalone probe and the trainer disagreed by ~89 GB per
    GPU on the same model, parallelism and sequence length, and the difference
    has to be either the microbatch shape or something the trainer holds that
    the probe does not. rocm-smi cannot tell those apart -- it sees the caching
    allocator's pool -- so the answer needs allocated and reserved side by side,
    from inside the process, on every rank rather than just rank 0.

    ``print`` rather than ``logger``: the two sides of that comparison run in
    different worlds. The trainer configures logging, a probe under plain
    torchrun does not, so an INFO record there reaches the level-WARNING handler
    of last resort and vanishes -- silently dropping exactly the half being
    compared. Ray forwards actor stdout to the driver, so print reaches both.
    """
    if not _mem_diag_on():
        return
    lens = [int(x.numel()) for mb in mbs for x in mb["ids_list"]]
    rows = sum(len(mb["rows"]) for mb in mbs)
    rank = dist.get_rank() if dist.is_initialized() else 0
    shown = lens if len(lens) <= 8 else f"{lens[:8]}...(+{len(lens) - 8})"
    print(
        f"MEMDIAG[rank {rank}] {phase} begin: {len(mbs)} microbatches, {rows} rows, "
        f"{sum(lens)} tokens, token lengths {shown} | "
        f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
        f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()


def _mem_diag_end(phase: str) -> None:
    if not _mem_diag_on():
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    print(
        f"MEMDIAG[rank {rank}] {phase} end: "
        f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
        f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB "
        f"of {total / 2**30:.0f}",
        flush=True,
    )


def _mem_diag_note(phase: str) -> None:
    """Memory at a point that is not a forward/backward phase."""
    if not _mem_diag_on():
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"MEMDIAG[rank {rank}] {phase}: "
        f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
        f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB",
        flush=True,
    )


def _mem_diag_stream(phase: str, named):
    """Pass ``named`` through, reporting the total streamed and the peak.

    The two numbers together are the point: bytes streamed is the size of the
    reconstructed model, and the peak says how much of it was resident at once.
    When those are close the gather is materializing the model; when the peak
    stays near its starting value the gather is streaming.
    """
    if not _mem_diag_on():
        yield from named
        return
    total = count = 0
    for name, t in named:
        total += t.numel() * t.element_size()
        count += 1
        yield name, t
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"MEMDIAG[rank {rank}] {phase}: streamed {count} tensors / "
        f"{total / 2**30:.1f} GiB | "
        f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
        f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB",
        flush=True,
    )

import re  # noqa: E402

# ``decoder.layers.{N}.`` -- state_dict keys use LOCAL (per-stage) layer numbers
# under pipeline parallelism; the HF-converted ``meg_full`` uses GLOBAL numbers.
_LAYER_RE = re.compile(r"(decoder\.layers\.)(\d+)(\.)")


def _pp_layer_offset(module) -> int:
    """Local->global layer offset for this pipeline stage, from the layers themselves.

    ``TransformerLayer.layer_number`` is the GLOBAL 1-based index -- it is built
    as ``local + get_transformer_layer_offset(...)`` -- and ``decoder.layers``
    holds only this stage's layers, so the offset is the first one minus one.
    Both name sources this offset is applied to (``named_parameters()`` and the
    ``sharded_state_dict()`` DICT keys) use the local index, so they agree.

    ⚠️ This used to be read out of checkpoint metadata instead: find a per-layer
    ``ShardedTensor`` with ``prepend_axis_num >= 1`` and subtract the layer index
    in its dict key from ``global_offset[0]``. That is only correct for a
    HOMOGENEOUS block. ``TransformerBlock.sharded_state_dict`` takes a separate
    branch when the layers differ -- which DSv4 hits, and so does anything with a
    ``moe_layer_freq`` list or ``heterogeneous_block_specs`` -- and that branch
    passes ``sharded_pp_offset = []``, putting the global index in the tensor's
    ``key`` rather than in an offset axis. ``prepend_axis_num`` is then 0, no
    candidate is ever found, and the offset came back 0 with no error.

    At PP=1 the answer is 0 either way, which is why every run so far hid it. At
    PP=2 the later stages republished their params under the FIRST stage's layer
    numbers, so the names collided and the rollout weight sync shipped half the
    model: ``probe_68 --pp 2`` reported 1592 missing / 14 extra tensors, and vLLM
    died on ``KeyError: 'layers.0.attn.compressor.ape'`` -- layer 0 has no
    compressor, layer 2 does.
    """
    layers = getattr(getattr(module, "decoder", None), "layers", None)
    if not layers:
        return 0
    return int(layers[0].layer_number) - 1


def _to_global_key(key: str, offset: int) -> str:
    """Rewrite a local-stage ``decoder.layers.{local}`` key to its global index."""
    if offset == 0:
        return key
    m = _LAYER_RE.search(key)
    if not m:
        return key
    return f"{key[:m.start(2)]}{int(m.group(2)) + offset}{key[m.end(2):]}"


class MegatronNativeEngine(MegatronBaseEngine):
    """Megatron-Core GPTModel engine using the TransformerEngine layer spec.

    Inherits forward / log-prob / packing / optimizer-step helpers from
    ``MegatronBaseEngine``; only the model construction (TE spec, pipeline-stage
    aware) and HF weight I/O (TE-named bridge) differ.
    """

    # DeepSeek-V4 takes a separate construction and weight path; see
    # ``dsv4_megatron_bridge``. Class-level so the forward-side dispatch is
    # answerable before ``initialize`` has run.
    _is_dsv4 = False
    _dsv4_align = 1

    def initialize(self) -> None:
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_block_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig

        from lumenrl.engine.training.megatron_te_gemm_compat import (
            install as install_te_gemm_compat,
        )

        # No-op unless megatron-core and TransformerEngine disagree about the
        # general_gemm signature, which the MoE router path would otherwise hit.
        install_te_gemm_compat()

        ec = self.engine_config
        tp = int(ec.get("tensor_model_parallel_size", 1))
        pp = int(ec.get("pipeline_model_parallel_size", 1))
        cp = int(ec.get("context_parallel_size", 1))
        ep = int(ec.get("expert_model_parallel_size", 1))
        etp = int(ec.get("expert_tensor_parallel_size") or tp)
        seed = int(ec.get("seed", 42))
        # Sequence parallelism shards activations along the sequence dim, which
        # requires seq_len % tp == 0. RL forwards are variable-length (per-seq or
        # packed thd), so SP is OFF by default; plain TP (all-reduce, full seq on
        # every TP rank) has no length constraint. Opt in via megatron_cfg only if
        # every forward length is a multiple of TP.
        sp = bool(ec.get("sequence_parallel", False))
        self._tp = tp
        self._pp = pp
        self._cp = cp
        self._ep = ep
        self._etp = etp
        self._sp = sp
        # Forward-side knobs shared with the base log-prob helpers.
        self._logprob_chunk_size = int(ec.get("log_probs_chunk_size") or 0)
        self._dynamic_batch = bool(ec.get("enable_dynamic_batch") or False)
        self._max_tokens_per_gpu = int(ec.get("max_tokens_per_gpu") or 0)

        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                expert_model_parallel_size=ep,
                expert_tensor_parallel_size=etp,
            )
        model_parallel_cuda_manual_seed(seed)

        # CP forwards always use packed ``thd`` inputs. The model receives each
        # rank's zigzag token pair and TE performs the cross-rank attention.
        self._is_first_stage = mpu.is_pipeline_first_stage()
        self._is_last_stage = mpu.is_pipeline_last_stage()

        # ---- HF config -> Qwen3 dims / TransformerConfig (TE-compatible) ----
        cfg_path = os.path.join(self.model_name, "config.json")
        with open(cfg_path) as fh:
            hf = json.load(fh)
        head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])

        # DeepSeek-V4 is the one family the inline config below cannot describe:
        # MLA head geometry, a 4-D hyper-connection residual stream, per-layer
        # heterogeneous attention, and hash routing on the first layers. It also
        # ships block-quantized FP8 weights that the HF-safetensors bridge cannot
        # read. Everything DSv4-specific lives in ``dsv4_megatron_bridge``.
        self._is_dsv4 = dsv4.is_dsv4(hf)
        if self._is_dsv4 and self._dynamic_batch:
            # See ``_dsv4_check_topology``: DSv4 attention derives token positions
            # from the tensor length alone, so bin-packing several sequences into
            # one microbatch would have them read as one long sequence.
            logger.warning(
                "MegatronNativeEngine: DSv4 -> forcing enable_dynamic_batch=False "
                "(its attention cannot consume cu_seqlens; see _dsv4_check_topology)."
            )
            self._dynamic_batch = False

        # ---- MoE detection: HF config declares routed experts (Qwen3-MoE etc.) ----
        # Any of these HF keys marks a MoE model; an explicit engine_config
        # ``num_experts`` overrides. dense models keep every ``moe_*`` off.
        hf_num_experts = (
            hf.get("num_experts") or hf.get("n_routed_experts") or hf.get("num_local_experts")
        )
        cfg_num_experts = ec.get("num_experts")
        num_experts = int(cfg_num_experts or hf_num_experts or 0)
        self._is_moe = num_experts > 1
        self._num_experts = num_experts
        # R3 routing replay (opt-in): record router logits in the old-logprob
        # forward, replay in the update. Only meaningful for MoE.
        self._r3_enabled = bool(ec.get("r3_enabled", False)) and self._is_moe
        self._r3_store: dict[int, list] = {}

        # Megatron hard-requires sequence parallelism when MoE and TP are both on
        # (the MoE token dispatcher assumes SP-scattered activations under TP).
        # RL forwards are variable-length, so the packed thd stream is padded to a
        # multiple of TP in ``_pp_forward_model`` to satisfy the SP sequence split.
        if self._is_moe and tp > 1 and not sp:
            sp = True
            self._sp = True
            logger.info(
                "MegatronNativeEngine: MoE+TP(%d) -> forcing sequence_parallel=True "
                "(packed thd padded to multiple of TP).", tp,
            )

        moe_kwargs: dict = {}
        if self._is_dsv4:
            # ``build_moe_dims`` and the ``Qwen3Dims`` twin below both read HF keys
            # DSv4 does not have. Its dims are only consumed by the HF weight
            # bridge (load + rollout weight sync), and DSv4 uses neither yet.
            self._dims = None
        elif self._is_moe:
            self._dims = build_moe_dims(hf)
            moe_ffn = self._dims.moe_ffn
            topk = int(ec.get("moe_router_topk") or hf.get("num_experts_per_tok") or 2)
            shared_ffn = int(
                ec.get("moe_shared_expert_intermediate_size")
                or hf.get("shared_expert_intermediate_size", 0) or 0
            )
            moe_kwargs = dict(
                num_moe_experts=num_experts,
                moe_ffn_hidden_size=moe_ffn,
                moe_router_topk=topk,
                moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", True)),
                moe_router_load_balancing_type=str(ec.get("moe_router_load_balancing_type", "aux_loss")),
                moe_aux_loss_coeff=float(ec.get("moe_aux_loss_coeff", 0.0) or 0.0),
                expert_model_parallel_size=ep,
                expert_tensor_parallel_size=etp,
                moe_permute_fusion=bool(ec.get("moe_permute_fusion", False)),
            )
            # Qwen3-MoE routing = softmax(all) -> top-k -> renormalize top-k
            # (HF ``norm_topk_prob=True``). That is mathematically identical to
            # Megatron's ``moe_router_pre_softmax=False`` (top-k of logits, then a
            # softmax over ONLY the top-k logits -> already sums to 1), because the
            # full-softmax denominator cancels under renormalization. Using
            # ``pre_softmax=True`` instead would leave the gate weights un-renormalized
            # (sum<1) and diverge from vLLM -> large rollout/train log-prob mismatch.
            pre_softmax = ec.get("moe_router_pre_softmax")
            moe_kwargs["moe_router_pre_softmax"] = (
                False if pre_softmax is None else bool(pre_softmax)
            )
            if ec.get("moe_router_score_function"):
                moe_kwargs["moe_router_score_function"] = str(ec.get("moe_router_score_function"))
            if ec.get("moe_router_dtype"):
                moe_kwargs["moe_router_dtype"] = str(ec.get("moe_router_dtype"))
            if ec.get("moe_router_topk_scaling_factor") is not None:
                moe_kwargs["moe_router_topk_scaling_factor"] = float(ec.get("moe_router_topk_scaling_factor"))
            if ec.get("moe_router_bias_update_rate") is not None:
                moe_kwargs["moe_router_bias_update_rate"] = float(ec.get("moe_router_bias_update_rate"))
            if shared_ffn > 0:
                moe_kwargs["moe_shared_expert_intermediate_size"] = shared_ffn
        else:
            self._dims = Qwen3Dims(
                num_layers=hf["num_hidden_layers"], hidden=hf["hidden_size"],
                num_heads=hf["num_attention_heads"], num_kv_groups=hf["num_key_value_heads"],
                head_dim=head_dim, ffn=hf["intermediate_size"], vocab=hf["vocab_size"],
            )

        recompute_kwargs: dict = {}
        rc_gran = ec.get("recompute_granularity") or None
        if rc_gran:
            recompute_kwargs["recompute_granularity"] = rc_gran
            recompute_kwargs["recompute_method"] = ec.get("recompute_method") or "uniform"
            recompute_kwargs["recompute_num_layers"] = int(ec.get("recompute_num_layers") or 1)

        if self._is_dsv4:
            # Field-for-field equal to what Megatron's own parser produces from
            # miles' deepseek-v4-flash.sh, which is the config every existing DSv4
            # numerical reference was measured on.
            det = ec.get("deterministic_mode")
            tfcfg = dsv4.build_dsv4_config(
                hf, ec, tp=tp, pp=pp, cp=cp, ep=ep, etp=etp, sp=sp,
                # Unset means "the model family decides", and DSv4 decides on:
                # non-deterministic forwards disagree with themselves on ~1.6% of
                # argmaxes, swamping the train/rollout gap DAPO measures.
                deterministic=True if det is None else bool(det),
            )
            if tfcfg.deterministic_mode:
                dsv4.enable_deterministic_mode()
            self._dsv4_align = dsv4.sequence_alignment(tfcfg)
        else:
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
                context_parallel_size=cp, sequence_parallel=sp,
                use_cpu_initialization=True,
                # PP: RL microbatches are variable-length (per-seq / packed thd), so
                # the pipeline P2P must exchange tensor shapes dynamically instead of
                # assuming a fixed [seq, mbs, hidden]. (alltoall MoE dispatcher just
                # passes the dense-model config validation that rejects the default
                # allgather dispatcher under variable_seq_lengths.)
                variable_seq_lengths=(pp > 1),
                moe_token_dispatcher_type="alltoall",
                **moe_kwargs,
                **recompute_kwargs,
            )

        if self._is_dsv4:
            # Heterogeneous per layer (sliding / compressed+indexed /
            # hyper-compressed), so no block-spec builder can produce it.
            spec = dsv4.build_dsv4_spec(
                tfcfg, dsa_topk_backend=str(ec.get("dsa_topk_backend", "torch")),
            )
        elif self._is_moe:
            # MoE: get_gpt_decoder_block_spec builds per-layer specs with the routed
            # expert MLP (grouped-GEMM TEGroupedLinear or sequential local experts)
            # and standalone pre-MLP RMSNorm, driven by the TransformerConfig MoE
            # fields above. Attention keeps the fused TE qkv layer-norm-linear.
            spec = get_gpt_decoder_block_spec(tfcfg, use_transformer_engine=True)
        else:
            # TransformerEngine spec: fused TELayerNormColumnParallelLinear +
            # TEDotProductAttention (CK/aotriton fused attn), TE RMSNorm.
            spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)

        model = GPTModel(
            config=tfcfg, transformer_layer_spec=spec, vocab_size=hf["vocab_size"],
            max_sequence_length=hf.get("max_position_embeddings", 32768),
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
            position_embedding_type="rope",
            rotary_base=hf.get("rope_theta", 1000000.0),
            share_embeddings_and_output_weights=bool(hf.get("tie_word_embeddings", False)),
            # Gather the TP-sharded vocab logits back to the full vocab on every TP
            # rank so we can reuse the parent's full-vocab log-prob/entropy path
            # unchanged. (parallel_output=True would return [s,b,V/tp] and require a
            # vocab-parallel log-prob; that optimization is phase 2a-optional.)
            parallel_output=False,
        )

        if self._is_dsv4:
            # DSv4 weights come from a torch_dist checkpoint converted offline
            # (native FP8 -> bf16 HF -> torch_dist), because the released
            # checkpoint is block-quantized FP8. ``dist_checkpointing.load``
            # reshards it to this rank's TP/PP/EP on the way in.
            ckpt = str(ec.get("dist_checkpoint_path") or self.model_name)
            dsv4.materialize_dsv4(model)
            report = dsv4.load_dsv4_dist_checkpoint(model, ckpt)
            self.module = model
            logger.info(
                "MegatronNativeEngine[%d]: DSv4 loaded %.2fB params (%d tensors) from %s "
                "| experts=%d topk=%d hash_layers=%d compress_ratios=%s hc_mult=%d "
                "| tp=%d pp=%d cp=%d ep=%d etp=%d deterministic=%s",
                self._rank(), report["num_params"] / 1e9, report["num_tensors"],
                report["path"], tfcfg.num_moe_experts, tfcfg.moe_router_topk,
                tfcfg.dsv4_n_hash_layers, tfcfg.dsv4_compress_ratios, tfcfg.dsv4_hc_mult,
                tp, pp, cp, ep, etp, tfcfg.deterministic_mode,
            )
        else:
            # ---- load HF weights via the TE-named bridge ----
            logger.info(
                "MegatronNativeEngine[%d]: loading HF weights (TE spec, moe=%s experts=%d "
                "tp=%d pp=%d cp=%d ep=%d etp=%d) from %s",
                self._rank(), self._is_moe, num_experts, tp, pp, cp, ep, etp, self.model_name,
            )
            hf_state = load_hf_safetensors(self.model_name)
            if self._is_moe:
                meg_state = self._shard_hf_for_moe(model, hf_state)
            elif tp == 1 and pp == 1:
                meg_state = hf_to_megatron(hf_state, self._dims, te=True)
            else:
                # TP/PP>1: each rank keeps only its model-parallel shard. For TP the
                # tensor is sliced along the sharded axis; for PP only this stage's
                # layers are kept (the local key's global layer index comes from the
                # ShardedTensor's global_offset). See ``_shard_hf_for_mp``.
                meg_state = self._shard_hf_for_mp(model, hf_state)
            del hf_state
            missing = model.load_state_dict(meg_state, strict=False)
            real_missing = [k for k in missing.missing_keys if "_extra_state" not in k]
            if real_missing:
                raise RuntimeError(f"Native(TE) load missing keys: {real_missing[:6]} ...")
            del meg_state
            self.module = model.cuda()
        self._tfcfg = tfcfg

        if self._is_moe and not self._is_dsv4 and self._rank() == 0:
            # Surface MoE + Expert-Parallel topology to the run log (stdout is
            # forwarded by Ray). Evidence of expert sharding / EP group width.
            print(
                f"[MegatronNativeEngine] MoE+EP spec: num_experts={num_experts} "
                f"topk={moe_kwargs.get('moe_router_topk')} moe_ffn={self._dims.moe_ffn} | "
                f"tp={tp} pp={pp} cp={cp} EP={ep} etp={etp} -> "
                f"local_experts/rank={num_experts // ep} | "
                f"grouped_gemm={moe_kwargs.get('moe_grouped_gemm')} "
                f"router_dtype={moe_kwargs.get('moe_router_dtype')} "
                f"pre_softmax={moe_kwargs.get('moe_router_pre_softmax')} "
                f"aux_loss_coeff={moe_kwargs.get('moe_aux_loss_coeff')}",
                flush=True,
            )

        # An engine that only ever runs forwards (bring-up smoke test, or a
        # frozen reference policy) can skip the DDP wrapper and the distributed
        # optimizer. On the 27B DSv4 slice that is the difference between the
        # bf16 weights alone and another ~330 GB of fp32 master + Adam moments,
        # i.e. between fitting on one GPU and not.
        if not bool(ec.get("build_optimizer", True)):
            self._ddp = None
            self.optimizer = None
            self.lr_scheduler = None
            logger.info(
                "MegatronNativeEngine[%d]: forward-only (build_optimizer=False), "
                "%d params, no DDP wrapper and no optimizer.",
                self._rank(), sum(p.numel() for p in self.module.parameters()),
            )
            return

        # ---- distributed optimizer + scheduler ----
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        oc = self.optimizer_config
        self._clip = float(oc.get("clip_grad", 1.0))
        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False,
            use_distributed_optimizer=True, average_in_collective=True, bucket_size=None,
        )
        self._ddp = DDP(config=tfcfg, ddp_config=ddp_cfg, module=self.module)

        opt_kwargs = {}
        if bool(ec.get("optimizer_cpu_offload", False)):
            # Megatron routes this to HybridDeviceOptimizer, which keeps
            # ``offload_fraction`` of the master weights and Adam moments in
            # pinned host RAM and steps them with torch's AdamW on the CPU.
            #
            # use_precision_aware_optimizer is not a precision choice here: the
            # offload path reuses that flag's code path and Megatron asserts on
            # it. All four state dtypes stay fp32.
            frac = float(ec.get("optimizer_offload_fraction", 1.0))
            opt_kwargs.update(
                optimizer_cpu_offload=True,
                use_precision_aware_optimizer=True,
                optimizer_offload_fraction=frac,
                overlap_cpu_optimizer_d2h_h2d=bool(
                    ec.get("overlap_cpu_optimizer_d2h_h2d", True)
                ),
            )
            if self._rank() == 0:
                logger.info(
                    "MegatronNativeEngine: optimizer CPU offload on, fraction=%.2f "
                    "(that share of the 12 bytes/param of master weights and Adam "
                    "moments lives in host RAM, and its step runs on the CPU)", frac,
                )

        opt_cfg = OptimizerConfig(
            optimizer="adam", lr=float(oc.get("lr", 1e-6)),
            weight_decay=float(oc.get("weight_decay", 0.1)),
            adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
            clip_grad=self._clip, bf16=True, fp16=False,
            params_dtype=torch.bfloat16, use_distributed_optimizer=True,
            **opt_kwargs,
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
                "MegatronNativeEngine: TE model+distributed-optimizer ready, %d params, dp_size=%d",
                n, self.get_data_parallel_size(),
            )

    # ---- TP/PP weight loading: slice full HF weights into this rank's shard ----
    def _shard_hf_for_mp(self, model, hf_state: dict) -> dict:
        """Build this rank's model-parallel (TP+PP) Megatron state dict from HF.

        Converts the full HF weights to the (unsharded, global-layer-numbered)
        Megatron layout via the TE bridge, then for each param in the model's own
        ``sharded_state_dict``:
          * PP: the state_dict key uses **local** layer numbering, but the
            ShardedTensor's ``global_offset[0]`` is the **global** layer index;
            we map local->global to pick the right full tensor (only this stage's
            layers are present in the state dict, so other layers are skipped).
          * TP: slice along the sharded axis via ``global_slice()`` (matches
            Megatron's Column/Row/Vocab parallel layout). The fused SwiGLU
            ``linear_fc1`` is a ``ShardedTensorFactory`` (gate/up each
            column-parallel, concatenated) handled explicitly.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.mapping import (
            ShardedTensor,
            ShardedTensorFactory,
        )

        meg_full = hf_to_megatron(hf_state, self._dims, te=True)
        tp = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        ffn = self._dims.ffn
        ssd = model.sharded_state_dict()
        offset = _pp_layer_offset(model)

        local_sd: dict = {}
        for key, st in ssd.items():
            if key.endswith("_extra_state"):
                continue
            gkey = _to_global_key(key, offset)
            if gkey not in meg_full:
                continue
            full = meg_full[gkey]
            if isinstance(st, ShardedTensorFactory):
                # fused SwiGLU fc1: full is [gate(ffn); up(ffn)] along dim 0, each
                # column(=output)-parallel -> rank r gets its 1/tp chunk of gate and
                # of up, concatenated (matches Megatron's gate/up sharding order).
                assert key.endswith("mlp.linear_fc1.weight"), f"unexpected factory: {key}"
                shard = ffn // tp
                gate = full[:ffn]
                up = full[ffn:]
                local = torch.cat(
                    [gate[tp_rank * shard:(tp_rank + 1) * shard],
                     up[tp_rank * shard:(tp_rank + 1) * shard]], dim=0,
                ).contiguous()
            elif isinstance(st, ShardedTensor):
                # global_slice() carries prepend axes (layer index) first; the
                # trailing entries index the real tensor dims of ``full``.
                sl = st.global_slice()[st.prepend_axis_num:]
                local = full[sl].contiguous()
            else:
                continue
            local_sd[key] = local.to(torch.bfloat16)
        del meg_full
        return local_sd

    # ---- MoE weight loading: HF -> this (TP+PP+EP+ETP) rank's shard ----
    def _shard_hf_for_moe(self, model, hf_state: dict) -> dict:
        """Build this rank's MoE Megatron state dict from full HF weights.

        Non-expert params (embedding, attention, norms, router, output) are TP/PP
        sharded exactly like the dense path (plain ShardedTensor ``global_slice``;
        MoE has no dense-MLP fused ``fc1`` factory). Expert params are selected by
        Expert-Parallel rank -- each EP rank owns ``E / ep`` consecutive global
        experts -- and, when Expert-Tensor-Parallel (ETP) > 1, additionally sliced:
        ``linear_fc1`` (fused gate;up) column-parallel, ``linear_fc2`` row-parallel.
        Handles both grouped-GEMM (``experts.linear_fc{1,2}.weight{e}``) and
        sequential (``experts.local_experts.{e}.linear_fc{1,2}.weight``) naming.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        d = self._dims
        ep = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp = mpu.get_expert_tensor_parallel_world_size()
        etp_rank = mpu.get_expert_tensor_parallel_rank()
        tp = mpu.get_tensor_model_parallel_world_size()
        num_local = d.num_experts // ep
        if num_local * ep != d.num_experts:
            raise RuntimeError(
                f"num_experts={d.num_experts} not divisible by expert_model_parallel_size={ep}"
            )
        moe_ffn = d.moe_ffn

        non_expert_full = _non_expert_hf_to_megatron(hf_state, d)
        ssd = model.sharded_state_dict()
        offset = _pp_layer_offset(model)

        local_sd: dict = {}
        for name, p in model.named_parameters():
            exp = _expert_local_index(name)
            if exp is not None:
                local_e, which_fc = exp
                m = _LAYER_RE.search(name)
                global_layer = int(m.group(2)) + offset
                global_e = ep_rank * num_local + local_e
                if which_fc == "1":
                    full = hf_expert_fc1(hf_state, d, global_layer, global_e)  # [2*moe_ffn, hidden]
                    if etp > 1:
                        shard = moe_ffn // etp
                        gate = full[:moe_ffn]
                        up = full[moe_ffn:]
                        full = torch.cat(
                            [gate[etp_rank * shard:(etp_rank + 1) * shard],
                             up[etp_rank * shard:(etp_rank + 1) * shard]], dim=0,
                        ).contiguous()
                else:
                    full = hf_expert_fc2(hf_state, d, global_layer, global_e)  # [hidden, moe_ffn]
                    if etp > 1:
                        shard = moe_ffn // etp
                        full = full[:, etp_rank * shard:(etp_rank + 1) * shard].contiguous()
                local_sd[name] = full.to(torch.bfloat16)
                continue

            # non-expert param: TP/PP slice via the ShardedTensor metadata.
            gkey = _to_global_key(name, offset)
            if gkey not in non_expert_full:
                continue
            full = non_expert_full[gkey]
            if tp > 1:
                st = ssd.get(name)
                if isinstance(st, ShardedTensor):
                    sl = st.global_slice()[st.prepend_axis_num:]
                    full = full[sl]
            local_sd[name] = full.contiguous().to(torch.bfloat16)
        del non_expert_full
        return local_sd

    # ---- weight sync: Megatron(TE) -> full (TP+PP gathered) named tensors ----
    def _tp_gather_named_params(self) -> dict:
        """All-gather this stage's params across the TP group -> full (unsharded-TP)
        tensors keyed by GLOBAL layer name. Inverse of the TP part of
        ``_shard_hf_for_mp`` (fc1 gate/up handled explicitly)."""
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.mapping import (
            ShardedTensor,
            ShardedTensorFactory,
        )

        tp = mpu.get_tensor_model_parallel_world_size()
        ssd = self.module.sharded_state_dict()
        offset = _pp_layer_offset(self.module)
        ffn = self._dims.ffn
        out: dict = {}
        group = mpu.get_tensor_model_parallel_group() if tp > 1 else None
        for name, p in self.module.named_parameters():
            p = p.detach().contiguous()
            gkey = _to_global_key(name, offset)
            if tp == 1:
                out[gkey] = p
                continue
            gathered = [torch.empty_like(p) for _ in range(tp)]
            dist.all_gather(gathered, p, group=group)
            st = ssd.get(name)
            if isinstance(st, ShardedTensorFactory):
                shard = ffn // tp
                gate = torch.cat([g[:shard] for g in gathered], dim=0)
                up = torch.cat([g[shard:] for g in gathered], dim=0)
                full = torch.cat([gate, up], dim=0)
            elif isinstance(st, ShardedTensor):
                gshape = tuple(st.global_shape[st.prepend_axis_num:])
                lshape = tuple(st.local_shape)
                split_dim = next(
                    (d for d in range(len(lshape)) if lshape[d] != gshape[d]), None
                )
                full = gathered[0] if split_dim is None else torch.cat(gathered, dim=split_dim)
            else:
                full = gathered[0]
            out[gkey] = full
        return out

    def _full_megatron_named_params(self):
        """Reconstruct the COMPLETE model as (global_name, tensor) on every rank.

        Each colocated rollout replica is TP=1/PP=1 and needs the whole model, so
        a TP/PP>1 actor rank must gather across both axes: first all-gather within
        the TP group (``_tp_gather_named_params``), then broadcast each stage's
        (disjoint) params across the PP group so every rank ends up holding all
        layers. Must be called concurrently on all actors (collective)."""
        from megatron.core import parallel_state as mpu

        stage_params = self._tp_gather_named_params()
        pp = mpu.get_pipeline_model_parallel_world_size()
        if pp == 1:
            return list(stage_params.items())

        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # exchange (key, shape, dtype) metadata so every rank joins each broadcast.
        # The dtype travels as a torch.dtype: all_gather_object pickles, and torch
        # dtypes pickle to the same singletons, so there is no name-keyed map that
        # can be missing an entry. See the MoE twin of this loop.
        meta_local = [(k, tuple(v.shape), v.dtype) for k, v in stage_params.items()]
        gathered_meta: list = [None] * pp
        dist.all_gather_object(gathered_meta, meta_local, group=pp_group)
        out: dict = {}
        for src in range(pp):
            src_global = dist.get_global_rank(pp_group, src)
            for (k, shape, dtype) in gathered_meta[src]:
                if src == pp_rank:
                    t = stage_params[k].contiguous()
                else:
                    t = torch.empty(shape, dtype=dtype, device="cuda")
                dist.broadcast(t, src=src_global, group=pp_group)
                out[k] = t
        return list(out.items())

    def _moe_stage_named_params(self):
        """Yield this PP stage's params as (global_name, tensor), gathered.

        Non-expert params: TP all-gather (attention shards). Expert params:
        ETP-gather each local expert (fc1 gate/up column shards, fc2 row shards),
        then all-gather across the EP group with local->global expert relabel.
        Expert names carry GLOBAL indices so ``megatron_to_hf_moe`` maps them
        back to HF ``mlp.experts.{e}.*``.

        ⚠️ A generator, and every ``yield`` sits downstream of a collective, so
        the caller MUST drain it on every rank. Abandoning it part-way leaves
        the ranks that kept going waiting in an all-gather that never completes.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.mapping import (
            ShardedTensor,
            ShardedTensorFactory,
        )
        from lumenrl.engine.training.qwen3moe_megatron_bridge import _relabel_expert_index

        ep = mpu.get_expert_model_parallel_world_size()
        etp = mpu.get_expert_tensor_parallel_world_size()
        tp = mpu.get_tensor_model_parallel_world_size()
        # Not ``self._dims.num_experts``: DSv4 has no Qwen3-shaped dims, and the
        # two agree wherever both exist.
        num_local = self._num_experts // ep

        ssd = self.module.sharded_state_dict()
        offset = _pp_layer_offset(self.module)
        tp_group = mpu.get_tensor_model_parallel_group() if tp > 1 else None
        etp_group = mpu.get_expert_tensor_parallel_group() if etp > 1 else None
        ep_group = mpu.get_expert_model_parallel_group() if ep > 1 else None

        for name, param in self.module.named_parameters():
            p = param.detach().contiguous()
            exp = _expert_local_index(name)
            if exp is None:
                # ---- non-expert: TP all-gather ----
                gkey = _to_global_key(name, offset)
                if tp == 1:
                    yield gkey, p
                    continue
                gathered = [torch.empty_like(p) for _ in range(tp)]
                dist.all_gather(gathered, p, group=tp_group)
                st = ssd.get(name)
                if isinstance(st, ShardedTensorFactory):
                    # SwiGLU fc1: Megatron fuses gate and up along dim 0, so each
                    # rank holds [gate_shard ; up_shard] and a plain cat would
                    # interleave them into [gate0, up0, gate1, up1]. Separate the
                    # halves before concatenating. A factory is exactly how
                    # Megatron marks that the checkpoint layout differs from the
                    # runtime one, which is why it needs its own branch and why
                    # matching only ShardedTensor silently shipped gathered[0]:
                    # half-width shared-expert weights that vLLM rejected with
                    # "load_merged_column_weight ... start out of range".
                    #
                    # ``p.shape[0] // 2`` rather than the dense path's
                    # ``self._dims.ffn // tp``: same value, but DSv4 has no
                    # Qwen3-shaped dims and leaves ``_dims`` None.
                    sh = p.shape[0] // 2
                    gate = torch.cat([g[:sh] for g in gathered], dim=0)
                    up = torch.cat([g[sh:] for g in gathered], dim=0)
                    yield gkey, torch.cat([gate, up], dim=0)
                elif isinstance(st, ShardedTensor):
                    gshape = tuple(st.global_shape[st.prepend_axis_num:])
                    lshape = tuple(st.local_shape)
                    split_dim = next(
                        (dd for dd in range(len(lshape)) if lshape[dd] != gshape[dd]), None
                    )
                    # split_dim None means genuinely replicated across TP. That is
                    # common here and NOT a red flag: DSv4 replicates wq_a, wkv and
                    # the indexer projections while still carrying Megatron's
                    # ``tensor_model_parallel`` marker, so the marker cannot be used
                    # to decide this -- only the shard metadata can.
                    full = gathered[0] if split_dim is None else torch.cat(gathered, dim=split_dim)
                    assert tuple(full.shape) == gshape, (
                        f"TP gather rebuilt {name} as {tuple(full.shape)}, but its "
                        f"shard metadata says the full tensor is {gshape}"
                    )
                    yield gkey, full
                else:
                    yield gkey, gathered[0]
                continue

            # ---- expert: ETP gather -> full local expert tensor ----
            local_e, which_fc = exp
            if etp > 1:
                g = [torch.empty_like(p) for _ in range(etp)]
                dist.all_gather(g, p, group=etp_group)
                if which_fc == "1":
                    sh = p.shape[0] // 2
                    gate = torch.cat([x[:sh] for x in g], dim=0)
                    up = torch.cat([x[sh:] for x in g], dim=0)
                    p = torch.cat([gate, up], dim=0)
                else:
                    p = torch.cat(g, dim=1)
                del g
            # ---- EP all-gather -> relabel local->global expert index ----
            if ep == 1:
                yield _to_global_key(_relabel_expert_index(name, local_e), offset), p
                continue
            g = [torch.empty_like(p) for _ in range(ep)]
            dist.all_gather(g, p, group=ep_group)
            for j in range(ep):
                global_e = j * num_local + local_e
                gname = _to_global_key(_relabel_expert_index(name, global_e), offset)
                yield gname, g[j]
                # The consumer has taken its copy. Dropping the reference here is
                # what keeps the resident set at one all-gather instead of the
                # whole model: with EP=32 this list alone is 32x the local shard.
                g[j] = None

    def _full_megatron_named_params_moe(self):
        """Reconstruct the COMPLETE MoE model as (global_name, tensor) on every rank.

        Streams rather than accumulates. That is not a refactor: the eager
        version held the entire reconstructed model in GPU memory on every rank
        before the first byte was sent -- 51.0 GiB measured on the 27.39B
        4-layer slice, and ~529 GiB per rank for the full 43-layer model, which
        no 288 GB card can hold. It is what put the trainer ~89 GB above a
        standalone probe doing the same step, because a probe never syncs
        weights. See the handoff's "the 89 GB" section.

        ⚠️ Collective, and lazily so: see ``_moe_stage_named_params``. Drain it.
        """
        from megatron.core import parallel_state as mpu

        pp = mpu.get_pipeline_model_parallel_world_size()
        if pp == 1:
            yield from self._moe_stage_named_params()
            return

        # ---- PP broadcast: every rank ends up with all stages' params ----
        # Still materializes the local stage, because a rank has to publish the
        # shapes it owns before any rank can join the broadcasts.
        stage: dict = dict(self._moe_stage_named_params())
        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # The dtype travels as a torch.dtype, not as its name: all_gather_object
        # pickles, and torch dtypes pickle to the same singletons. A name-keyed
        # map here used to cover only the three float types and died with
        # KeyError: 'torch.int32' on the first PP=2 weight sync -- a failure mode
        # that only exists if the map can be incomplete.
        meta_local = [(k, tuple(v.shape), v.dtype) for k, v in stage.items()]
        gathered_meta: list = [None] * pp
        dist.all_gather_object(gathered_meta, meta_local, group=pp_group)
        for src in range(pp):
            src_global = dist.get_global_rank(pp_group, src)
            for (k, shape, dtype) in gathered_meta[src]:
                if src == pp_rank:
                    t = stage.pop(k).contiguous()
                else:
                    t = torch.empty(shape, dtype=dtype, device="cuda")
                dist.broadcast(t, src=src_global, group=pp_group)
                yield k, t

    def _dsv4_router_bias_buffers(self):
        """The aux-loss-free load-balancing bias, which is a buffer, not a param.

        ``moe_router_enable_expert_bias`` updates it every step from the observed
        expert load, and the rollout's top-k uses it. Left out of the weight sync
        it would drift away from the trainer's routing, which is precisely the
        divergence the importance ratio is supposed to be measuring. It is
        replicated (full ``[num_experts]`` on every EP rank), so no EP gather --
        but it does need the same PP broadcast the params get, because a stage
        holds only its own layers' buffers. Under PP=1 the local stage IS the
        whole model, which is why this was a list comprehension and why the gap
        only showed up as one missing tensor at PP=2
        (``layers.3.ffn.gate.bias``).
        """
        from megatron.core import parallel_state as mpu

        offset = _pp_layer_offset(self.module)
        local = [
            (_to_global_key(name, offset), buf.detach())
            for name, buf in self.module.named_buffers()
            if name.endswith("mlp.router.expert_bias")
        ]
        pp = mpu.get_pipeline_model_parallel_world_size()
        if pp == 1:
            return local

        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        meta_local = [(k, tuple(v.shape), v.dtype) for k, v in local]
        gathered_meta: list = [None] * pp
        dist.all_gather_object(gathered_meta, meta_local, group=pp_group)
        mine = dict(local)
        out = []
        for src in range(pp):
            src_global = dist.get_global_rank(pp_group, src)
            for (k, shape, dtype) in gathered_meta[src]:
                t = (
                    mine[k].contiguous() if src == pp_rank
                    else torch.empty(shape, dtype=dtype, device="cuda")
                )
                dist.broadcast(t, src=src_global, group=pp_group)
                out.append((k, t))
        return out

    def get_per_tensor_param(self, **kwargs):
        """Yield the whole model as (rollout_name, tensor), one tensor at a time.

        ⚠️ Lazy, and every tensor comes out of a collective, so the caller must
        drain the generator on every rank. Consuming it partially deadlocks.
        """
        assert self.module is not None
        _mem_diag_note("weight_sync gather begin")
        if self._is_dsv4:
            # The gather is shared: DSv4's grouped experts carry the same Megatron
            # names Qwen3-MoE does. Only the naming on the way out differs, and it
            # has to land on the DSv4 *checkpoint* names, because the rollout side
            # feeds them to vLLM's own ``load_weights``.
            named = itertools.chain(
                self._full_megatron_named_params_moe(),
                self._dsv4_router_bias_buffers(),
            )
            gen = dsv4.megatron_to_dsv4_native(named)
        elif getattr(self, "_is_moe", False):
            gen = megatron_to_hf_moe(self._full_megatron_named_params_moe(), self._dims)
        else:
            gen = megatron_to_hf(self._full_megatron_named_params(), self._dims, te=True)
        return _mem_diag_stream("weight_sync gather", gen), None

    def is_mp_src_rank_with_outputs(self) -> bool:
        """Only TP-rank 0 / CP-rank 0 on the last pipeline stage reports output.

        All TP/PP/CP members of one data-parallel shard receive the same rows and
        participate in collectives. CP ranks first reconstruct the complete
        log-prob tensor; returning it from only one rank prevents duplicates in
        the controller merge."""
        try:
            from megatron.core import parallel_state as mpu
            return (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_context_parallel_rank() == 0
                and mpu.is_pipeline_last_stage()
            )
        except Exception:
            return True

    # ============== Packed CP + pipeline-schedule forward/backward ==============
    #
    # PP requires Megatron's pipeline schedule. CP also uses this path when PP=1,
    # because it gives both topologies the same packed-microbatch semantics and
    # gradient finalization. Each microbatch is one packed bin of sequences.

    def _pp_setup_config(self) -> None:
        """Wire the schedule's grad hooks onto the transformer config (once).

        Both hooks are backward-only, so a forward-only engine
        (``build_optimizer=False``) leaves them unset rather than dereferencing
        an optimizer it never built.
        """
        if getattr(self, "_pp_cfg_ready", False):
            return
        from megatron.core.distributed import finalize_model_grads
        if self.optimizer is not None:
            self._tfcfg.finalize_model_grads_func = finalize_model_grads
            self._tfcfg.grad_scale_func = self.optimizer.scale_loss
        self._tfcfg.timers = None
        self._pp_cfg_ready = True

    def _collect_rows(self, seqs, am):
        rows = []
        B = seqs.shape[0]
        for r in range(B):
            start, L = self._real_block(am[r])
            if L >= 2:
                rows.append((r, start, L))
        return rows

    def _cp_local_length(self, length: int) -> int:
        """Number of this CP rank's packed tokens for one sequence."""
        if self._cp == 1:
            return length
        chunk = (length + 2 * self._cp - 1) // (2 * self._cp)
        return 2 * chunk

    def _build_microbatches(self, seqs, rows):
        """Group valid rows into packed bins (one bin == one pipeline microbatch)."""
        if self._dynamic_batch:
            budget = self._max_tokens_per_gpu if self._max_tokens_per_gpu > 0 else 21504
            # The budget is per GPU, hence CP uses the padded LOCAL token count.
            lengths = [self._cp_local_length(L) for (_, _, L) in rows]
            bins = self._build_bins(lengths, budget)
        else:
            bins = [[i] for i in range(len(rows))]
        mbs = []
        for bin_rows in bins:
            ids_list = [seqs[rows[j][0], rows[j][1]:rows[j][1] + rows[j][2]].to("cuda") for j in bin_rows]
            mbs.append({"rows": [rows[j] for j in bin_rows], "ids_list": ids_list})
        return mbs

    def _pp_forward_model(self, model, ids_list):
        """Pack full sequences into this rank's ``thd`` token stream.

        Two CP layouts, because the two attention implementations want different
        ones and neither checks:

        * **zigzag** (TE, the default). A sequence is padded to ``2*cp*chunk``
          and rank ``r`` takes ``chunk[r] + chunk[2*cp-r-1]``. The pairing is
          what keeps a causal mask's work balanced across ranks.
        * **contiguous** (DSv4). Rank ``r`` takes ``chunk[r]`` only. DSv4 does
          not use TE attention: it all-gathers KV itself and derives every
          position from ``cp_rank * seqlen_local``
          (``cp_utils.get_q_positions_for_cp`` is a single ``arange``), while
          ``all_gather_cp`` relies on ``cat`` over ranks being natural token
          order and the compressor's overlap gathers, transforms and re-slices
          contiguously. Handing DSv4 zigzag shards is silently wrong -- wrong
          RoPE, wrong causal spans -- because nothing on its side asserts the
          layout. The load imbalance zigzag exists to fix is also much smaller
          here: DSv4's attention is sparse (128-token window plus compressed
          groups), so per-query work is nearly constant rather than growing with
          position.

        Packed ``cu_seqlens`` are cumulative LOCAL lengths multiplied by CP size,
        as required by TE ring attention. DSv4 ignores ``packed_seq_params``
        entirely.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.packed_seq_params import PackedSeqParams

        cp_rank = mpu.get_context_parallel_rank()

        # Token-count alignment, computed BEFORE the CP split because under the
        # contiguous layout the per-rank chunk itself has to be aligned.
        #   * sequence parallelism (required for MoE+TP) scatters the packed
        #     sequence across TP ranks, so it must be a multiple of TP;
        #   * DSv4's compressor asserts ``seqlen % ratio == 0`` on every forward,
        #     and under CP additionally ``seqlen % (ratio*2) == 0``, while RL
        #     sequence lengths are whatever the rollout produced.
        align = self._tp if (self._sp and self._tp > 1) else 1
        if self._is_dsv4:
            align = math.lcm(align, self._dsv4_align * (2 if self._cp > 1 else 1))

        cp_contiguous = self._is_dsv4 and self._cp > 1
        if cp_contiguous and len(ids_list) > 1:
            # DSv4 reads the packed stream as ONE sequence (it ignores
            # cu_seqlens), which is why enable_dynamic_batch is forced off and
            # every microbatch holds a single sequence. Under CP that stops being
            # merely wrong-attention and starts breaking the position arithmetic
            # below, so refuse instead of computing something plausible.
            raise RuntimeError(
                f"DSv4 with CP={self._cp} needs one sequence per microbatch, got "
                f"{len(ids_list)}; enable_dynamic_batch must stay False."
            )

        local_ids = []
        local_offsets = [0]
        layouts = []
        for ids in ids_list:
            length = int(ids.numel())
            if self._cp == 1:
                chunk = length
                sliced = ids.view(-1)
            elif cp_contiguous:
                # ``chunk`` is rounded up to ``align`` so the padded sequence
                # divides evenly and NO trailing pad is needed. That is not
                # cosmetic: DSv4 takes this rank's global positions to be
                # ``[cp_rank*seqlen_local, (cp_rank+1)*seqlen_local)``, so any
                # padding sitting between two ranks' shards would shift every
                # later position by ``cp_rank*pad`` and silently corrupt RoPE
                # and the causal spans.
                chunk = -(-length // self._cp)
                chunk = -(-chunk // align) * align
                padded_length = self._cp * chunk
                ids_padded = (
                    F.pad(ids.view(-1), (0, padded_length - length), value=0)
                    if padded_length > length else ids.view(-1)
                )
                start = cp_rank * chunk
                sliced = ids_padded[start:start + chunk]
            else:
                chunk = (length + 2 * self._cp - 1) // (2 * self._cp)
                padded_length = 2 * self._cp * chunk
                if padded_length > length:
                    ids_padded = F.pad(ids.view(-1), (0, padded_length - length), value=0)
                else:
                    ids_padded = ids.view(-1)
                first = cp_rank * chunk
                second = (2 * self._cp - cp_rank - 1) * chunk
                sliced = torch.cat(
                    [ids_padded[first:first + chunk], ids_padded[second:second + chunk]],
                    dim=0,
                )
            begin = local_offsets[-1]
            end = begin + int(sliced.numel())
            layouts.append(
                {
                    "length": length,
                    "chunk": chunk,
                    "local_start": begin,
                    "local_end": end,
                    # Recorded rather than re-derived: the reconstruction in
                    # ``_cp_row_logprob_entropy`` has to invert exactly this
                    # split, and the two must not be able to drift apart.
                    "cp_layout": "contiguous" if cp_contiguous else "zigzag",
                }
            )
            local_ids.append(sliced)
            local_offsets.append(end)

        local_total = local_offsets[-1]
        # Pad the packed stream with a trailing dummy segment. It gets its own
        # cu_seqlens bin so attention stays isolated, and no layout entry points
        # at it, so its logits are never read for loss. Under DSv4 cu_seqlens is
        # ignored, but the padding is still invisible to the real positions
        # because everything the attention does is causal.
        #
        # Under the contiguous layout every chunk is already a multiple of
        # ``align``, so this is a no-op there -- which is the point.
        if align > 1:
            pad = (-local_total) % align
            if pad:
                local_ids.append(torch.zeros(pad, dtype=torch.long, device=ids_list[0].device))
                local_offsets.append(local_total + pad)
                local_total = local_offsets[-1]
            # Variable-length efficiency: padding is bounded by align-1 tokens
            # per microbatch. Under SP alone that is negligible; under DSv4's
            # 128-token alignment with one short sequence per microbatch it is
            # not, so log the ratio once (rank 0) rather than leave it invisible.
            if not getattr(self, "_pad_logged", False) and self._rank() == 0:
                real = local_total - pad
                logger.info(
                    "MegatronNativeEngine forward pad: align=%d (+%d/%d tokens, %.2f%%) "
                    "per microbatch; worst case (align-1)/tokens=%.2f%%",
                    align, pad, local_total, 100.0 * pad / max(1, local_total),
                    100.0 * (align - 1) / max(1, real),
                )
                self._pad_logged = True
        tokens = torch.cat(local_ids, dim=0).view(1, local_total)
        # For RoPE + packed thd, Megatron derives positions from cu_seqlens and
        # CP rank; explicit position_ids are neither needed nor consumed.
        cu = torch.tensor(local_offsets, dtype=torch.int32, device=tokens.device) * self._cp
        full_lens = [
            (local_offsets[i + 1] - local_offsets[i]) * self._cp
            for i in range(len(ids_list))
        ]
        mx = max(full_lens) if full_lens else 0
        psp = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=mx,
            max_seqlen_kv=mx,
            qkv_format="thd",
        )
        out = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            packed_seq_params=psp,
        )
        return out, layouts

    def _cp_row_logprob_entropy(
        self,
        packed_logits: torch.Tensor,
        ids: torch.Tensor,
        layout: dict,
        *,
        training: bool,
        want_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute local chunk outputs and reconstruct the full ``L-1`` order.

        A logit at global position ``i`` predicts token ``i+1``. This explicitly
        retains the prediction at a chunk boundary even when its target token is
        owned by another CP rank, which is the common shift-by-one failure mode.
        """
        from megatron.core import parallel_state as mpu

        length = int(layout["length"])
        local = packed_logits[layout["local_start"]:layout["local_end"]]
        if self._cp == 1:
            if training:
                return self._token_logprob_train(local[:-1], ids[1:]), None
            return self._logprob_entropy_nograd(local[:-1], ids[1:], want_entropy)

        cp_rank = mpu.get_context_parallel_rank()
        chunk = int(layout["chunk"])
        # Invert whatever split _pp_forward_model chose. One chunk under the
        # contiguous layout, the zigzag pair otherwise.
        if layout.get("cp_layout") == "contiguous":
            global_starts = (cp_rank * chunk,)
        else:
            global_starts = (
                cp_rank * chunk,
                (2 * self._cp - cp_rank - 1) * chunk,
            )
        logits_parts = []
        target_parts = []
        spans = []
        for half, global_start in enumerate(global_starts):
            # Valid causal-logit positions are [0, L-1); positions >= L-1 are
            # padding or the final token (which has no next-token target).
            global_end = min(global_start + chunk, length - 1)
            if global_start >= global_end:
                continue
            n = global_end - global_start
            local_start = half * chunk
            logits_parts.append(local[local_start:local_start + n])
            target_parts.append(ids[global_start + 1:global_end + 1])
            spans.append((global_start, global_end))

        if logits_parts:
            logits_cat = torch.cat(logits_parts, dim=0)
            targets_cat = torch.cat(target_parts, dim=0)
            if training:
                local_lp = self._token_logprob_train(logits_cat, targets_cat)
                local_ent = None
            else:
                local_lp, local_ent = self._logprob_entropy_nograd(
                    logits_cat, targets_cat, want_entropy,
                )
            split_sizes = [end - start for start, end in spans]
            lp_splits = local_lp.split(split_sizes)
            full_parts = [
                F.pad(part, (start, length - 1 - end))
                for part, (start, end) in zip(lp_splits, spans, strict=True)
            ]
            full_lp = full_parts[0]
            for part in full_parts[1:]:
                full_lp = full_lp + part
            if want_entropy and local_ent is not None:
                ent_splits = local_ent.split(split_sizes)
                ent_parts = [
                    F.pad(part, (start, length - 1 - end))
                    for part, (start, end) in zip(ent_splits, spans, strict=True)
                ]
                full_ent = ent_parts[0]
                for part in ent_parts[1:]:
                    full_ent = full_ent + part
            else:
                full_ent = None
        else:
            # Preserve a zero-valued autograd path on ranks that own only padding.
            zero = local.sum() * 0.0
            full_lp = torch.zeros(
                length - 1, dtype=local.dtype, device=local.device,
            ) + zero
            full_ent = (
                torch.zeros(length - 1, dtype=local.dtype, device=local.device)
                if want_entropy else None
            )

        cp_group = mpu.get_context_parallel_group()
        if training:
            # SUM in both forward and backward is intentional. Every CP rank
            # computes the same reconstructed loss; backward SUM supplies a CP
            # factor that cancels Megatron DDP's averaging over DP×CP. Therefore
            # ``dp_size`` in DAPO remains the pure (non-CP) DP width.
            from torch.distributed.nn.functional import all_reduce
            full_lp = all_reduce(full_lp, group=cp_group)
        else:
            dist.all_reduce(full_lp, group=cp_group)
            if full_ent is not None:
                dist.all_reduce(full_ent, group=cp_group)
        return full_lp, full_ent

    def _pad_mbs_for_ep(self, mbs: list) -> list:
        """Lockstep the microbatch COUNT across the Expert-Parallel group.

        MoE's all-to-all token dispatch is a collective over the EP group, so every
        EP rank must run the same NUMBER of forward passes or the collective hangs.
        RL microbatch counts are data-dependent and differ per DP/EP rank, so we
        all-reduce the max count over the EP group and pad short ranks with dummy
        2-token microbatches (empty ``rows`` -> zero loss / zero grad, but they
        still participate in the expert all-to-all)."""
        if not getattr(self, "_is_moe", False) or self._ep <= 1:
            return mbs
        from megatron.core import parallel_state as mpu
        cnt = torch.tensor([len(mbs)], device="cuda", dtype=torch.long)
        dist.all_reduce(cnt, op=dist.ReduceOp.MAX, group=mpu.get_expert_model_parallel_group())
        target = int(cnt.item())
        while len(mbs) < target:
            mbs.append({"rows": [], "ids_list": [torch.zeros(2, dtype=torch.long, device="cuda")]})
        return mbs

    def _dsv4_check_topology(self) -> None:
        """Refuse the topologies DSv4 is known or suspected to get wrong.

        ``DeepseekV4Attention.forward`` accepts ``packed_seq_params`` and never
        reads it: RoPE frequencies, the sliding window, the compressor's
        ``(p+1)//ratio`` grouping and the indexer's causal span are all derived
        from the tensor's own length. So a microbatch holding several sequences
        would be read as one long sequence -- no error, just wrong attention.
        DSv4 therefore keeps ``enable_dynamic_batch=False``, which makes every
        microbatch exactly one sequence and the packing degenerate; the pipeline
        path is otherwise safe to use, and it is where the Expert-Parallel
        microbatch lockstep lives.

        Pipeline parallelism: MEASURED CORRECT on both sides, so it is no longer
        refused. Two independent checks, because one of them is not enough:

        * training -- ``probe_69`` (``run_probes.sh topology``) diffs log-probs,
          loss and grad_norm against a PP=1 baseline at the SAME EP. PP=2 EP=4 vs
          PP=1 EP=4: bitwise-equal log-probs (max |d| 0.0 over 255 positions),
          bitwise-equal loss (-0.04613047), grad_norm to 4.5e-08.
        * rollout weight sync -- ``probe_68 --pp 2`` (``PP=2 run_probes.sh
          syncshapes``): 3176/3176 tensors, 0 missing, 0 extra, 0 shape mismatch.
          probe_69 cannot see this path at all, because it never syncs weights.

        Getting the second one to pass took two fixes, both PP-only and both
        invisible at PP=1: ``_pp_layer_offset`` (the stage offset was read out of
        checkpoint metadata in a way that only works for homogeneous blocks, so
        later stages republished their params under the first stage's layer
        numbers and half the model never shipped) and the PP broadcast for the
        router's ``expert_bias`` buffers (a stage holds only its own).

        The old synthetic PP=2 probe's differing loss (1.77839053 -> 1.78812289)
        was a measurement artifact, not a PP defect: its baseline did not hold EP
        fixed. Changing EP alone moves these numbers by far more than PP does --
        PP=2 EP=4 against an EP=8 baseline reports a grad_norm gap of 2.35e-03,
        all of which is the EP change (PP=1 EP=4 and PP=2 EP=4 agree to 4.5e-08),
        and the reference deviation moves from 5.106e-02 at EP=8 to 5.212e-02 at
        EP=4. Anything comparing topologies has to change exactly one at a time.
        ``MHC_CONTIGUOUS`` was therefore never implicated; consistent with its own
        comment that a non-contiguous p2p send only makes NCCL warn.

        The old synthetic PP=2 probe's differing loss (1.77839053 -> 1.78812289)
        was a measurement artifact, not a PP defect: its baseline did not hold EP
        fixed. Changing EP alone moves these numbers by far more than PP does --
        PP=2 EP=4 against an EP=8 baseline reports a grad_norm gap of 2.35e-03,
        all of which is the EP change (PP=1 EP=4 and PP=2 EP=4 agree to 4.5e-08),
        and the reference deviation moves from 5.106e-02 at EP=8 to 5.212e-02 at
        EP=4. Anything comparing topologies has to change exactly one at a time.
        ``MHC_CONTIGUOUS`` was therefore never implicated; consistent with its own
        comment that a non-contiguous p2p send only makes NCCL warn.

        The plumbing this relies on: mHC makes the tensor crossing a stage
        boundary 4-D ``[s, b, hc, d]``, which the vendored p2p accounts for
        (``_communicate_shapes`` uses ``num_dims = 4 if config.dsv4_mode``);
        ``variable_seq_lengths`` is set whenever pp > 1 so the shape is exchanged
        rather than inferred from config; and the hc streams survive the boundary
        because ``transformer_block`` expands only on ``pre_process`` and
        collapses only on ``post_process``.

        Context parallelism: MEASURED WRONG, still refused. The layout conflict
        described below is handled -- ``_pp_forward_model`` emits contiguous
        shards for DSv4 instead of the zigzag pairs TE wants, and records the
        choice so ``_cp_row_logprob_entropy`` inverts the same split -- but that
        is not sufficient. CP=2 EP=4 against PP=1 CP=1 EP=4 differs on 509 of 511
        positions, **starting at global position 0**, max |d log-prob| 3.86e-01.

        Position 0's logit can only depend on token 0, so the three suspects that
        motivated the layout work cannot produce this: a chunk-alignment shift, a
        position offset and an off-by-one in the reconstruction all leave a clean
        prefix. Padding is also ruled out -- the run above tiles the reference to
        512 tokens so that ``chunk`` lands on 256 exactly and there is no padding
        at all, and it fails the same way as the 256-token run (250/255, also from
        position 0). What is left is DSv4's own CP attention: it all-gathers KV
        over the CP group and derives every position from ``cp_rank *
        seqlen_local`` (``cp_utils.get_q_positions_for_cp`` is a single
        ``arange``), and something in that path -- most likely the causal masking
        over the gathered KV or the compressor's ``overlap_transform_with_cp`` --
        does not reproduce the single-device result. Not investigated further: CP
        is not needed for the DAPO runs.
        """
        if self._dynamic_batch:
            raise RuntimeError(
                "DSv4 cannot run with enable_dynamic_batch=True: several sequences in one "
                "microbatch are read as one long sequence. initialize() forces it off, so "
                "reaching this means something re-enabled it."
            )
        if self._cp > 1:
            raise NotImplementedError(
                f"DSv4 does not support context_parallel_size={self._cp}: the forward is "
                f"measured wrong, not merely unimplemented -- probe_69 sees 509/511 log-prob "
                f"positions differ from a CP=1 baseline, starting at position 0. "
                f"See _dsv4_check_topology. TP / PP / EP / DP are available."
            )

    def engine_update_policy(self, batch):
        if self._is_dsv4:
            self._dsv4_check_topology()
            return self._pp_update_policy(batch)
        if self._pp == 1 and self._cp == 1 and not getattr(self, "_is_moe", False):
            return super().engine_update_policy(batch)
        return self._pp_update_policy(batch)

    def _pp_update_policy(self, batch) -> dict[str, float]:
        from megatron.core.pipeline_parallel import get_forward_backward_func

        if batch.batch_size == 0 and not getattr(self, "_is_moe", False):
            return {"loss": 0.0, "lr": self._cur_lr(), "grad_norm": 0.0}
        self._pp_setup_config()
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

        loss_agg_mode = _cfg("loss_agg_mode", "token-mean")
        global_batch_size = (
            meta.get("global_batch_size") or batch.batch_size * dp
        )
        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        rows = self._collect_rows(seqs, am)
        mbs = self._build_microbatches(seqs, rows)
        # MoE: lockstep the microbatch count across the EP group (all-to-all).
        mbs = self._pad_mbs_for_ep(mbs)
        num_mb = len(mbs)

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()
        _mem_diag_begin("update", mbs)

        if num_mb == 0:
            # keep the pipeline in lockstep even with no data (rare); no step.
            return {"loss": 0.0, "lr": self._cur_lr(), "grad_norm": 0.0}

        data_iter = iter(mbs)

        def forward_step(di, model, *args, **kwargs):
            mb = next(di[0] if isinstance(di, list) else di)
            out, layouts = self._pp_forward_model(model, mb["ids_list"])

            def loss_func(output_tensor):
                # reshape by ACTUAL length: under SP the packed stream is padded to
                # a TP multiple, so the output can be longer than the real token
                # total. Real sequences index via their (unpadded) layouts.
                lt = output_tensor.logits if hasattr(output_tensor, "logits") else output_tensor
                logits = lt.reshape(-1, lt.shape[-1]).float() / temperature
                bin_loss = None
                agg = {"loss": 0.0, "n": 0, "ppo_kl_sum": 0.0, "ppo_kl_tok": 0.0,
                       "rc_kl_sum": 0.0, "rc_kl_tok": 0.0}
                for k, (r, start, _L) in enumerate(mb["rows"]):
                    token_lp, _ = self._cp_row_logprob_entropy(
                        logits,
                        mb["ids_list"][k],
                        layouts[k],
                        training=True,
                        want_entropy=False,
                    )
                    token_lp = token_lp.view(1, -1)
                    loss, stats = self._row_policy_loss(
                        t,
                        r,
                        start,
                        token_lp,
                        algo_name,
                        _cfg,
                        bnt,
                        dp,
                        loss_agg_mode,
                        global_batch_size,
                    )
                    if loss is None:
                        continue
                    bin_loss = loss if bin_loss is None else bin_loss + loss
                    agg["loss"] += stats["loss"]; agg["n"] += 1
                    agg["ppo_kl_sum"] += stats["ppo_kl_sum"]; agg["ppo_kl_tok"] += stats["ppo_kl_tok"]
                    agg["rc_kl_sum"] += stats["rc_kl_sum"]; agg["rc_kl_tok"] += stats["rc_kl_tok"]
                if bin_loss is None:
                    bin_loss = logits.sum() * 0.0
                # Megatron's two-value loss contract applies ``*cp/num_mb``.
                # Cancel both factors here. CP is already accounted for by the
                # differentiable reconstruction's backward SUM followed by DDP's
                # DP×CP average; retaining the schedule's CP multiplier would make
                # CP=2 gradients exactly 2x too large.
                return bin_loss * num_mb / self._cp, agg

            return out, loss_func

        fwd_bwd = get_forward_backward_func()
        # R3: replay the router logits recorded during the old-logprob forward so
        # the importance ratio reflects only weight changes, not router drift.
        from contextlib import nullcontext
        if self._r3_enabled and self._r3_store:
            from lumenrl.moe.moe_utils import megatron_replay_router_logits
            r3_ctx = megatron_replay_router_logits(self.module, self._r3_store)
        else:
            r3_ctx = nullcontext()
        with r3_ctx:
            losses = fwd_bwd(
                forward_step_func=forward_step, data_iterator=[data_iter], model=[self._ddp],
                num_microbatches=num_mb, seq_length=1, micro_batch_size=1, forward_only=False,
            )

        _mem_diag_end("update")
        update_successful, grad_norm, _ = self.optimizer.step()
        _mem_diag_end("update+optstep")
        if not update_successful:
            logger.warning("PP optimizer.step reported update_successful=False")
        lr = self._sched_step()
        gn = float(grad_norm) if grad_norm is not None else 0.0

        # Loss/KL only exist on the last pipeline stage; other stages report just
        # the (global) grad_norm/lr so the controller's metric averaging isn't
        # diluted with zeros from non-last stages.
        if not self._is_last_stage:
            return {"grad_norm": gn, "lr": lr}

        # Aggregate metrics on the last stage (where losses were computed).
        loss_accum = ppo_kl_sum = ppo_kl_tok = rc_kl_sum = rc_kl_tok = 0.0
        n_rows = 0
        for agg in losses:
            loss_accum += agg["loss"]; n_rows += agg["n"]
            ppo_kl_sum += agg["ppo_kl_sum"]; ppo_kl_tok += agg["ppo_kl_tok"]
            rc_kl_sum += agg["rc_kl_sum"]; rc_kl_tok += agg["rc_kl_tok"]
        metrics = {
            "loss": loss_accum / max(1, n_rows),
            "lr": lr,
            "grad_norm": gn,
        }
        if ppo_kl_tok > 0:
            metrics["ppo_kl_sum"] = ppo_kl_sum
            metrics["ppo_kl_tok"] = ppo_kl_tok
        if rc_kl_tok > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl_sum
            metrics["rollout_corr_kl_tok"] = rc_kl_tok

        from lumenrl.engine.training.megatron_base_engine import _GAP_ROWS, _gap_dump_dir
        _dump = _gap_dump_dir()
        if _dump and _GAP_ROWS:
            rank = dist.get_rank() if dist.is_initialized() else 0
            rows = [r for r in _GAP_ROWS if r["rollout_lp"] is not None]
            if rows:
                torch.save(
                    {
                        "train_lp": torch.cat([r["train_lp"] for r in rows]),
                        "old_lp": torch.cat([r["old_lp"] for r in rows]),
                        "rollout_lp": torch.cat([r["rollout_lp"] for r in rows]),
                        "rc_kl_sum": rc_kl_sum, "rc_kl_tok": rc_kl_tok,
                        "ppo_kl_sum": ppo_kl_sum, "ppo_kl_tok": ppo_kl_tok,
                    },
                    f"{_dump}/engine_gap_rank{rank}.pt",
                )
        _GAP_ROWS.clear()
        return metrics

    def engine_compute_log_probs(self, batch):
        if self._is_dsv4:
            self._dsv4_check_topology()
            return self._pp_compute_log_probs(batch)
        if self._pp == 1 and self._cp == 1 and not getattr(self, "_is_moe", False):
            return super().engine_compute_log_probs(batch)
        return self._pp_compute_log_probs(batch)

    def _pp_compute_log_probs(self, batch):
        from megatron.core.pipeline_parallel import get_forward_backward_func

        self._pp_setup_config()
        seqs = batch["input_ids"]
        B, S = seqs.shape
        am = batch.tensors.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        want_ent = bool(batch.meta.get("calculate_entropy", False))
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        rows = self._collect_rows(seqs, am)
        mbs = self._build_microbatches(seqs, rows)
        # MoE: lockstep the microbatch count across the EP group (all-to-all).
        mbs = self._pad_mbs_for_ep(mbs)
        num_mb = len(mbs)
        self.module.eval()
        _mem_diag_begin("log_probs", mbs)

        lp_out = torch.zeros(B, S, dtype=torch.float32)
        ent_out = torch.zeros(B, S, dtype=torch.float32) if want_ent else None

        if num_mb > 0:
            data_iter = iter(mbs)

            def forward_step(di, model, *args, **kwargs):
                mb = next(di[0] if isinstance(di, list) else di)
                out, layouts = self._pp_forward_model(model, mb["ids_list"])

                def collect(output_tensor, non_loss_data=True):
                    lt = output_tensor.logits if hasattr(output_tensor, "logits") else output_tensor
                    logits = lt.reshape(-1, lt.shape[-1]).float() / temperature
                    res = []
                    for k, (r, start, L) in enumerate(mb["rows"]):
                        tok_lp, ent = self._cp_row_logprob_entropy(
                            logits,
                            mb["ids_list"][k],
                            layouts[k],
                            training=False,
                            want_entropy=want_ent,
                        )
                        res.append((r, start, L, tok_lp.cpu(), ent.cpu() if ent is not None else None))
                    return res

                return out, collect

            fwd_bwd = get_forward_backward_func()
            # R3: record this (old-logprob) forward's router logits so the update
            # can replay the identical routing. Reset the per-step store first.
            from contextlib import nullcontext
            if self._r3_enabled:
                from lumenrl.moe.moe_utils import megatron_record_router_logits
                self._r3_store = {}
                r3_ctx = megatron_record_router_logits(self.module, self._r3_store)
            else:
                r3_ctx = nullcontext()
            with torch.no_grad(), r3_ctx:
                data_store = fwd_bwd(
                    forward_step_func=forward_step, data_iterator=[data_iter], model=[self.module],
                    num_microbatches=num_mb, seq_length=1, micro_batch_size=1,
                    forward_only=True, collect_non_loss_data=True,
                )
            # ``data_store`` is populated on the last stage only.
            for res in data_store:
                for (r, start, L, tok_lp, ent) in res:
                    lp_out[r, start:start + L - 1] = tok_lp
                    if want_ent and ent is not None:
                        ent_out[r, start:start + L - 1] = ent
        self.module.train()
        _mem_diag_end("log_probs")
        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    # ---- native Megatron distributed checkpoint (sharded, resharding-capable) ----
    def _dist_sharded_state_dict(self, is_loading: bool):
        """Assemble the model+optimizer sharded_state_dict for dist-checkpoint.

        Uses ``dp_zero_gather_scatter`` optimizer sharding: the default
        ``fully_sharded_model_space`` emits ``flattened_range`` ShardedTensors
        which the torch_dist save strategy rejects. gather_scatter is torch_dist
        compatible and still supports DP-resharding on load.
        """
        model_ssd = self.module.sharded_state_dict()
        opt_ssd = self.optimizer.sharded_state_dict(
            model_ssd, is_loading=is_loading,
            metadata={"distrib_optim_sharding_type": "dp_zero_gather_scatter"},
        )
        return {"model": model_ssd, "optimizer": opt_ssd}

    def save_dist_checkpoint(self, local_path: str, global_step: int = 0) -> bool:
        """Save a Megatron dist-checkpoint (torch_dist sharded). Unlike the
        per-rank ``torch.save`` path, this is sharded (no DP weight duplication)
        and can be reloaded under a different TP/PP/DP topology."""
        import megatron.core.dist_checkpointing as dc
        sharded_sd = self._dist_sharded_state_dict(is_loading=False)
        # common (non-sharded) metadata replicated on every rank
        sharded_sd["global_step"] = int(global_step)
        if self.lr_scheduler is not None:
            sharded_sd["lr_scheduler"] = self.lr_scheduler.state_dict()
        os.makedirs(local_path, exist_ok=True)
        dc.save(sharded_sd, str(local_path))
        if dist.is_initialized():
            dist.barrier()
        return True

    def load_dist_checkpoint(self, local_path: str) -> int:
        """Load a Megatron dist-checkpoint (auto-resharded to current topology)."""
        import megatron.core.dist_checkpointing as dc
        sharded_sd = self._dist_sharded_state_dict(is_loading=True)
        sharded_sd["global_step"] = 0
        loaded = dc.load(sharded_sd, str(local_path))
        self.module.load_state_dict(loaded["model"])
        self.optimizer.load_state_dict(loaded["optimizer"])
        if self.lr_scheduler is not None and loaded.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(loaded["lr_scheduler"])
        if dist.is_initialized():
            dist.barrier()
        return int(loaded.get("global_step", 0))


@EngineRegistry.register(model_type="language_model", backend="megatron_native")
class MegatronNativeEngineWithLMHead(MegatronNativeEngine):
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron_native")
class MegatronNativeEngineWithValueHead(MegatronNativeEngine):
    pass
