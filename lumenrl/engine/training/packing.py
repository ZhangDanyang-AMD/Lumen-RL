"""Sequence packing for efficient training with AITER flash_attn_varlen_func.

Eliminates padding waste by concatenating actual tokens from multiple sequences
into a single flat tensor and using variable-length flash attention to maintain
correct cross-sequence isolation.

Usage in training step::

    packed = pack_sequences(input_ids, attention_mask)
    with PackingContext(packed.cu_seqlens, packed.max_seqlen):
        outputs = model(input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None)
        logits = outputs.logits.squeeze(0)
        flat_lp = packed_token_log_probs(logits, packed.input_ids.squeeze(0),
                                         packed.cu_seqlens)
        log_probs = unpack_log_probs(flat_lp, packed.cu_seqlens,
                                      packed.seq_lens, input_ids.shape[1])
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Flash-Attention's Triton cross-entropy, the same kernel verl reaches through
# `logprobs_from_logits`. It fuses log_softmax + gather and supports an in-place
# backward, so per-token log-probs cost no extra vocab-sized allocation.
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _FUSED_CROSS_ENTROPY
except ImportError:  # fall back to chunked log_softmax + gather
    _FUSED_CROSS_ENTROPY = None

# ---------------------------------------------------------------------------
# Packing context (module-global, survives gradient checkpointing recompute)
# ---------------------------------------------------------------------------
# PyTorch's non-reentrant gradient checkpointing (use_reentrant=False)
# recomputes forward in a context where threading.local() state is NOT
# visible.  We use plain module globals instead — safe because training
# is single-threaded per process (FSDP2 uses one process per GPU).

_cu_seqlens: torch.Tensor | None = None
_max_seqlen: int = 0


def set_packing_context(cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
    """Store packing metadata (used by attention patch during forward)."""
    global _cu_seqlens, _max_seqlen
    _cu_seqlens = cu_seqlens
    _max_seqlen = max_seqlen


def get_packing_context() -> tuple[torch.Tensor, int] | None:
    """Return (cu_seqlens, max_seqlen) or None if not in a packed forward pass."""
    if _cu_seqlens is None:
        return None
    return _cu_seqlens, _max_seqlen


def clear_packing_context() -> None:
    global _cu_seqlens, _max_seqlen
    _cu_seqlens = None
    _max_seqlen = 0


class PackingContext:
    """Context manager that keeps packing metadata alive through backward.

    Must wrap both forward AND backward so that gradient checkpointing
    recomputation can see the packing state.
    """

    def __init__(self, cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = max_seqlen

    def __enter__(self) -> "PackingContext":
        set_packing_context(self.cu_seqlens, self.max_seqlen)
        return self

    def __exit__(self, *args: object) -> None:
        clear_packing_context()


# ---------------------------------------------------------------------------
# Pack / unpack utilities
# ---------------------------------------------------------------------------


class PackedBatch(NamedTuple):
    input_ids: torch.Tensor      # (1, total_tokens)
    position_ids: torch.Tensor   # (1, total_tokens)
    cu_seqlens: torch.Tensor     # (B+1,) int32
    seq_lens: torch.Tensor       # (B,) long
    max_seqlen: int


def pack_from_nested(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> PackedBatch:
    """Create a :class:`PackedBatch` from pre-packed (remove-padding) tensors.

    Unlike :func:`pack_sequences` which takes padded ``(B, S)`` inputs,
    this accepts already-concatenated tokens (e.g. from nested tensors or
    an upstream packing pass).

    Args:
        input_ids: ``(1, total_tokens)`` or ``(total_tokens,)`` already-packed tokens.
        cu_seqlens: ``(B+1,)`` int32 cumulative sequence lengths.
        seq_lens: ``(B,)`` long per-sequence actual lengths.

    Returns:
        A :class:`PackedBatch` ready for model forward + ``PackingContext``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    total_tokens = input_ids.shape[-1]
    device = input_ids.device
    max_seqlen = int(seq_lens.max().item())

    position_ids = torch.empty(total_tokens, dtype=torch.long, device=device)
    B = seq_lens.shape[0]
    for i in range(B):
        start = int(cu_seqlens[i].item())
        sl = int(seq_lens[i].item())
        position_ids[start:start + sl] = torch.arange(sl, device=device)

    return PackedBatch(
        input_ids=input_ids,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
    )


def pack_sequences(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tp_align: int = 1,
) -> PackedBatch:
    """Pack left-padded sequences into a contiguous flat tensor.

    Args:
        input_ids: ``(B, S)`` left-padded token IDs.
        attention_mask: ``(B, S)`` where 1 = real token, 0 = pad.
        tp_align: Pad total_tokens to a multiple of this value (TP size)
            so that sequence-parallel reduce_scatter doesn't assert.

    Returns:
        A :class:`PackedBatch` with concatenated tokens, cumulative sequence
        lengths, per-sequence position IDs, and per-sequence actual lengths.
    """
    B, S = input_ids.shape
    device = input_ids.device

    seq_lens = attention_mask.sum(dim=1).long()          # (B,)
    total_tokens = int(seq_lens.sum().item())
    max_seqlen = int(seq_lens.max().item())

    pad_len = (tp_align - total_tokens % tp_align) % tp_align
    total_padded = total_tokens + pad_len

    packed_ids = torch.zeros(total_padded, dtype=input_ids.dtype, device=device)
    position_ids = torch.zeros(total_padded, dtype=torch.long, device=device)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens.int(), dim=0)

    offset = 0
    for i in range(B):
        sl = int(seq_lens[i].item())
        # Left-padded: actual tokens are at the right end [S-sl, S)
        packed_ids[offset:offset + sl] = input_ids[i, S - sl:S]
        position_ids[offset:offset + sl] = torch.arange(sl, device=device)
        offset += sl

    return PackedBatch(
        input_ids=packed_ids.unsqueeze(0),       # (1, total_padded)
        position_ids=position_ids.unsqueeze(0),  # (1, total_padded)
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
    )


def packed_token_log_probs(
    logits: torch.Tensor,
    packed_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token log-probs from packed (flat) logits.

    Performs the shifted log_softmax + gather per sequence, vectorized.

    Args:
        logits: ``(total_padded, V)`` model output (may include TP-alignment
            padding beyond ``cu_seqlens[-1]``).
        packed_ids: ``(total_padded,)`` packed token IDs.
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.
        temperature: logits are divided by this before ``log_softmax``, matching
            verl's ``logits.div_(temperature)`` in ``_forward_micro_batch`` (so the
            training log-probs reflect the sampling distribution). Must be applied
            consistently to old/train/ref log-probs or the importance ratio drifts.

    Returns:
        Flat ``(total_real_tokens - B,)`` float32 tensor of shifted log-probs.
        Each sequence *i* contributes ``sl_i - 1`` values.
    """
    total_real = int(cu_seqlens[-1].item())
    logits = logits[:total_real]
    packed_ids = packed_ids[:total_real]
    total_tokens = total_real
    device = logits.device

    # not_last[j] = True if j is NOT the last token of any sequence
    not_last = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_last[cu_seqlens[1:].long() - 1] = False

    # ATOM records torch.log_softmax(logits.float()). The fused Triton
    # cross-entropy sits about 1e-6 away from that reduction. Rowwise
    # attention is the path that is compared with those records, so it
    # uses the same float32 log_softmax.
    exact_log_softmax = os.environ.get("LUMENRL_ROWWISE_TRAIN_ATTN", "0") == "1"

    # ATOM's rollout blanks the padded vocabulary tail before that log_softmax:
    # ModelRunnerExt.postprocess sets logits[..., n:] = -inf for n =
    # true_vocab_size, the tokenizer's real token count (151669 against a
    # 151936-row lm_head on Qwen3-8B). Training otherwise sums those extra
    # columns into its denominator, so every rollout log-prob sits above the
    # training one by about the tail probability mass. Setting this to the same
    # n puts both sides on one denominator. The mask needs its own float32
    # copy of each chunk, so it takes the chunked path rather than the fused
    # kernel, which writes dlogits in place.
    vocab_mask = int(os.environ.get("LUMENRL_TRAIN_VOCAB_MASK", "0") or 0)
    if vocab_mask >= logits.shape[-1]:
        vocab_mask = 0
    if vocab_mask:
        exact_log_softmax = True

    if _FUSED_CROSS_ENTROPY is not None and logits.is_cuda and not exact_log_softmax:
        # Fused path: one Triton kernel over the full (total_tokens, V) buffer,
        # with the backward writing dlogits in place. Never materializes a
        # log_softmax tensor, and the boundary rows are dropped *after* the
        # kernel so no second vocab-sized copy is made either. Their gradient is
        # zero because the selection below routes no upstream grad to them.
        targets = torch.empty_like(packed_ids)
        targets[:-1] = packed_ids[1:]
        targets[-1] = 0  # position is dropped by not_last
        losses, _ = _FUSED_CROSS_ENTROPY(
            logits if logits.is_contiguous() else logits.contiguous(),
            targets,
            logit_scale=(1.0 / temperature) if temperature != 1.0 else 1.0,
            inplace_backward=True,
        )
        return (-losses)[not_last].float()  # (total - B,)

    # Fallback: chunked log_softmax + gather. Needs the shifted copies.
    not_first = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_first[cu_seqlens[:-1].long()] = False
    shifted_logits = logits[not_last]                          # (total - B, V)
    shifted_targets = packed_ids[not_first].unsqueeze(-1)      # (total - B, 1)

    chunk_size = 4096
    lp_parts = []
    for start in range(0, shifted_logits.shape[0], chunk_size):
        end = min(start + chunk_size, shifted_logits.shape[0])
        z = shifted_logits[start:end]
        if exact_log_softmax:
            z = z.float()
        if temperature != 1.0:
            z = z / temperature
        if vocab_mask:
            z[..., vocab_mask:] = float("-inf")
        row_lp = F.log_softmax(z, dim=-1)
        lp_parts.append(
            row_lp.gather(-1, shifted_targets[start:end]).squeeze(-1)
        )
    return torch.cat(lp_parts, dim=0).float()  # (total - B,)


@torch.no_grad()
def packed_token_entropy(
    logits: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 1024,
    temperature: float = 1.0,
    upcast: bool = True,
) -> torch.Tensor:
    """Per-token predictive entropy for packed (flat) logits, shifted.

    Entropy at each non-last position ``H = logsumexp(z) - sum(softmax(z) * z)``,
    chunked to bound peak memory. Detached (metric only).

    Args:
        temperature: logits are divided by this before the entropy computation,
            matching verl's ``logits.div_(temperature)`` prior to
            ``entropy_from_logits`` (so the metric reflects the sampling
            distribution at the rollout temperature).
        upcast: if True (default) compute in float32; set False to compute in the
            logits' native dtype (e.g. bf16) to match verl's ``actor/entropy``,
            which is computed on the bf16 forward logits without upcasting.

    Returns a flat ``(total_tokens - B,)`` tensor aligned with
    :func:`packed_token_log_probs` output.
    """
    total_real = int(cu_seqlens[-1].item())
    logits = logits[:total_real]
    total_tokens = total_real
    device = logits.device
    not_last = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_last[cu_seqlens[1:].long() - 1] = False

    # Chunk over row slices (views) and drop the boundary rows afterwards, so no
    # vocab-sized copy of the logits is ever materialized.
    parts = []
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        z = logits[start:end]
        if upcast:
            z = z.float()
        if temperature != 1.0:
            z = z / temperature
        lse = torch.logsumexp(z, dim=-1)
        probs = torch.softmax(z, dim=-1)
        ent = lse - (probs * z).sum(dim=-1)
        parts.append(ent.float())
    if not parts:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(parts, dim=0)[not_last]


def unpack_log_probs(
    flat_lp: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    padded_seq_len: int,
) -> torch.Tensor:
    """Scatter packed log-probs back into left-padded ``[B, S-1]`` format.

    Args:
        flat_lp: ``(total_tokens - B,)`` from :func:`packed_token_log_probs`.
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.
        seq_lens: ``(B,)`` per-sequence actual lengths.
        padded_seq_len: ``S`` — the original padded sequence dimension.

    Returns:
        ``[B, S-1]`` tensor with log-probs placed at the correct positions
        for left-padded sequences.  Padding positions are zero.
    """
    B = seq_lens.shape[0]
    S_m1 = padded_seq_len - 1
    result = flat_lp.new_zeros(B, S_m1)

    # Shifted cu_seqlens: cumulative (sl - 1) for indexing into flat_lp
    shifted_lens = seq_lens - 1
    shifted_cu = torch.zeros(B + 1, dtype=torch.long, device=flat_lp.device)
    shifted_cu[1:] = torch.cumsum(shifted_lens, dim=0)

    for i in range(B):
        sl = int(seq_lens[i].item())
        if sl <= 1:
            continue
        src_start = int(shifted_cu[i].item())
        src_end = int(shifted_cu[i + 1].item())
        # Left-padded: shifted positions are [S-sl, S-1)
        dst_start = S_m1 - (sl - 1)
        result[i, dst_start:dst_start + (sl - 1)] = flat_lp[src_start:src_end]

    return result


# ---------------------------------------------------------------------------
# Varlen attention patch
# ---------------------------------------------------------------------------

_original_attn_fn = None  # saved reference to the previous sdpa function


_row_meta_cache: tuple | None = None


def _packed_rows(cu_seqlens: torch.Tensor, total: int):
    """Row starts and lengths for ``cu_seqlens``, shared by all layers of a forward."""
    global _row_meta_cache
    from lumenrl.aiter_patch.qlen1_attention import packed_causal_rows

    cached = _row_meta_cache
    if cached is not None and cached[0] is cu_seqlens and cached[1] == total:
        return cached[2]
    rows = packed_causal_rows(cu_seqlens, total)
    # Holding the tensor keeps its id unique while it is the cache key.
    _row_meta_cache = (cu_seqlens, total, rows)
    return rows


class _RowwiseAttention(torch.autograd.Function):
    """q_len=1 attention values, with the packed-varlen gradient as a stand-in.

    The forward is the decode-shaped call. There is no backward for that
    launch, so the backward differentiates the ordinary causal varlen call
    on the same Q, K and V. Those two kernels agree on short sequences and
    diverge once a query reads past 2048 keys, so this gradient is not the
    gradient of the forward that produced the logprob.
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, softmax_scale):
        from lumenrl.aiter_patch.qlen1_attention import row_prefix_attention

        starts, lens = _packed_rows(cu_seqlens, q.shape[0])
        out = row_prefix_attention(q, k, v, starts, lens, max_seqlen, softmax_scale)
        ctx.save_for_backward(q, k, v, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from aiter import flash_attn_varlen_func

        q, k, v, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            q_ = q.detach().requires_grad_()
            k_ = k.detach().requires_grad_()
            v_ = v.detach().requires_grad_()
            result = flash_attn_varlen_func(
                q_, k_, v_,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=ctx.max_seqlen,
                max_seqlen_k=ctx.max_seqlen,
                dropout_p=0.0,
                softmax_scale=ctx.softmax_scale,
                causal=True,
                return_lse=True,
            )
            out = result[0] if isinstance(result, tuple) else result
            torch.autograd.backward(out, grad_out)
        return q_.grad, k_.grad, v_.grad, None, None, None


def _rowwise_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    needs_grad: bool,
) -> torch.Tensor:
    """Evaluate packed causal attention with ATOM decode's q_len=1 arithmetic.

    Every token is one query row over its own KV prefix, the reduction ATOM
    decode runs. All rows go through one launch per layer
    (``lumenrl.aiter_patch.qlen1_attention``).
    """
    if needs_grad:
        return _RowwiseAttention.apply(q, k, v, cu_seqlens, max_seqlen, softmax_scale)
    from lumenrl.aiter_patch.qlen1_attention import row_prefix_attention

    starts, lens = _packed_rows(cu_seqlens, q.shape[0])
    return row_prefix_attention(q, k, v, starts, lens, max_seqlen, softmax_scale)


def _varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """AITER varlen flash attention wrapper for HF's attention interface.

    When a packing context is active, uses ``flash_attn_varlen_func`` with
    ``cu_seqlens`` for correct cross-sequence isolation.  Otherwise delegates
    to the original attention function (Lumen's ``flash_attn_func`` path).
    """
    ctx = get_packing_context()
    if ctx is None:
        # Not in a packed forward — use original attention path
        return _original_attn_fn(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    cu_seqlens, max_seqlen = ctx

    from aiter import flash_attn_varlen_func

    # HF passes QKV as (B, H, S, D) where B=1 for packed input
    # Transpose to (B, S, H, D) then flatten to (total_tokens, H, D)
    B_dim, H_q, S_dim, D = query.shape
    H_k = key.shape[1]

    q = query.transpose(1, 2).reshape(-1, H_q, D)   # (total_tokens, H_q, D)
    k = key.transpose(1, 2).reshape(-1, H_k, D)     # (total_tokens, H_k, D)
    v = value.transpose(1, 2).reshape(-1, H_k, D)   # (total_tokens, H_k, D)

    softmax_scale = scaling if scaling is not None else D ** (-0.5)
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in [q, k, v]
    )

    if os.environ.get("LUMENRL_ROWWISE_TRAIN_ATTN", "0") == "1" and dropout == 0.0:
        attn_output = _rowwise_varlen_attention(
            q, k, v, cu_seqlens, max_seqlen, softmax_scale, needs_grad
        )
    else:
        result = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=needs_grad,
        )
        attn_output = result[0] if isinstance(result, tuple) else result
    # Reshape back to (B, S, H_q, D) for HF
    attn_output = attn_output.view(B_dim, S_dim, H_q, D).contiguous()
    return attn_output, None


def patch_attention_for_packing() -> bool:
    """Install varlen-aware attention as the HF ``sdpa`` backend.

    Must be called AFTER Lumen's ``patch_hf_sdpa()`` so that the original
    Lumen attention function is captured and used as the fallback.

    Returns True if the patch was applied, False otherwise.
    """
    global _original_attn_fn

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        logger.warning("Cannot import ALL_ATTENTION_FUNCTIONS; packing attention patch skipped.")
        return False

    # Save the current sdpa function (should be Lumen's
    # _lumen_aiter_attention_forward). Actor and frozen-reference workers can
    # share a process through an ActorRefFusedWorker, so initialization reaches
    # this function twice. Do not capture this wrapper as its own fallback on
    # the second call: the reference's non-packed forward would recurse forever.
    current_attn_fn = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if current_attn_fn is _varlen_attention_forward:
        logger.info("Packing: HF sdpa attention already uses the varlen patch.")
        return True

    _original_attn_fn = current_attn_fn
    if _original_attn_fn is None:
        logger.warning("No 'sdpa' attention function registered; packing patch skipped.")
        return False

    ALL_ATTENTION_FUNCTIONS["sdpa"] = _varlen_attention_forward
    logger.info("Packing: HF sdpa attention patched -> varlen-aware (AITER flash_attn_varlen_func)")
    return True
