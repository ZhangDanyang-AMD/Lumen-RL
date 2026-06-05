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
from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
) -> PackedBatch:
    """Pack left-padded sequences into a contiguous flat tensor.

    Args:
        input_ids: ``(B, S)`` left-padded token IDs.
        attention_mask: ``(B, S)`` where 1 = real token, 0 = pad.

    Returns:
        A :class:`PackedBatch` with concatenated tokens, cumulative sequence
        lengths, per-sequence position IDs, and per-sequence actual lengths.
    """
    B, S = input_ids.shape
    device = input_ids.device

    seq_lens = attention_mask.sum(dim=1).long()          # (B,)
    total_tokens = int(seq_lens.sum().item())
    max_seqlen = int(seq_lens.max().item())

    packed_ids = torch.empty(total_tokens, dtype=input_ids.dtype, device=device)
    position_ids = torch.empty(total_tokens, dtype=torch.long, device=device)

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
        input_ids=packed_ids.unsqueeze(0),       # (1, total_tokens)
        position_ids=position_ids.unsqueeze(0),  # (1, total_tokens)
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
    )


def packed_token_log_probs(
    logits: torch.Tensor,
    packed_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probs from packed (flat) logits.

    Performs the shifted log_softmax + gather per sequence, vectorized.

    Args:
        logits: ``(total_tokens, V)`` model output (squeezed from batch dim).
        packed_ids: ``(total_tokens,)`` packed token IDs.
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.

    Returns:
        Flat ``(total_tokens - B,)`` float32 tensor of shifted log-probs.
        Each sequence *i* contributes ``sl_i - 1`` values.
    """
    total_tokens = logits.shape[0]
    B = cu_seqlens.shape[0] - 1
    device = logits.device

    # Build masks to exclude boundary tokens for shifted prediction
    # not_last[j] = True if j is NOT the last token of any sequence
    not_last = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_last[cu_seqlens[1:].long() - 1] = False

    # not_first[j] = True if j is NOT the first token of any sequence
    not_first = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_first[cu_seqlens[:-1].long()] = False

    # Shifted logits: all positions except the last token of each sequence
    shifted_logits = logits[not_last]                          # (total - B, V)
    shifted_targets = packed_ids[not_first].unsqueeze(-1)      # (total - B, 1)

    # Per-row log_softmax (bf16-safe, matches _fused_token_log_probs pattern)
    # Process in chunks to bound peak memory from float32 promotion
    chunk_size = 4096
    lp_parts = []
    for start in range(0, shifted_logits.shape[0], chunk_size):
        end = min(start + chunk_size, shifted_logits.shape[0])
        row_lp = F.log_softmax(shifted_logits[start:end], dim=-1)
        lp_parts.append(
            row_lp.gather(-1, shifted_targets[start:end]).squeeze(-1)
        )
    return torch.cat(lp_parts, dim=0).float()  # (total - B,)


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

    # Save the current sdpa function (should be Lumen's _lumen_aiter_attention_forward)
    _original_attn_fn = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if _original_attn_fn is None:
        logger.warning("No 'sdpa' attention function registered; packing patch skipped.")
        return False

    ALL_ATTENTION_FUNCTIONS["sdpa"] = _varlen_attention_forward
    logger.info("Packing: HF sdpa attention patched -> varlen-aware (AITER flash_attn_varlen_func)")
    return True
