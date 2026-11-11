"""Decode-shaped attention for many query rows in one launch.

ATOM decode scores a token with one q_len=1 ``flash_attn_varlen_func`` call:
one query row over its key prefix. Prefill and training reproduce that value
bit for bit only through the same q_len=1 specialization, which costs one
launch per token per layer when done row by row. This runs every row in one
call while keeping each row's arithmetic:

- ``cu_seqlens_k_padded`` holds each row's physical key start and
  ``cu_seqlens_k`` its real length, so rows read overlapping prefixes of one
  K/V buffer without copying them.
- The query heads that share a KV head attend to the same keys, so they enter
  as the rows of one non-causal sequence of length ``hq // hk``. K/V is then
  read once per KV head instead of once per query head.

Both were checked bit for bit against the one-row calls at lengths 127 to 4096
(``atom_stability/test_qlen1_batched_attn.py``).
"""

import torch


def row_prefix_attention(
    q,
    k,
    v,
    row_starts,
    row_lens,
    max_len,
    softmax_scale,
    window_size=(-1, -1, 0),
    sink_ptr=None,
):
    """Attention of query row ``i`` over ``k[row_starts[i] : row_starts[i] + row_lens[i]]``.

    ``q`` is ``(rows, hq, d)``; ``k``/``v`` are ``(keys, hk, d)``.
    ``row_starts`` and ``row_lens`` are int32 device tensors of length ``rows``.
    ``max_len`` is the host-side maximum of ``row_lens``.
    """
    from aiter import flash_attn_varlen_func

    rows, hq, dim = q.shape
    hk = k.shape[1]
    rep = hq // hk
    device = q.device
    cu_k = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(row_lens, 0, out=cu_k[1:])
    starts = torch.empty(rows + 1, dtype=torch.int32, device=device)
    starts[:rows] = row_starts
    starts[rows] = k.shape[0]
    cu_q = torch.arange(0, rep * rows + 1, rep, dtype=torch.int32, device=device)
    grouped = q.contiguous().view(rows, hk, rep, dim).transpose(1, 2).reshape(rows * rep, hk, dim)
    result = flash_attn_varlen_func(
        grouped,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=rep,
        max_seqlen_k=int(max_len),
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window_size,
        sink_ptr=sink_ptr,
        return_lse=True,
        cu_seqlens_k_padded=starts,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out.view(rows, rep, hk, dim).transpose(1, 2).reshape(rows, hq, dim)


def row_prefix_attention_planned(q, k, v, cu_q, cu_k, padded_starts, max_len, softmax_scale):
    """``row_prefix_attention`` with its index tensors already on the device.

    ``cu_q`` is ``arange(0, rep * rows + 1, rep)``, ``cu_k`` the prefix sums of
    the row lengths and ``padded_starts`` the row starts followed by
    ``k.shape[0]``, the values ``row_prefix_attention`` builds on every call.
    Decode builds them once per step and reuses them in every layer.
    """
    from aiter import flash_attn_varlen_func

    rows, hq, dim = q.shape
    hk = k.shape[1]
    rep = hq // hk
    grouped = q.contiguous().view(rows, hk, rep, dim).transpose(1, 2).reshape(rows * rep, hk, dim)
    result = flash_attn_varlen_func(
        grouped,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=rep,
        max_seqlen_k=int(max_len),
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1, 0),
        sink_ptr=None,
        return_lse=True,
        cu_seqlens_k_padded=padded_starts,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out.view(rows, rep, hk, dim).transpose(1, 2).reshape(rows, hq, dim)


def packed_causal_rows(cu_seqlens, total):
    """Row starts and lengths of causal attention over packed sequences.

    Row ``t`` of sequence ``s`` attends to ``[cu_seqlens[s], t]``. Everything is
    computed on the device, so no host synchronization is needed.
    """
    device = cu_seqlens.device
    index = torch.arange(total, dtype=torch.int32, device=device)
    seq = torch.searchsorted(cu_seqlens[1:].to(torch.int32), index, right=True)
    starts = cu_seqlens.to(torch.int32)[seq]
    return starts, index - starts + 1
