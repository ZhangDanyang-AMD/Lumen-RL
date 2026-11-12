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
    rows, hq, dim = q.shape
    hk = k.shape[1]
    rep = hq // hk
    grouped = q.contiguous().view(rows, hk, rep, dim).transpose(1, 2).reshape(rows * rep, hk, dim)
    out, _ = _varlen_fwd(
        grouped, k, v, cu_q, cu_k, rep, max_len, softmax_scale, False, k_padded=padded_starts
    )
    return out.view(rows, rep, hk, dim).transpose(1, 2).reshape(rows, hq, dim)


# Must not exceed the Q tile of the hd128 bf16 asm group kernels (256 rows in
# hsa/gfx950/fmha_v3_fwd/fmha_fwd.csv). A batch entry that fits in one Q tile
# takes the same per-row path as a q_len=1 call; a longer causal entry
# schedules its second half of Q tiles differently and loses that equality.
CAUSAL_CHUNK = 256

# Split-KV: a row with more than SPLIT_KEYS keys attends to each SPLIT_KEYS
# block of its prefix, counted from the sequence's first key, in its own batch
# entry; ``_merge_partials`` combines the blocks in order. A row with at most
# SPLIT_KEYS keys is one entry and needs no merge. The split depends on the
# row's own position only, so decode, prefill and training split it alike.
# A multiple of CAUSAL_CHUNK, so no chunk straddles a block boundary.
SPLIT_KEYS = 1024

_merge_kernel = None
# Compiled merge kernels by argument dtypes and 16-byte pointer alignment, the
# properties Triton specializes on. Launching a compiled kernel skips about
# 8 us of JIT dispatch per call.
_merge_compiled = {}


def _varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal,
                out=None, q_padded=None, k_padded=None):
    """(out, lse) of aiter varlen attention without dropout, window or sink.

    For bf16 hd128 this is the fmha_v3 call ``flash_attn_varlen_func`` ends
    in, with the same arguments, minus about 13 us of Python dispatch.
    """
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        from aiter import flash_attn_varlen_func

        result = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            min_seqlen_q=0,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1, 0),
            sink_ptr=None,
            return_lse=True,
            out=out,
            cu_seqlens_q_padded=q_padded,
            cu_seqlens_k_padded=k_padded,
        )
        return result[0], result[1]
    from aiter.ops.mha import fmha_v3_varlen_fwd

    result = fmha_v3_varlen_fwd(
        q, k, v, cu_q, cu_k, int(max_q), int(max_k),
        0,  # min_seqlen_q
        0.0,  # dropout_p
        softmax_scale,
        0.0,  # logits_soft_cap
        False,  # zero_tensors
        causal,
        -1, -1,  # window
        True,  # return_softmax_lse
        False,  # return_dropout_randval
        1,  # how_v3_bf16_cvt
        out,
        None, None, None,  # block_table, bias, alibi_slopes
        None, None, None,  # q/k/v descale
        None,  # gen
        q_padded,
        k_padded,
    )
    return result[0], result[1]


def _merge_partials(o, lse, o_strides, l_strides, idx, npart, dst, out, rep):
    """``out[dst[r]] = sum_p w_p * o_p / sum_p w_p`` over row ``r``'s partials.

    ``idx[r, :npart[r]]`` are the partial entries of row ``r`` in block order
    and ``w_p = exp(lse_p - max lse)``. Query head ``h`` of entry ``e`` sits
    at ``e * se + (h // rep) * skv + (h % rep) * sj`` in ``o`` (strides
    ``o_strides``) and in ``lse`` (``l_strides``), which covers the grouped
    decode layout and the flat training layout with one kernel.
    """
    global _merge_kernel
    if _merge_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["o_se", "o_skv", "o_sj", "l_se", "l_skv", "l_sj", "idx_s", "out_sr"])
        def kernel(o_ptr, lse_ptr, idx_ptr, np_ptr, dst_ptr, out_ptr,
                   o_se, o_skv, o_sj, l_se, l_skv, l_sj, idx_s, out_sr,
                   REP: tl.constexpr, D: tl.constexpr):
            r = tl.program_id(0)
            h = tl.program_id(1)
            kvh = h // REP
            j = h % REP
            n = tl.load(np_ptr + r)
            offs = tl.arange(0, D)
            m = tl.full((), float("-inf"), tl.float32)
            for p in range(0, n):
                e = tl.load(idx_ptr + r * idx_s + p).to(tl.int64)
                m = tl.maximum(m, tl.load(lse_ptr + e * l_se + kvh * l_skv + j * l_sj))
            s = tl.zeros((), tl.float32)
            acc = tl.zeros((D,), tl.float32)
            for p in range(0, n):
                e = tl.load(idx_ptr + r * idx_s + p).to(tl.int64)
                w = tl.exp(tl.load(lse_ptr + e * l_se + kvh * l_skv + j * l_sj) - m)
                part = tl.load(o_ptr + e * o_se + kvh * o_skv + j * o_sj + offs).to(tl.float32)
                s += w
                acc += w * part
            row = tl.load(dst_ptr + r).to(tl.int64)
            tl.store(out_ptr + row * out_sr + h * D + offs, (acc / s).to(out_ptr.dtype.element_ty))

        _merge_kernel = kernel
    heads, dim = out.shape[1], out.shape[2]
    tensors = (o, lse, idx, npart, dst, out)
    args = (*tensors, *o_strides, *l_strides, idx.stride(0), out.stride(0))
    grid = (idx.shape[0], heads, 1)
    key = (rep, dim) + tuple((t.dtype, t.data_ptr() % 16 == 0) for t in tensors)
    compiled = _merge_compiled.get(key)
    if compiled is None:
        compiled = _merge_kernel.warmup(*args, REP=rep, D=dim, grid=grid)
        _merge_compiled[key] = compiled
    compiled[grid](*args, rep, dim, stream=torch.cuda.current_stream().cuda_stream)


_copy_kernel = None
_copy_compiled = {}


def pool_write(k_new, v_new, dst, k_pool, v_pool):
    """``k_pool[dst[i]] = k_new[i]`` and ``v_pool[dst[i]] = v_new[i]`` in one launch.

    ``k_new``/``v_new`` are ``(rows, hk, d)`` with contiguous rows (any row
    stride); the pools are contiguous.
    """
    global _copy_kernel
    if _copy_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["k_sr", "v_sr"])
        def kernel(k_ptr, v_ptr, dst_ptr, kp_ptr, vp_ptr, k_sr, v_sr, ROW: tl.constexpr):
            r = tl.program_id(0)
            offs = tl.arange(0, ROW)
            row = tl.load(dst_ptr + r).to(tl.int64)
            if tl.program_id(1) == 0:
                tl.store(kp_ptr + row * ROW + offs, tl.load(k_ptr + r.to(tl.int64) * k_sr + offs))
            else:
                tl.store(vp_ptr + row * ROW + offs, tl.load(v_ptr + r.to(tl.int64) * v_sr + offs))

        _copy_kernel = kernel
    rows = k_new.shape[0]
    width = k_new.shape[1] * k_new.shape[2]
    tensors = (k_new, v_new, dst, k_pool, v_pool)
    args = (*tensors, k_new.stride(0), v_new.stride(0))
    grid = (rows, 2, 1)
    key = (width,) + tuple((t.dtype, t.data_ptr() % 16 == 0) for t in tensors)
    compiled = _copy_compiled.get(key)
    if compiled is None:
        compiled = _copy_kernel.warmup(*args, ROW=width, grid=grid)
        _copy_compiled[key] = compiled
    compiled[grid](*args, width, stream=torch.cuda.current_stream().cuda_stream)


def _upload(parts, device):
    """Host int lists as int32 device views of one pinned buffer."""
    flat = [x for part in parts for x in part]
    host = torch.tensor(flat, dtype=torch.int32).pin_memory()
    buf = host.to(device, non_blocking=True)
    views, offset = [], 0
    for part in parts:
        views.append(buf[offset : offset + len(part)])
        offset += len(part)
    return views


def _pad_rows(lists):
    width = max(len(x) for x in lists)
    return [x for row in lists for x in row + [0] * (width - len(row))], width


def causal_chunk_meta(q_bounds, k_bounds, total_keys, device):
    """Plan of ``chunked_causal_attention`` for packed sequences.

    ``q_bounds`` and ``k_bounds`` are the host-side sequence boundaries of the
    packed query and key rows; a sequence may have more keys than queries, the
    extra ones being a cached prefix. Every sequence is cut into chunks of
    ``CAUSAL_CHUNK`` query rows. A chunk whose rows hold at most
    ``SPLIT_KEYS`` keys is one causal entry over the whole prefix. Otherwise
    the chunk's last block is a causal entry from the block start, and every
    full block before it is a non-causal entry over a copy of the chunk's
    queries.
    """
    ca_q, ca_k, ca_start = [0], [0], []
    nc_src, nc_q, nc_k, nc_start = [], [0], [0], []
    merged = []  # (row, [non-causal partial rows]) of rows with more than one block
    max_q = max_k = 1
    for s in range(len(q_bounds) - 1):
        a, b = q_bounds[s], q_bounds[s + 1]
        ka = k_bounds[s]
        prefix = k_bounds[s + 1] - ka - (b - a)
        c = a
        while c < b:
            first = prefix + c - a  # keys before the chunk's first row
            # A cached prefix can put a block boundary inside a chunk; cut there.
            n = min(CAUSAL_CHUNK, b - c, SPLIT_KEYS - first % SPLIT_KEYS)
            last = first + n  # keys of the chunk's last row
            blocks = first // SPLIT_KEYS
            base = []
            for blk in range(blocks):
                base.append(len(nc_src))
                nc_src.extend(range(c, c + n))
                nc_q.append(nc_q[-1] + n)
                nc_k.append(nc_k[-1] + SPLIT_KEYS)
                nc_start.append(ka + blk * SPLIT_KEYS)
            ca_q.append(ca_q[-1] + n)
            ca_k.append(ca_k[-1] + last - blocks * SPLIT_KEYS)
            ca_start.append(ka + blocks * SPLIT_KEYS)
            max_q = max(max_q, n)
            max_k = max(max_k, last - blocks * SPLIT_KEYS)
            if blocks:
                merged.extend((c + i, [b0 + i for b0 in base]) for i in range(n))
            c += n
    ca_start.append(total_keys)
    nc_start.append(total_keys)
    parts = [ca_q, ca_k, ca_start]
    plan = {"max_q": max_q, "max_k": max_k, "partials": len(nc_src)}
    if merged:
        rows_total = q_bounds[-1]
        idx, width = _pad_rows([part + [len(nc_src) + row] for row, part in merged])
        npart = [len(part) + 1 for _row, part in merged]
        dst = [row for row, _part in merged]
        parts += [nc_src, nc_q, nc_k, nc_start, idx, npart, dst]
        plan["nc_max_q"] = max(nc_q[i + 1] - nc_q[i] for i in range(len(nc_q) - 1))
        plan["width"] = width
        plan["rows_total"] = rows_total
    views = _upload(parts, device)
    plan["ca"] = views[:3]
    if merged:
        nc_src_t, nc_q_t, nc_k_t, nc_start_t, idx_t, np_t, dst_t = views[3:]
        plan["nc"] = (nc_src_t.long(), nc_q_t, nc_k_t, nc_start_t)
        plan["merge"] = (idx_t.view(-1, plan["width"]), np_t, dst_t)
    return plan


def chunked_causal_attention(q, k, v, plan, softmax_scale):
    """Causal attention over packed sequences with q_len=1 arithmetic per row.

    Rows with at most ``SPLIT_KEYS`` keys get the values of
    ``row_prefix_attention`` over ``packed_causal_rows`` bit for bit
    (``opus_splitk_probe/chunked_causal_bitwise.py``); longer rows get the
    split-KV values decode computes with ``decode_split_attention``
    (``opus_splitk_probe/splitkv_bitwise.py``). ``plan`` comes from
    ``causal_chunk_meta``.
    """
    rows, hq, dim = q.shape
    n1 = plan["partials"]
    o = torch.empty((n1 + rows, hq, dim), dtype=q.dtype, device=q.device)
    out = o[n1:]
    cu_q, cu_k, k_start = plan["ca"]
    _, lse_ca = _varlen_fwd(
        q, k, v, cu_q, cu_k, plan["max_q"], plan["max_k"], softmax_scale, True,
        out=out, q_padded=cu_q, k_padded=k_start,
    )
    if "merge" not in plan:
        return out
    nc_src, nc_q, nc_k, nc_start = plan["nc"]
    _, lse_nc = _varlen_fwd(
        q.index_select(0, nc_src), k, v, nc_q, nc_k, plan["nc_max_q"], SPLIT_KEYS,
        softmax_scale, False, out=o[:n1], q_padded=nc_q, k_padded=nc_start,
    )
    lse = torch.cat([lse_nc, lse_ca], 1)
    rep = hq // k.shape[1]
    idx, npart, dst = plan["merge"]
    # The kernel reads every partial of a row before it writes that row, so
    # writing over the causal partials in place is safe.
    _merge_partials(
        o, lse, (hq * dim, rep * dim, dim), (1, rep * lse.stride(0), lse.stride(0)),
        idx, npart, dst, out, rep,
    )
    return out


def decode_split_plan(src, starts, lens, pool_rows, rep, device):
    """Plan of ``decode_split_attention``; ``None`` if no row needs a split.

    Query row ``src[i]`` attends to ``lens[i]`` keys from pool row
    ``starts[i]``. Every ``SPLIT_KEYS`` block of a row is one grouped entry.
    """
    if max(lens, default=0) <= SPLIT_KEYS:
        return None
    gather, cu_q, cu_k, k_start, idx = [], [0], [0], [], []
    for row, start, length in zip(src, starts, lens):
        mine = []
        for p in range(0, length, SPLIT_KEYS):
            mine.append(len(gather))
            gather.append(row)
            cu_q.append(cu_q[-1] + rep)
            cu_k.append(cu_k[-1] + min(SPLIT_KEYS, length - p))
            k_start.append(start + p)
        idx.append(mine)
    k_start.append(pool_rows)
    flat_idx, width = _pad_rows(idx)
    views = _upload([gather, cu_q, cu_k, k_start, flat_idx, [len(x) for x in idx], list(src)], device)
    return {
        "gather": views[0].long(),
        "attn": views[1:4],
        "merge": (views[4].view(-1, width), views[5], views[6]),
        "entries": len(gather),
    }


def decode_split_attention(q, k_pool, v_pool, plan, softmax_scale):
    """Decode rows over the pool, split as ``causal_chunk_meta`` splits them.

    The grouped layout of ``row_prefix_attention`` (query heads of one KV head
    as the rows of one entry) with each row repeated once per block. The merge
    writes every row back to its own index of ``q``.
    """
    rows, hq, dim = q.shape
    hk = k_pool.shape[1]
    rep = hq // hk
    grouped = q.contiguous().view(rows, hk, rep, dim).transpose(1, 2).index_select(0, plan["gather"])
    cu_q, cu_k, k_start = plan["attn"]
    o, lse = _varlen_fwd(
        grouped.reshape(-1, hk, dim), k_pool, v_pool, cu_q, cu_k, rep, SPLIT_KEYS,
        softmax_scale, False, k_padded=k_start,
    )
    out = torch.empty_like(q)
    idx, npart, dst = plan["merge"]
    _merge_partials(
        o, lse, (rep * hk * dim, dim, hk * dim), (rep, lse.stride(0), 1),
        idx, npart, dst, out, rep,
    )
    return out


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
