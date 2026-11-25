"""Default-off graph-capturable aligned decode (candidate C4).

Why
---
Every aligned arm runs eager today because ``validate_aligned_attention_config``
requires ``enforce_eager`` and ``_batch`` / ``_gather_live_kv`` read decode
metadata back to the host (``.cpu().tolist()``), build index tensors with
``torch.tensor`` and size the packed KV from host lengths.  None of that can be
recorded in a HIP graph, and at 28 seq/GPU it costs ~27 ms of host time per
decode step against ~4 ms of extra GPU work.

What changes with ``LUMEN_ATOM_ALIGNED_GRAPH=1``
-----------------------------------------------
* The runner may build the aligned engine with ``enforce_eager=False`` and
  ``compilation_config={"level": 0, "cudagraph_capture_sizes": [1..max_num_seqs]}``
  (:func:`engine_overrides`).  Level 0 keeps torch.compile off; ATOM then
  records one FULL decode ``ReqGraph`` per batch size and keeps prefill eager.
* Inside ATOM's capture loop (``in_hipgraph``) aligned decode attention uses
  :func:`graph_decode_attention`: the per-row K lengths, K offsets and the
  packed K/V are produced on the GPU from ATOM's own device metadata
  (``context_lens``, ``slot_mapping``, ``block_tables``) into persistent
  buffers, and the same pinned Triton varlen kernel is launched with
  ``max_seqlen_q=1`` and ``max_seqlen_k=max_model_len``.  ``SEQLEN_K`` is a
  runtime (non-constexpr) argument that the VARLEN kernel does not read, so the
  per-row math is the eager math.
* The host validation that ``_batch`` performed per layer is performed once per
  replayed step on the host mirrors ATOM already filled (``forward_vars[*].np``)
  before replay (:func:`validate_decode_host`), so no device readback is needed.
* Prefill, eager decode (e.g. above the largest capture size) and every path
  with the switch off are unchanged.
* ``LUMEN_ATOM_ALIGNED_GRAPH_PACK_LOOP=1`` (default off) launches the same KV copy
  on a fixed (rows, 256) grid that strides over each row's live tokens instead of
  one program per 16-token tile of ``max_model_len``; the packed K/V is bitwise
  identical and the grid is still fixed at capture.
* ``LUMEN_ATOM_ALIGNED_GRAPH_PAGED=1`` (default off) skips the pack and runs
  ``aligned_paged_attention`` (generated fork of the pinned kernel whose K/V tile
  loads read the paged cache through the block table). The packed buffers stay
  allocated so ATOM's KV capacity is unchanged; the first uncaptured call per
  device must equal pack + pinned varlen bitwise or the run stops.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

_LOG = logging.getLogger(__name__)

SWITCH = "LUMEN_ATOM_ALIGNED_GRAPH"
HOSTCHECK_SWITCH = "LUMEN_ATOM_ALIGNED_GRAPH_HOSTCHECK"
PACK_LOOP_SWITCH = "LUMEN_ATOM_ALIGNED_GRAPH_PACK_LOOP"
ENABLED = os.environ.get(SWITCH, "0") == "1"
HOSTCHECK = os.environ.get(HOSTCHECK_SWITCH, "1") == "1"
PACK_LOOP = os.environ.get(PACK_LOOP_SWITCH, "0") == "1"
PAGED_SWITCH = "LUMEN_ATOM_ALIGNED_GRAPH_PAGED"
PAGED = os.environ.get(PAGED_SWITCH, "0") == "1"

_HEADS = 8
_HEAD_DIM = 128
_DIM_PACK = 16
_PACK = 8
_TOKENS_PER_PROGRAM = 16

_LOOP_TOKENS = 32
_LOOP_PROGRAMS = 256

_LOCK = threading.Lock()
_BUFFERS = {}
_GEOMETRY = {}
_KERNEL = None
_LOOP_KERNEL = None
_PAGED_CHECKED = set()


class AlignedGraphError(RuntimeError):
    """The selected graph-capturable aligned decode contract cannot be met."""


def enabled() -> bool:
    return ENABLED


def engine_overrides(mode: str, max_num_seqs: int) -> dict:
    """Engine kwargs that turn on decode-only FULL graphs for the aligned arm."""
    if not ENABLED:
        return {}
    if mode != "aligned":
        raise AlignedGraphError(f"{SWITCH}=1 is defined only for the aligned arm")
    if type(max_num_seqs) is not int or max_num_seqs <= 0:
        raise AlignedGraphError("aligned graph capture needs a positive max_num_seqs")
    return {
        "enforce_eager": False,
        "compilation_config": {
            "level": 0,
            "cudagraph_capture_sizes": list(range(1, max_num_seqs + 1)),
        },
    }


def is_graph_decode(fwd_ctx) -> bool:
    """True only for a decode forward ATOM runs inside its capture loop."""
    if not ENABLED:
        return False
    context = getattr(fwd_ctx, "context", None)
    if context is None or bool(getattr(context, "is_prefill", True)):
        return False
    if getattr(fwd_ctx, "in_hipgraph", False):
        return True
    mode = getattr(context, "forward_mode", None)
    return bool(mode is not None and mode.use_cudagraph)


def _check_static_contract(impl, q, key_cache, value_cache, meta):
    """Shape/dtype checks on Python-side values only; no tensor data is read."""
    import torch

    if getattr(meta, "max_seqlen_q", None) != 1:
        raise AlignedGraphError("graph aligned decode requires max_seqlen_q == 1")
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, _HEAD_DIM) or q.dtype != torch.bfloat16:
        raise AlignedGraphError("graph aligned decode requires BF16 Q[T,32,128]")
    if key_cache.ndim != 5 or value_cache.ndim != 5:
        raise AlignedGraphError("graph aligned decode requires native 5D SHUFFLE caches")
    pages, heads, dim_pack, page, pack = key_cache.shape
    if (heads, dim_pack, pack) != (_HEADS, _DIM_PACK, _PACK) or page <= 0 or page % _PACK:
        raise AlignedGraphError("invalid physical K cache")
    if tuple(value_cache.shape) != (pages, _HEADS, page // _PACK, _HEAD_DIM, _PACK):
        raise AlignedGraphError("invalid physical SHUFFLE V cache")
    for cache in (key_cache, value_cache):
        if cache.dtype != torch.bfloat16 or cache.device != q.device:
            raise AlignedGraphError("KV cache dtype/device differs from Q")
    rows = len(q)
    context_lens, slots = meta.context_lens, meta.slot_mapping
    table, cuq = meta.block_tables, meta.cu_seqlens_q
    for name, tensor in (("context_lens", context_lens), ("slot_mapping", slots),
                         ("block_tables", table), ("cu_seqlens_q", cuq)):
        if not isinstance(tensor, torch.Tensor) or tensor.device != q.device:
            raise AlignedGraphError(f"{name} must be a device tensor on the query GPU")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise AlignedGraphError(f"{name} must be integer metadata")
    if context_lens.ndim != 1 or len(context_lens) < rows:
        raise AlignedGraphError("context_lens does not cover the graph rows")
    if slots.ndim != 1 or len(slots) < rows:
        raise AlignedGraphError("slot_mapping does not cover the graph rows")
    if table.ndim != 2 or len(table) < rows:
        raise AlignedGraphError("block_tables does not cover the graph rows")
    if cuq.ndim != 1 or len(cuq) < rows + 1:
        raise AlignedGraphError("cu_seqlens_q does not cover the graph rows")
    max_len = impl._aligned_attention_max_len
    if table.shape[1] * page < max_len:
        raise AlignedGraphError("block table width cannot address max_model_len")
    return rows, pages, page, max_len


class _GraphBuffers:
    """Persistent per-device storage; allocated before any stream capture."""

    def __init__(self, device, max_rows, max_len):
        import torch

        self.max_rows = max_rows
        self.max_len = max_len
        tokens = max_rows * max_len
        self.k = torch.empty((tokens, _HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)
        self.klen = torch.ones(max_rows, dtype=torch.int32, device=device)
        self.valid = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.cuk = torch.zeros(max_rows + 1, dtype=torch.int32, device=device)


def _buffers(device, max_rows, max_len, capturing):
    key = (device.type, device.index)
    with _LOCK:
        buffers = _BUFFERS.get(key)
        if buffers is None:
            if capturing:
                raise AlignedGraphError(
                    "graph aligned buffers must be allocated by the warmup forward, "
                    "not inside stream capture"
                )
            buffers = _GraphBuffers(device, max_rows, max_len)
            _BUFFERS[key] = buffers
        elif buffers.max_len != max_len or buffers.max_rows < max_rows:
            raise AlignedGraphError("graph aligned buffers were sized for another contract")
        return buffers


def device_row_plan(context_lens, slot_mapping, rows, klen, valid, cuk):
    """Write per-row K lengths, validity and K offsets in place (capture-safe).

    A row is real iff it writes a KV slot and has a positive context; ATOM sets
    padded rows to ``slot=-1`` and ``context_len=0``.  Padded rows get one
    zeroed K/V token so their attention output is exactly zero, matching the
    eager path's zero padding, and they never read a block-table entry.
    """
    import torch

    real = (slot_mapping[:rows] >= 0) & (context_lens[:rows] > 0)
    valid[:rows].copy_(real)
    klen[:rows].copy_(context_lens[:rows])
    klen[:rows].masked_fill_(~real, 1)
    torch.cumsum(klen[:rows], 0, dtype=torch.int32, out=cuk[1:rows + 1])
    return real


def reference_pack(key_cache, value_cache, block_tables, klen, valid, cuk, rows, out_k, out_v):
    """Torch reference for the Triton pack (tests only; same index math)."""
    import torch

    page = key_cache.shape[3]
    keys = key_cache.permute(0, 3, 1, 2, 4).reshape(key_cache.shape[0], page, _HEADS, _HEAD_DIM)
    values = value_cache.permute(0, 2, 4, 1, 3).reshape(value_cache.shape[0], page, _HEADS, _HEAD_DIM)
    for row in range(rows):
        n, start = int(klen[row]), int(cuk[row])
        if not int(valid[row]):
            out_k[start:start + n].zero_()
            out_v[start:start + n].zero_()
            continue
        tok = torch.arange(n)
        phy = block_tables[row, tok // page].long()
        out_k[start:start + n] = keys[phy, tok % page]
        out_v[start:start + n] = values[phy, tok % page]


def _pack_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _graph_pack(
        k_ptr, v_ptr, out_k_ptr, out_v_ptr, table_ptr, table_stride,
        klen_ptr, valid_ptr, cuk_ptr,
        page: tl.constexpr, page_pack: tl.constexpr, TOK: tl.constexpr,
    ):
        row = tl.program_id(0)
        start = tl.program_id(1) * TOK
        n = tl.load(klen_ptr + row)
        if start < n:
            real = tl.load(valid_ptr + row) != 0
            base = tl.load(cuk_ptr + row).to(tl.int64)
            tok = start + tl.arange(0, TOK)
            live = tok < n
            phy = tl.load(table_ptr + row * table_stride + tok // page,
                          mask=live & real, other=0).to(tl.int64)
            pos = (tok % page).to(tl.int64)
            heads = tl.arange(0, 8)[None, :, None].to(tl.int64)
            dims = tl.arange(0, 128)[None, None, :].to(tl.int64)
            phy3 = phy[:, None, None]
            pos3 = pos[:, None, None]
            k_src = (phy3 * (8 * 16 * page * 8) + heads * (16 * page * 8)
                     + (dims // 8) * (page * 8) + pos3 * 8 + (dims % 8))
            v_src = (phy3 * (8 * page_pack * 128 * 8) + heads * (page_pack * 128 * 8)
                     + (pos3 // 8) * (128 * 8) + dims * 8 + (pos3 % 8))
            load_mask = (live & real)[:, None, None]
            store_mask = live[:, None, None]
            dst = (base + tok.to(tl.int64))[:, None, None] * (8 * 128) + heads * 128 + dims
            tl.store(out_k_ptr + dst, tl.load(k_ptr + k_src, mask=load_mask, other=0.0), mask=store_mask)
            tl.store(out_v_ptr + dst, tl.load(v_ptr + v_src, mask=load_mask, other=0.0), mask=store_mask)

    return _graph_pack


def _pack_loop_kernel():
    """Same copy as ``_graph_pack`` with a fixed (rows, NPROG) grid that strides over each
    row's live tokens, instead of one program per TOK tile of ``max_len`` (mostly empty)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _graph_pack_loop(
        k_ptr, v_ptr, out_k_ptr, out_v_ptr, table_ptr, table_stride,
        klen_ptr, valid_ptr, cuk_ptr,
        page: tl.constexpr, page_pack: tl.constexpr, TOK: tl.constexpr, NPROG: tl.constexpr,
    ):
        row = tl.program_id(0)
        n = tl.load(klen_ptr + row)
        real = tl.load(valid_ptr + row) != 0
        base = tl.load(cuk_ptr + row).to(tl.int64)
        heads = tl.arange(0, 8)[None, :, None].to(tl.int64)
        dims = tl.arange(0, 128)[None, None, :].to(tl.int64)
        for start in range(tl.program_id(1) * TOK, n, NPROG * TOK):
            tok = start + tl.arange(0, TOK)
            live = tok < n
            phy = tl.load(table_ptr + row * table_stride + tok // page,
                          mask=live & real, other=0).to(tl.int64)
            pos = (tok % page).to(tl.int64)
            phy3 = phy[:, None, None]
            pos3 = pos[:, None, None]
            k_src = (phy3 * (8 * 16 * page * 8) + heads * (16 * page * 8)
                     + (dims // 8) * (page * 8) + pos3 * 8 + (dims % 8))
            v_src = (phy3 * (8 * page_pack * 128 * 8) + heads * (page_pack * 128 * 8)
                     + (pos3 // 8) * (128 * 8) + dims * 8 + (pos3 % 8))
            load_mask = (live & real)[:, None, None]
            store_mask = live[:, None, None]
            dst = (base + tok.to(tl.int64))[:, None, None] * (8 * 128) + heads * 128 + dims
            tl.store(out_k_ptr + dst, tl.load(k_ptr + k_src, mask=load_mask, other=0.0), mask=store_mask)
            tl.store(out_v_ptr + dst, tl.load(v_ptr + v_src, mask=load_mask, other=0.0), mask=store_mask)

    return _graph_pack_loop


def _launch_pack_loop(key_cache, value_cache, block_tables, buffers, rows, page):
    global _LOOP_KERNEL
    if _LOOP_KERNEL is None:
        _LOOP_KERNEL = _pack_loop_kernel()
    _LOOP_KERNEL[(rows, _LOOP_PROGRAMS)](
        key_cache, value_cache, buffers.k, buffers.v,
        block_tables, block_tables.stride(0),
        buffers.klen, buffers.valid, buffers.cuk,
        page=page, page_pack=page // _PACK, TOK=_LOOP_TOKENS, NPROG=_LOOP_PROGRAMS, num_warps=8,
    )


def _launch_pack(key_cache, value_cache, block_tables, buffers, rows, page):
    global _KERNEL
    if PACK_LOOP:
        return _launch_pack_loop(key_cache, value_cache, block_tables, buffers, rows, page)
    if _KERNEL is None:
        _KERNEL = _pack_kernel()
    tiles = -(-buffers.max_len // _TOKENS_PER_PROGRAM)
    _KERNEL[(rows, tiles)](
        key_cache, value_cache, buffers.k, buffers.v,
        block_tables, block_tables.stride(0),
        buffers.klen, buffers.valid, buffers.cuk,
        page=page, page_pack=page // _PACK, TOK=_TOKENS_PER_PROGRAM, num_warps=4,
    )


def _paged_runtime(varlen, device):
    """Pinned config and aiter launch globals, read from the same verified runtime ``varlen`` uses."""
    import sys

    from aiter.ops.triton.attention import mha as mha_api

    am = sys.modules[varlen.__module__]
    path, digest = am._policy_key()
    state = am._RUNTIME.get((device.index, path, digest))
    if state is None:
        raise AlignedGraphError("paged aligned decode needs the verified pinned Attention runtime")
    return (dict(state["config"]), mha_api.get_num_xcds(), mha_api._MHA_SWIZZLE,
            bool(mha_api._USE_INT64_STRIDES))


def _paged_decode(impl, q, key_cache, value_cache, block_tables, buffers, rows, page, max_len, cuq,
                  varlen, capturing):
    """Paged fork over the cache. The packed buffers stay allocated (unused) so KV capacity is
    unchanged; the first uncaptured call per device must match pack+pinned bitwise."""
    import torch

    from lumenrl.engine.inference.atom_patch.aligned_paged_attention import paged_varlen

    config, num_xcd, swizzle, int64 = _paged_runtime(varlen, q.device)
    cuk = buffers.cuk[:rows + 1]
    output = paged_varlen(q, key_cache, value_cache, block_tables, buffers.valid, cuq, cuk, max_len,
                          impl.scale, config, num_xcd, swizzle, int64)
    key = (q.device.type, q.device.index)
    if key not in _PAGED_CHECKED:
        if capturing:
            raise AlignedGraphError("paged aligned decode self-check must run before stream capture")
        _launch_pack(key_cache, value_cache, block_tables, buffers, rows, page)
        reference = varlen(q, buffers.k, buffers.v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                           max_seqlen_q=1, max_seqlen_k=max_len,
                           dropout_p=0.0, softmax_scale=impl.scale, causal=True)
        if not torch.equal(output.view(torch.int16), reference.view(torch.int16)):
            raise AlignedGraphError("paged aligned decode differs from pack+pinned varlen")
        _PAGED_CHECKED.add(key)
        import sys

        print(f"LUMEN_ALIGNED_PAGED_SELFCHECK bitwise_equal device={key} rows={rows}", file=sys.stderr, flush=True)
    return output


def graph_decode_attention(impl, q, key_cache, value_cache, fwd_ctx, varlen, pack=None):
    """Capture-safe aligned decode attention over ATOM's device metadata.

    ``varlen`` is ``alignment_main._pinned_varlen`` so the kernel, its fixed
    config and the per-call policy checks are exactly the eager ones.
    """
    import torch

    meta = fwd_ctx.attn_metadata
    rows, pages, page, max_len = _check_static_contract(impl, q, key_cache, value_cache, meta)
    max_rows = getattr(impl, "_aligned_graph_max_rows", None)
    if type(max_rows) is not int or rows > max_rows:
        raise AlignedGraphError("graph rows exceed the aligned graph buffer capacity")
    capturing = q.is_cuda and torch.cuda.is_current_stream_capturing()
    buffers = _buffers(q.device, max_rows, max_len, capturing)
    _GEOMETRY[(q.device.type, q.device.index)] = (pages, page, max_len)
    device_row_plan(meta.context_lens, meta.slot_mapping, rows,
                    buffers.klen, buffers.valid, buffers.cuk)
    cuq = meta.cu_seqlens_q[:rows + 1]
    if cuq.dtype != torch.int32:
        raise AlignedGraphError("graph aligned decode requires int32 cu_seqlens_q")
    if PAGED and pack is None:
        output = _paged_decode(impl, q, key_cache, value_cache, meta.block_tables, buffers, rows, page,
                               max_len, cuq, varlen, capturing)
    else:
        (pack or _launch_pack)(key_cache, value_cache, meta.block_tables, buffers, rows, page)
        output = varlen(q, buffers.k, buffers.v,
                        cu_seqlens_q=cuq, cu_seqlens_k=buffers.cuk[:rows + 1],
                        max_seqlen_q=1, max_seqlen_k=max_len,
                        dropout_p=0.0, softmax_scale=impl.scale, causal=True)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (rows, 32, _HEAD_DIM) or output.dtype != q.dtype or output.device != q.device:
        raise AlignedGraphError("public varlen returned an unexpected output contract")
    return output


def validate_decode_host(context_lens, slot_mapping, block_tables, positions,
                         scheduled_bs, running_bs, pages, page, max_len):
    """Per-step host contract for a replayed decode, on ATOM's numpy mirrors.

    Equivalent to the decode branch of ``alignment_main._batch`` for q_len=1:
    real rows have a legal context, a unique in-capacity write slot equal to the
    logical position of their last token, legal unique history pages and
    ``position == context_len - 1``; padded rows write nothing.
    """
    import numpy as np

    if not 0 < scheduled_bs <= running_bs:
        raise AlignedGraphError("invalid scheduled/running decode rows")
    n = np.asarray(context_lens[:scheduled_bs], dtype=np.int64)
    slots = np.asarray(slot_mapping[:running_bs], dtype=np.int64)
    table = np.asarray(block_tables[:scheduled_bs], dtype=np.int64)
    pos = np.asarray(positions[:scheduled_bs], dtype=np.int64)
    if n.min() < 1 or n.max() > max_len:
        raise AlignedGraphError("context length violates query/context capacity")
    if (slots[scheduled_bs:] != -1).any():
        raise AlignedGraphError("padded queries must not write KV")
    pad_ctx = np.asarray(context_lens[scheduled_bs:running_bs], dtype=np.int64)
    if (pad_ctx != 0).any():
        raise AlignedGraphError("padded rows must carry zero context")
    live = slots[:scheduled_bs]
    if (live < 0).any() or (live >= pages * page).any() or len(np.unique(live)) != len(live):
        raise AlignedGraphError("invalid, duplicate or out-of-capacity live KV write slot")
    if not np.array_equal(pos, n - 1):
        raise AlignedGraphError("positions do not describe each request's causal suffix")
    counts = -(-n // page)
    if counts.max() > table.shape[1]:
        raise AlignedGraphError("block table does not cover context")
    cols = np.arange(table.shape[1])[None, :]
    used = cols < counts[:, None]
    ids = np.where(used, table, -1)
    if ((ids[used] < 0) | (ids[used] >= pages)).any():
        raise AlignedGraphError("illegal physical page in logical history")
    for row in range(scheduled_bs):
        row_ids = ids[row, :counts[row]]
        if len(np.unique(row_ids)) != len(row_ids):
            raise AlignedGraphError("illegal physical page in logical history")
    last = n - 1
    expected = table[np.arange(scheduled_bs), last // page] * page + last % page
    if not np.array_equal(live, expected):
        raise AlignedGraphError("write slots do not match each request's causal suffix")


def _host_check(runner, context, mode):
    if getattr(mode, "max_seqlen_q", 1) != 1:
        raise AlignedGraphError("graph aligned decode replay requires q_len == 1")
    if len(_GEOMETRY) != 1:
        raise AlignedGraphError("graph aligned replay needs exactly one captured KV geometry")
    pages, page, max_len = next(iter(_GEOMETRY.values()))
    var = runner.forward_vars
    try:
        arrays = [var[name].np for name in ("context_lens", "slot_mapping", "block_tables", "positions")]
    except (KeyError, AttributeError) as exc:
        raise AlignedGraphError("ATOM forward_vars host mirrors unavailable") from exc
    validate_decode_host(*arrays, context.scheduled_bs, mode.running_bs, pages, page, max_len)


def install_runner(module) -> bool:
    """Wrap ``ModelRunner.run_model`` with the per-replay host contract."""
    if not ENABLED:
        return False
    if os.environ.get("ATOM_DECODE_PROFILE", "0").strip().lower() in ("1", "true", "yes", "on"):
        raise AlignedGraphError("ATOM_DECODE_PROFILE syncs per step and cannot run with C4")
    if os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") != "1":
        raise AlignedGraphError(f"{SWITCH}=1 requires LUMENRL_ATOM_OP_ALIGNMENT=1")
    cls = getattr(module, "ModelRunner", None)
    original = getattr(cls, "run_model", None) if cls is not None else None
    if original is None:
        raise AlignedGraphError("ATOM ModelRunner.run_model unavailable")
    if getattr(original, "_lumen_aligned_graph", False):
        return True
    from atom.utils.forward_context import get_forward_context

    @functools.wraps(original)
    def run_model(self, *args, **kwargs):
        if HOSTCHECK:
            context = get_forward_context().context
            mode = context.forward_mode
            if mode is not None and mode.use_cudagraph and not context.is_prefill:
                _host_check(self, context, mode)
        return original(self, *args, **kwargs)

    run_model._lumen_aligned_graph = True
    cls.run_model = run_model
    return True
