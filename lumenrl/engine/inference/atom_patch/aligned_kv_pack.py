"""Default-off fused KV pack for aligned decode attention (candidate C1).

Why
---
The aligned decode path re-packs live KV pages on every attention layer:
``aligned_attention._gather_live_kv`` runs a Python loop per request with
~2*28 host->device ``torch.tensor`` calls plus ~150 small GPU ops per layer.
The KV layout metadata (page table, context lengths) is *identical for all
layers within one decode step*, so the plan can be built once per step and the
per-layer work can shrink to one fused copy kernel.

What this module does
---------------------
* ``fused_gather_live_kv`` returns the exact same packed BF16 K/V bytes and
  ``cuk`` list as the legacy eager gather, with the same 5D SHUFFLE source
  layout and the same (T, 8, 128) contiguous destination layout.
* The per-token (physical page, in-page position) plan is cached per decode
  step, keyed by ``(lengths, page, block-table bytes)``.
* Any selected C1 failure raises through the aligned adapter; no candidate
  step silently falls back to legacy gather.

Enable with ``ATOM_ALIGNED_KV_PACK=1`` (default 0).  It must not change
numerics: the copy kernel reads exactly the bytes the eager gather reads.
"""

from __future__ import annotations

import os

import torch

_TRUE = ("1", "true", "yes", "on")
ENABLED = os.environ.get("ATOM_ALIGNED_KV_PACK", "0").strip().lower() in _TRUE

_HEADS = 8
_HEAD_DIM = 128
_DIM_PACK = 16
_PACK = 8

# Last decoded plan: (key, page_of_token, pos_of_token).  One entry is enough:
# within a decode step every layer hits; the next step builds a new plan once.
_PLAN_KEY = None
_PLAN = None


def _plan_key(table, lengths, page):
    table_bytes = (
        table.detach().cpu().contiguous().numpy().tobytes()
        if hasattr(table, "detach")
        else bytes(table)
    )
    return (tuple(int(x) for x in lengths), int(page), table_bytes)


def _build_plan(table, lengths, page, device):
    """Per-token source mapping, built on GPU from one small H2D table copy."""
    nseq = len(lengths)
    table_gpu = table.to(device=device, dtype=torch.int64)
    pages = []
    positions = []
    for row, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            continue
        n_pages = (length + page - 1) // page
        ids = table_gpu[row, :n_pages]
        pages.append(ids.repeat_interleave(page)[:length])
        positions.append(torch.arange(length, device=device, dtype=torch.int64) % page)
    if not pages:
        return (
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.int64),
        )
    return torch.cat(pages), torch.cat(positions)


def _get_plan(table, lengths, page, device):
    global _PLAN_KEY, _PLAN
    key = _plan_key(table, lengths, page)
    if _PLAN_KEY == key and _PLAN is not None:
        return _PLAN
    _PLAN = _build_plan(table, lengths, page, device)
    _PLAN_KEY = key
    return _PLAN


def _pack_kernel():
    # Imported lazily so importing atom.model_ops stays CPU-safe.
    import triton
    import triton.language as tl

    @triton.jit
    def _pack_one_token(
        k_ptr,
        v_ptr,
        out_k_ptr,
        out_v_ptr,
        page_of_token_ptr,
        pos_of_token_ptr,
        page: tl.constexpr,
        page_pack: tl.constexpr,
    ):
        idx = tl.program_id(0)
        phy = tl.load(page_of_token_ptr + idx)
        pos = tl.load(pos_of_token_ptr + idx)
        heads = tl.arange(0, 8)[:, None]
        dims = tl.arange(0, 128)[None, :]
        # K source strides for (pages, 8, 16, page, 8):
        #   page: 8*16*page*8, head: 16*page*8, dim_pack: page*8, pos: 8, pack: 1
        k_src = (
            phy * (8 * 16 * page * 8)
            + heads * (16 * page * 8)
            + (dims // 8) * (page * 8)
            + pos * 8
            + (dims % 8)
        )
        # V source strides for (pages, 8, page//8, 128, 8):
        #   page: 8*page_pack*128*8, head: page_pack*128*8,
        #   pos//8: 128*8, dim: 8, pos%8: 1
        v_src = (
            phy * (8 * page_pack * 128 * 8)
            + heads * (page_pack * 128 * 8)
            + (pos // 8) * (128 * 8)
            + dims * 8
            + (pos % 8)
        )
        dst = idx * (8 * 128) + heads * 128 + dims
        tl.store(out_k_ptr + dst, tl.load(k_ptr + k_src))
        tl.store(out_v_ptr + dst, tl.load(v_ptr + v_src))

    return _pack_one_token


_KERNEL = None


def fused_gather_live_kv(key_cache, value_cache, table, lengths, page, device):
    """Drop-in replacement for ``aligned_attention._gather_live_kv``.

    Returns ``(packed_k, packed_v, cuk)`` with identical bytes to the eager
    gather; raises on any unsupported input so the caller can fall back.
    """
    global _KERNEL
    if key_cache.ndim != 5 or value_cache.ndim != 5:
        raise ValueError("fused KV pack requires native 5D SHUFFLE caches")
    if key_cache.shape[1:] != (_HEADS, _DIM_PACK, page, _PACK):
        raise ValueError("unexpected K cache geometry")
    page_pack = page // _PACK
    if value_cache.shape != (
        key_cache.shape[0],
        _HEADS,
        page_pack,
        _HEAD_DIM,
        _PACK,
    ):
        raise ValueError("unexpected V cache geometry")
    lengths = [int(x) for x in lengths]
    total = sum(lengths)
    pages, positions = _get_plan(table, lengths, page, device)
    out_k = torch.empty((total, _HEADS, _HEAD_DIM), dtype=key_cache.dtype, device=device)
    out_v = torch.empty_like(out_k)
    if total:
        if _KERNEL is None:
            _KERNEL = _pack_kernel()
        _KERNEL[(total,)](
            key_cache,
            value_cache,
            out_k,
            out_v,
            pages,
            positions,
            page=page,
            page_pack=page_pack,
            num_warps=4,
        )
    cuk = [0]
    for length in lengths:
        cuk.append(cuk[-1] + length)
    return out_k, out_v, cuk


def reset_plan_cache():
    """Test helper: drop the per-step plan."""
    global _PLAN_KEY, _PLAN
    _PLAN_KEY = None
    _PLAN = None
