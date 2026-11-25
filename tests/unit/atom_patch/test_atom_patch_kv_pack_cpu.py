###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU proof that Lumen-RL's fused KV-pack offsets equal the eager gather.

The diagnostic fused pack is selected by the Lumen-RL ATOM adapter and must
not be represented as an ATOM source change or a GEMM CSV retuning.  Its Triton copy
kernel reads K/V with per-token offset arithmetic; this test rebuilds both the
legacy byte semantics and the fused offsets in numpy, for several page sizes,
request lengths and page layouts, and requires bit-identical packed K/V.

No torch, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

HEADS = 8
DIM_PACK = 16
PACK = 8
HEAD_DIM = 128


def legacy_gather(key_cache, value_cache, table, lengths, page):
    """Accepted path: select live raw pages -> permute/reshape -> per-request index."""
    live = sorted(
        {
            int(p)
            for row, length in zip(table, lengths)
            for p in row[: (length + page - 1) // page]
        }
    )
    remap = {value: i for i, value in enumerate(live)}
    raw_k = key_cache[live]
    raw_v = value_cache[live]
    keys = raw_k.transpose(0, 3, 1, 2, 4).reshape(len(live), page, HEADS, HEAD_DIM)
    values = raw_v.transpose(0, 2, 4, 1, 3).reshape(len(live), page, HEADS, HEAD_DIM)
    packed_k, packed_v = [], []
    for row, length in zip(table, lengths):
        index = [remap[int(p)] for p in row[: (length + page - 1) // page]]
        packed_k.append(keys[index].reshape(-1, HEADS, HEAD_DIM)[:length])
        packed_v.append(values[index].reshape(-1, HEADS, HEAD_DIM)[:length])
    return np.concatenate(packed_k), np.concatenate(packed_v)


def fused_gather(key_cache, value_cache, table, lengths, page):
    """Fused path: per-token source offsets used by the Triton copy kernel."""
    page_pack = page // PACK
    total = int(sum(lengths))
    out_k = np.empty((total, HEADS, HEAD_DIM), dtype=key_cache.dtype)
    out_v = np.empty_like(out_k)
    idx = 0
    for row, length in zip(table, lengths):
        for token in range(int(length)):
            phy = int(row[token // page])
            pos = token % page
            for head in range(HEADS):
                for dim in range(HEAD_DIM):
                    # K cache layout: (pages, 8, 16, page, 8)
                    k_src = (
                        phy * (8 * 16 * page * 8)
                        + head * (16 * page * 8)
                        + (dim // 8) * (page * 8)
                        + pos * 8
                        + (dim % 8)
                    )
                    # V cache layout: (pages, 8, page//8, 128, 8)
                    v_src = (
                        phy * (8 * page_pack * 128 * 8)
                        + head * (page_pack * 128 * 8)
                        + (pos // 8) * (128 * 8)
                        + dim * 8
                        + (pos % 8)
                    )
                    out_k[idx, head, dim] = key_cache.reshape(-1)[k_src]
                    out_v[idx, head, dim] = value_cache.reshape(-1)[v_src]
            idx += 1
    return out_k, out_v


def make_case(npages, page, rows, lengths, seed):
    rng = np.random.default_rng(seed)
    key_cache = rng.integers(
        0, 4096, size=(npages, HEADS, DIM_PACK, page, PACK), dtype=np.uint16
    ).astype(np.uint16)
    value_cache = rng.integers(
        0, 4096, size=(npages, HEADS, page // PACK, HEAD_DIM, PACK), dtype=np.uint16
    ).astype(np.uint16)
    return key_cache, value_cache, rows, lengths, page


CASES = [
    # (n_pages, page, block table rows, lengths)
    (8, 16, [[7, 1, 4, 0], [2, 6, 3, 5]], [40, 42]),
    (16, 16, [[15, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], [64, 17, 1]),
    (8, 32, [[7, 2], [1, 6]], [64, 33]),
    (6, 64, [[5], [4], [3]], [64, 1, 32]),
    (4, 128, [[3, 1]], [129]),
]


@pytest.mark.parametrize(
    "npages,page,rows,lengths",
    CASES,
    ids=[f"page{case[1]}_len{case[3]}" for case in CASES],
)
def test_fused_offset_mapping_is_bit_identical(npages, page, rows, lengths):
    key_cache, value_cache, rows, lengths, page = make_case(
        npages, page, rows, lengths, seed=100 + CASES.index((npages, page, rows, lengths))
    )
    table = np.asarray(rows, dtype=np.int64)
    need = max((length + page - 1) // page for length in lengths)
    table = table[:, :need]

    want_k, want_v = legacy_gather(key_cache, value_cache, table, lengths, page)
    got_k, got_v = fused_gather(key_cache, value_cache, table, lengths, page)

    np.testing.assert_array_equal(got_k, want_k)
    np.testing.assert_array_equal(got_v, want_v)
