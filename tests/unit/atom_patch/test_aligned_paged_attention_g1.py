"""G1: the paged-KV fork equals pack + pinned aiter varlen bit for bit on decode inputs.

220 cases with rows 1..64 and KV lengths 1..21504 (page and tile boundaries, the
21504 maximum, random lengths) and padded tail rows, whose output must be zero.
Needs gfx950, aiter and the canonical aligned-Attention config
(LUMENRL_ATOM_ATTENTION_CONFIG / LUMENRL_ATOM_ATTENTION_CONFIG_SHA256).
"""

from __future__ import annotations

import math
import os
import random
from types import SimpleNamespace

import pytest

PAGE, MAX_LEN, MAX_ROWS = 16, 21504, 64
WIDTH = MAX_LEN // PAGE
PAGES = MAX_ROWS * WIDTH + 64
BOUNDS = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4096, 8191, 8192, 8193,
          16383, 16384, 21440, 21503, 21504]
NUM_CASES = 220


def _cases():
    """Row counts, KV lengths and padding, drawn exactly as in the validated G1 run."""
    rng = random.Random(20261125)
    cases = []
    for case in range(NUM_CASES):
        rows = rng.choice([1, 2, 3, 8, 16, 31, 32, 33, 48, 63, 64]) if case % 3 else rng.randint(1, 64)
        lens = []
        for _ in range(rows):
            mode = rng.random()
            if mode < 0.3:
                lens.append(rng.choice(BOUNDS))
            elif mode < 0.5:
                lens.append(rng.randint(1, 200))
            else:
                lens.append(rng.randint(1, MAX_LEN))
        pad = min(rng.choice([0, 0, 1, 2, 5]), rows - 1)
        for r in range(rows - pad, rows):
            lens[r] = 0
        cases.append((case, rows, lens, pad))
    return cases


CASES = _cases()


def test_case_coverage():
    assert len(CASES) == NUM_CASES
    assert min(c[1] for c in CASES) == 1
    assert max(c[1] for c in CASES) == MAX_ROWS
    assert sum(1 for c in CASES if c[3]) == 133
    assert set(BOUNDS) <= {n for c in CASES for n in c[2]}


@pytest.fixture(scope="module")
def env():
    torch = pytest.importorskip("torch")
    pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("requires gfx950 (MI350X/MI355X)")
    if not (os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG") and os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG_SHA256")):
        pytest.skip("requires the canonical aligned-Attention config")
    from lumenrl.engine.inference.atom_patch import aligned_graph as ag
    from lumenrl.engine.inference.atom_patch import aligned_paged_attention as pa
    from lumenrl.engine.inference.atom_patch import alignment_main as am

    device = torch.device("cuda", 0)
    am.initialize_aligned_attention(device)
    gen = torch.Generator(device=device).manual_seed(3)
    kc = torch.randn((PAGES, 8, 16, PAGE, 8), generator=gen, device=device).to(torch.bfloat16)
    vc = torch.randn((PAGES, 8, PAGE // 8, 128, 8), generator=gen, device=device).to(torch.bfloat16)
    yield SimpleNamespace(
        torch=torch, ag=ag, pa=pa, am=am, device=device, kc=kc, vc=vc,
        runtime=ag._paged_runtime(am._pinned_varlen, device),
        buffers=ag._buffers(device, MAX_ROWS, MAX_LEN, False),
        scale=1.0 / math.sqrt(128),
    )
    ag._BUFFERS.clear()


def _meta(env, lens, table):
    torch, dev = env.torch, env.device
    rows = len(lens)
    ctx = torch.tensor(lens, dtype=torch.int32, device=dev)
    last = (ctx - 1).clamp(min=0).long()
    slot = table[torch.arange(rows, device=dev), last // PAGE].long() * PAGE + last % PAGE
    slot = torch.where(ctx > 0, slot, torch.full_like(slot, -1)).to(torch.int32)
    return SimpleNamespace(context_lens=ctx, slot_mapping=slot, block_tables=table,
                           cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device=dev))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("case,rows,lens,pad", CASES, ids=[f"case{c[0]}-rows{c[1]}-pad{c[3]}" for c in CASES])
def test_paged_fork_bitwise_equals_pack_and_pinned(env, case, rows, lens, pad):
    torch, ag, buffers = env.torch, env.ag, env.buffers
    gen = torch.Generator(device=env.device).manual_seed(case)
    table = torch.randperm(PAGES, generator=gen, device=env.device)[: rows * WIDTH].to(torch.int32).view(rows, WIDTH)
    meta = _meta(env, lens, table.contiguous())
    q = torch.randn((rows, 32, 128), generator=gen, device=env.device).to(torch.bfloat16)
    ag.device_row_plan(meta.context_lens, meta.slot_mapping, rows, buffers.klen, buffers.valid, buffers.cuk)
    cuq, cuk = meta.cu_seqlens_q, buffers.cuk[:rows + 1]

    ag._launch_pack(env.kc, env.vc, meta.block_tables, buffers, rows, PAGE)
    ref = env.am._pinned_varlen(q, buffers.k, buffers.v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, max_seqlen_q=1,
                                max_seqlen_k=MAX_LEN, dropout_p=0.0, softmax_scale=env.scale, causal=True)
    out = env.pa.paged_varlen(q, env.kc, env.vc, meta.block_tables, buffers.valid, cuq, cuk, MAX_LEN, env.scale,
                              *env.runtime)
    torch.cuda.synchronize()

    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if pad:
        assert bool((out[rows - pad:] == 0).all())
