#!/usr/bin/env python3
"""Assert the flex_attention path computes what the dense mask path computes.

Phase B of the first 8192 smoke run logged 8 recovered allocator OOMs trying to
reserve 9.705 GiB. That is exactly the score matrix an explicit boolean
attn_mask forces SDPA to materialise:

    64 heads x 3584 draft (anchor_num 512 x block 7) x 11749 kv x 4 B = 10.05 GiB

DSparkMLAAttention now prefers a flex_attention BlockMask, which evaluates
visibility per block and never forms it. This checks that swapping the kernel
did not quietly change the mask, since the two encode the same rules in very
different shapes (a dense [B,1,Lq,Lkv] tensor versus a mask_mod closure).

Run inside the training image (needs a GPU):

    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
        --group-add render --security-opt seccomp=unconfined --ipc host \
        -v /home/jimguo12/Lumen-RL:/workspace -w /workspace \
        kimi_k3_dspark_atom:latest \
        python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/verify_flex_attention.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from lumenrl.models.dspark import _HAS_FLEX, DSparkMLAAttention  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    # anchor_num and block_size are the shipped recipe; ctx is the padded batch
    # width the 8192 smoke run actually reached.
    ap.add_argument("--anchor-num", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--ctx-len", type=int, default=8165)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    if not _HAS_FLEX:
        print("FAIL: torch has no flex_attention; the dense path is the only one")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(0)

    block_size = args.block_size
    num_anchors = args.anchor_num
    draft_len = num_anchors * block_size
    ctx_len = args.ctx_len
    B, H = 1, 64

    attn = DSparkMLAAttention(
        hidden_size=7168,
        num_heads=H,
        num_kv_heads=H,
        rope_scaling={
            "rope_type": "yarn", "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1,
            "mscale": 1.0, "mscale_all_dim": 1.0,
        },
    ).to(device=device, dtype=torch.bfloat16)
    attn.block_size = block_size

    draft_hidden = torch.randn(B, draft_len, 7168, device=device, dtype=torch.bfloat16)
    context_hidden = torch.randn(B, ctx_len, 7168, device=device, dtype=torch.bfloat16)

    # Anchors strictly inside the context and ordered, matching _sample_anchors.
    anchors = torch.sort(
        torch.randperm(ctx_len - 1, device=device)[:num_anchors] + 1
    ).values.unsqueeze(0)

    # _sample_anchors keeps the first valid_count slots and pads the rest with
    # anchor=0 / keep=False, and with last_turn_loss_only the supervised span is
    # usually shorter than 512 tokens, so a partly-False keep is the common case
    # rather than the corner one. Reproduce that here: an all-True keep is what
    # let an earlier version of this check pass while training still fell back
    # to the dense path on every step.
    keep = torch.ones(B, num_anchors, device=device, dtype=torch.bool)
    valid_count = int(num_anchors * 0.6)
    keep[:, valid_count:] = False
    anchors = torch.where(keep, anchors, torch.zeros_like(anchors))

    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    draft_position_ids = (anchors.unsqueeze(-1) + offsets).reshape(B, -1)
    context_position_ids = torch.arange(ctx_len, device=device).unsqueeze(0).expand(B, -1)

    # A DSparkModel instance is not needed: both mask builders are methods that
    # only read self.block_size, so borrow them onto the attention module.
    from lumenrl.models.dspark import DSparkModel
    attn._build_dual_source_mask = DSparkModel._build_dual_source_mask.__get__(attn)
    attn._build_dual_source_block_mask = (
        DSparkModel._build_dual_source_block_mask.__get__(attn)
    )

    dense_mask = attn._build_dual_source_mask(anchors, keep, ctx_len)
    block_mask, keep_rows = attn._build_dual_source_block_mask(anchors, keep, ctx_len)
    if block_mask is None:
        print("FAIL: block mask builder declined this batch; nothing to compare")
        return 1

    common = dict(
        draft_hidden=draft_hidden,
        context_hidden=context_hidden,
        draft_position_ids=draft_position_ids,
        context_position_ids=context_position_ids,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        ref = attn(attn_mask=dense_mask, **common)
    torch.cuda.synchronize()
    t_dense = time.time() - t0
    mem_dense = torch.cuda.max_memory_allocated() / 2**30

    with torch.no_grad():  # warm up the compile before timing
        attn(block_mask=block_mask, keep_rows=keep_rows, **common)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = attn(block_mask=block_mask, keep_rows=keep_rows, **common)
    torch.cuda.synchronize()
    t_flex = time.time() - t0
    mem_flex = torch.cuda.max_memory_allocated() / 2**30

    diff = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)

    dropped = ~keep_rows.view(B, draft_len)
    ref_dropped_max = ref.float().abs()[dropped].max()
    out_dropped_max = out.float().abs()[dropped].max()

    print(f"shape: draft={draft_len} ctx={ctx_len} heads={H}")
    print(f"  blocks kept {int(keep.sum())}/{num_anchors}, "
          f"dropped rows |ref| max {ref_dropped_max:.2e}, "
          f"|flex| max {out_dropped_max:.2e}")
    print(f"  dense mask + SDPA   {t_dense * 1e3:9.1f} ms   peak {mem_dense:6.2f} GiB")
    print(f"  BlockMask + flex    {t_flex * 1e3:9.1f} ms   peak {mem_flex:6.2f} GiB")
    print(f"  rel L2              {diff:.3e}")
    print(f"  speedup {t_dense / max(t_flex, 1e-9):.1f}x   "
          f"memory {mem_dense / max(mem_flex, 1e-9):.1f}x smaller")

    # Batches are padded to their own longest sequence, so KV_LEN moves nearly
    # every step. Under dynamic=False that recompiles each time, exhausts the
    # dynamo cache and then silently degrades to eager flex — slower and larger
    # than the SDPA being replaced. Automatic mode is allowed a couple of
    # recompiles to go from specialised to dynamic; what it may not do is pay one
    # per shape, so walk several widths and require the tail to have settled.
    shape_times = []
    for i, extra in enumerate((0, -37, -128, -211, -304, -419)):
        c = ctx_len + extra
        a = torch.clamp(anchors, max=c - 1)
        bm, kr = attn._build_dual_source_block_mask(a, keep, c)
        ctx_i = context_hidden[:, :c]
        pos_i = torch.arange(c, device=device).unsqueeze(0).expand(B, -1)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            attn(draft_hidden=draft_hidden, context_hidden=ctx_i,
                 draft_position_ids=draft_position_ids,
                 context_position_ids=pos_i, block_mask=bm, keep_rows=kr)
        torch.cuda.synchronize()
        shape_times.append((c, time.time() - t0))

    worst_settled = max(t for _, t in shape_times[-3:])
    print("  varying KV_LEN: " + ", ".join(
        f"{c}:{t * 1e3:.0f}ms" for c, t in shape_times))

    if not torch.isfinite(out).all():
        print("\nFAIL: flex output has non-finite values; an empty row reached "
              "the softmax and a NaN here would survive the loss mask")
        return 1
    if max(float(ref_dropped_max), float(out_dropped_max)) != 0.0:
        print("\nFAIL: rows of dropped blocks are not exactly zero")
        return 1
    if diff >= args.tol:
        print(f"\nFAIL: the two masks do not agree (rel L2 {diff:.3e} >= {args.tol})")
        return 1
    if mem_flex >= mem_dense:
        print("\nFAIL: flex used no less memory; it is probably running uncompiled")
        return 1
    # A recompile costs seconds; a settled dynamic kernel costs milliseconds, so
    # any threshold in between separates them. 20x the single-shape time is well
    # clear of both.
    if worst_settled > 20 * t_flex:
        print(f"\nFAIL: KV_LEN never settles — still {worst_settled * 1e3:.0f} ms "
              f"against {t_flex * 1e3:.1f} ms after several widths. flex is "
              f"recompiling per shape and will degrade to eager once the "
              f"dynamo cache fills")
        return 1
    print("\nPASS: flex_attention matches the dense mask path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
