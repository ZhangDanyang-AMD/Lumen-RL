#!/usr/bin/env python3
"""Attribute the cost of one DSpark micro-batch forward+backward.

Written because an estimate was wrong. Smoke run 1 blamed the 92-134 s optimizer
step on masked SDPA, on the grounds that the benchmarked kernel cost times the
number of calls came out near the step time. Replacing that kernel with
flex_attention removed the allocator OOMs and changed the step time by nothing,
so the arithmetic had been fitting a number rather than explaining it.

This runs the real DSparkModel at the real per-rank shape (train_micro_batch_size
1, anchor_num 512, block_size 7, 8192 window) and reports measured CUDA time per
op, so the next optimisation target is chosen from data.

Run inside the training image (needs one GPU):

    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
        --group-add render --security-opt seccomp=unconfined --ipc host \
        -v /home/jimguo12/Lumen-RL:/workspace -w /workspace \
        kimi_k3_dspark_atom:latest \
        python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/profile_train_step.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, ".")

from lumenrl.models.dspark import DSparkModel  # noqa: E402

VOCAB = 163840
HIDDEN = 7168


def build(args, device):
    model = DSparkModel(
        hidden_dim=HIDDEN,
        vocab_size=VOCAB,
        num_heads=64,
        num_kv_heads=64,
        num_layers=5,
        num_target_layers=5,
        block_size=args.block_size,
        ffn_dim=14336,
        anchor_num=args.anchor_num,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        rope_theta=50000.0,
        rope_scaling={
            "rope_type": "yarn", "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1,
            "mscale": 1.0, "mscale_all_dim": 1.0,
        },
    ).to(device=device, dtype=torch.bfloat16)
    return model


def make_batch(args, device):
    B, T = 1, args.seq_len
    g = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (B, T), generator=g).to(device)
    # last_turn_loss_only: one assistant turn near the end of the sequence.
    loss_mask = torch.zeros(B, T, device=device)
    loss_mask[:, -args.supervised:] = 1.0
    aux = torch.randn(B, T, HIDDEN * 5, device=device, dtype=torch.bfloat16)
    target_hidden = torch.randn(B, T, HIDDEN, device=device, dtype=torch.bfloat16)
    lm_head = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16) * 0.02
    embed_weight = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16) * 0.02
    token_embeds = embed_weight[input_ids.reshape(-1)].view(B, T, HIDDEN)
    return (input_ids, loss_mask, aux, target_hidden, lm_head, embed_weight,
            token_embeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-num", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--seq-len", type=int, default=8165)
    # Median assistant turn; anchors can only be sampled inside it.
    ap.add_argument("--supervised", type=int, default=400)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rows", type=int, default=18)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = build(args, device)
    (input_ids, loss_mask, aux, target_hidden, lm_head, embed_weight,
     token_embeds) = make_batch(args, device)

    def one_step():
        out = model(
            input_ids=input_ids,
            token_embeds=token_embeds,
            aux_hidden_states=aux,
            teacher_lm_head_weight=lm_head,
            embed_weight=embed_weight,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden,
        )
        loss = out["total_loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        return loss

    one_step()  # warm up allocator and any compile
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iters):
        one_step()
    torch.cuda.synchronize()
    per_iter = (time.time() - t0) / args.iters
    peak = torch.cuda.max_memory_allocated() / 2**30

    print(f"micro-batch fwd+bwd: {per_iter * 1e3:.0f} ms   peak {peak:.1f} GiB")
    print(f"  x16 micro-batches = {per_iter * 16:.1f} s per optimizer step "
          f"(measured in training: ~110 s)")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        one_step()
        torch.cuda.synchronize()

    print()
    print(prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=args.rows,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
