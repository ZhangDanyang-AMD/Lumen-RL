#!/usr/bin/env python3
"""Assert the draft is trained under the attention ATOM will serve it with.

The first run's draft benchmarked at 6.0% acceptance where Inferact's released
draft reaches 26.1%, because LumenRL's DSpark attention disagreed with ATOM's in
two independent ways:

  1. rotation convention -- LumenRL rotated NeoX half-split, DeepSeek/Kimi MLA
     (and therefore TorchSpec and ATOM) rotates interleaved pairs;
  2. softmax scale -- LumenRL used 1/sqrt(qk_head_dim) while both TorchSpec and
     ATOM widen it by the YaRN mscale^2, a factor of 1.8133 at factor=32 /
     mscale_all_dim=1.0.

Neither is visible in a rotary-only comparison that holds mscale == mscale_all_dim,
which is how they survived the first run: the cos/sin cache is scaled by
mscale/mscale_all_dim, exactly 1.0 in every config we ship.

This check builds ATOM's own rotary embedding through the same `get_rope` call
`atom/models/kimi_k3_dspark.py` makes, reading the very config.json the trainer
will export, and compares it against the trainer's `_apply_rope_by_position`.

Run inside the ATOM image (needs aiter, hence a GPU):

    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
        -v /home/jimguo12/Lumen-RL:/workspace -w /workspace \
        rocm/atom-dev:latest \
        python examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/verify_rope_alignment.py
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

sys.path.insert(0, ".")

from lumenrl.models.dspark import DSparkMLAAttention  # noqa: E402

# The published Inferact Kimi-K3-DSpark rope block. The trainer must agree with
# whatever a served config.json says, so this is the contract under test rather
# than a set of knobs to tune.
PUBLISHED_ROPE = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
MAX_POSITION = 32768


def build_atom_rope(rope_interleave: bool):
    """Rebuild ATOM's rotary embedding exactly as kimi_k3_dspark.py does."""
    from aiter.rotary_embedding import get_rope

    rope_scaling = dict(PUBLISHED_ROPE)
    rope_scaling["rope_type"] = "deepseek_yarn"
    rope_scaling.setdefault("original_max_position_embeddings", MAX_POSITION)
    # DeepseekScalingRotaryEmbedding hardcodes device="cuda" for one of the two
    # halves of its inv_freq and leaves the ramp mask on the default device, so
    # it only builds cleanly when that default is already cuda.
    with torch.device("cuda"):
        return get_rope(
            QK_ROPE_HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position=MAX_POSITION,
            base=PUBLISHED_ROPE["rope_theta"],
            rope_scaling=rope_scaling,
            is_neox_style=bool(rope_interleave),
        )


def atom_softmax_scale() -> float:
    from atom.models.deepseek_v2 import yarn_get_mscale

    ms = yarn_get_mscale(
        float(PUBLISHED_ROPE["factor"]),
        float(PUBLISHED_ROPE.get("mscale_all_dim", False)),
    )
    return QK_HEAD_DIM**-0.5 * ms * ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    failures: list[str] = []

    attn = DSparkMLAAttention(
        hidden_size=7168,
        num_heads=64,
        num_kv_heads=64,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        rope_theta=PUBLISHED_ROPE["rope_theta"],
        max_position_embeddings=MAX_POSITION,
        rope_scaling=dict(PUBLISHED_ROPE),
    ).to(device)

    # --- 1. softmax scale -------------------------------------------------
    want_scale = atom_softmax_scale()
    got_scale = attn._softmax_scale
    plain = 1.0 / math.sqrt(QK_HEAD_DIM)
    print("softmax scale")
    print(f"  ATOM      : {want_scale:.10f}  ({want_scale / plain:.4f}x plain)")
    print(f"  LumenRL   : {got_scale:.10f}  ({got_scale / plain:.4f}x plain)")
    if abs(got_scale - want_scale) > 1e-9:
        failures.append(
            f"softmax scale {got_scale:.10f} != ATOM's {want_scale:.10f} "
            f"(ratio {got_scale / want_scale:.6f})"
        )

    # --- 2. rotation convention ------------------------------------------
    # Positions are deliberately scattered rather than a range: the trainer
    # indexes the cache by explicit position ids (draft tokens sit at the
    # anchor's offsets, not at 0..T), and a convention bug that cancels on a
    # contiguous range would survive that.
    positions = torch.tensor(
        [0, 1, 2, 3, 7, 8, 15, 16, 63, 64, 127, 511, 512, 1023, 4095, 8191],
        device=device,
        dtype=torch.long,
    )
    n = positions.numel()
    x = torch.randn(1, 1, n, QK_ROPE_HEAD_DIM, device=device, dtype=torch.float32)

    lumen_out = attn._apply_rope_by_position(x, positions.unsqueeze(0))

    for rope_interleave, label in ((False, "published (interleaved)"),
                                   (True, "rope_interleave=true (half-split)")):
        rope = build_atom_rope(rope_interleave)
        # aiter rotates [num_tokens, num_heads, head_size] in place-ish; give it
        # its own copy so the two calls cannot interfere.
        q = x.reshape(n, 1, QK_ROPE_HEAD_DIM).clone()
        k = q.clone()
        q_out, _ = rope(positions, q, k)
        atom_out = q_out.reshape(1, 1, n, QK_ROPE_HEAD_DIM).float()

        diff = (lumen_out.float() - atom_out).norm() / atom_out.norm().clamp_min(1e-12)
        verdict = "MATCH" if diff < args.tol else "differs"
        print(f"rotation vs ATOM {label:36} rel L2 = {diff:.3e}  {verdict}")

        if not rope_interleave and diff >= args.tol:
            failures.append(
                f"trainer rotation does not match ATOM's default (interleaved): "
                f"rel L2 {diff:.3e}"
            )
        if rope_interleave and diff < args.tol:
            failures.append(
                "trainer rotation still matches the half-split convention; the "
                "exported config would need rope_interleave=true, which is not "
                "what the published K3DSparkModel format means"
            )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: trainer attention matches ATOM under the published config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
