#!/usr/bin/env python3
"""Patch op_harness.py to add missing operator aliases."""
# This script patches all 150 op_harness.py files

import re
from pathlib import Path

ARTIFACT_ROOT = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")

# Lines to add after existing _reg calls (before _FALLBACK)
EXTRA_REGS = '''
# ── Missing operator aliases ────────────────────────────────────────────────
_reg(["mxfp4_gemm", "preshuffle_gemm", "gemm_a16w16"], OpHarness(
    shapes={
        "a": (_M, _K), "b": (_N, _K), "c": (_M, _N),
        "x": (_M, _K), "w": (_N, _K), "y": (_M, _N),
        "input": (_M, _K), "weight": (_N, _K), "output": (_M, _N),
        "bias": (_N,),
        "a_scale": (_M,), "b_scale": (_N,),
        "x_scale": (_M,), "w_scale": (_N,),
        "scale_a": (_M,), "scale_b": (_N,),
        "a_scale_block": (_M, _K // 128), "b_scale_block": (_N, _K // 128),
        "m": _M, "n": _N, "k": _K,
        "stride_am": _K, "stride_bk": _N, "stride_cm": _N,
    },
    dtypes={
        "a": torch.bfloat16, "b": torch.bfloat16,
        "x": torch.bfloat16, "w": torch.bfloat16,
        "input": torch.bfloat16, "weight": torch.bfloat16,
    },
))

_reg(["causal_conv1d"], OpHarness(
    shapes={
        "x": (_B, 256, _S), "input": (_B, 256, _S),
        "weight": (256, 1, 4), "conv_weight": (256, 1, 4),
        "bias": (256,),
        "state": (_B, 256, 4), "conv_state": (_B, 256, 4),
        "a": (_B, 256, _S), "b": (256, 1, 4),
        "d_model": 256, "d_conv": 4,
    },
    dtypes={
        "x": torch.bfloat16, "input": torch.bfloat16,
        "weight": torch.bfloat16, "conv_weight": torch.bfloat16,
        "state": torch.bfloat16, "conv_state": torch.bfloat16,
    },
))

_reg(["moe_routing", "moe_grouped_gemm", "moe_reduce", "dispatch_combine"], OpHarness(
    shapes={
        "hidden_states": (_M, _K), "input": (_M, _K),
        "x": (_M, _K),
        "w1": (_EXPERTS, _N, _K), "w2": (_EXPERTS, _K, _N),
        "w3": (_EXPERTS, _N, _K),
        "gate_up_weight": (_EXPERTS, _N * 2, _K),
        "down_weight": (_EXPERTS, _K, _N),
        "topk_weights": (_M, 2), "topk_ids": (_M, 2),
        "expert_ids": (_M * 2,), "token_ids": (_M * 2,),
        "sorted_token_ids": (_M * 2,), "expert_weights": (_M, 2),
        "num_tokens": _M, "num_experts": _EXPERTS, "top_k": 2,
        "hidden_size": _K, "intermediate_size": _N,
    },
    dtypes={
        "hidden_states": torch.bfloat16, "input": torch.bfloat16,
        "x": torch.bfloat16,
        "w1": torch.bfloat16, "w2": torch.bfloat16, "w3": torch.bfloat16,
        "gate_up_weight": torch.bfloat16, "down_weight": torch.bfloat16,
        "topk_weights": torch.float32,
        "topk_ids": torch.int32, "expert_ids": torch.int32,
        "token_ids": torch.int32, "sorted_token_ids": torch.int32,
        "expert_weights": torch.float32,
    },
))

_reg(["chunk_gated_delta"], OpHarness(
    shapes={
        "q": (_B, _H, _S, _D), "k": (_B, _H, _S, _D),
        "v": (_B, _H, _S, _D), "beta": (_B, _H, _S),
        "g": (_B, _H, _S), "gate": (_B, _H, _S),
        "output": (_B, _H, _S, _D),
        "h0": (_B, _H, _D, _D),
        "seq_len": _S, "num_heads": _H, "head_dim": _D,
    },
    dtypes={
        "q": torch.bfloat16, "k": torch.bfloat16,
        "v": torch.bfloat16, "output": torch.bfloat16,
        "h0": torch.bfloat16, "beta": torch.bfloat16,
        "g": torch.bfloat16, "gate": torch.bfloat16,
    },
))
'''

count = 0
for task_dir in sorted(ARTIFACT_ROOT.iterdir()):
    harness_path = task_dir / "initial" / "scripts" / "op_harness.py"
    if not harness_path.exists():
        continue

    content = harness_path.read_text()

    # Skip if already patched
    if "causal_conv1d" in content:
        continue

    # Insert before _FALLBACK definition
    marker = "# ── Generic fallback"
    if marker in content:
        content = content.replace(marker, EXTRA_REGS + "\n" + marker)
        harness_path.write_text(content)
        count += 1

print(f"Patched {count} op_harness.py files")
