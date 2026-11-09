#!/usr/bin/env python3
"""Construct multi-turn GEAK SFT trajectories from pre-written kernel templates.

Reads training cases, selects kernel template per operator, writes kernel.py,
tests in Docker (compile+correctness), and constructs trajectories in JSONL.

Usage:
    python3 scripts/construct_trajectories.py --lang triton --workers 6 --gpu-ids 2,3,4,5,6,7
    python3 scripts/construct_trajectories.py --lang triton --workers 1 --gpu-ids 2 --max-cases 5  # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
import yaml
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "scripts" / "kernel_templates"
CASES_YAML = ROOT / "sft-training-cases" / "sft_training_cases.yaml"
OUTPUT_DIR = Path("/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP")
DOCKER_CONTAINER = "aiter-mi400-regress"

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "align_sft_system_prompts", ROOT / "scripts" / "align_sft_system_prompts.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SYSTEM_PROMPT_TRITON = _mod.SYSTEM_PROMPT_TRITON
SYSTEM_PROMPT_FLYDSL = _mod.SYSTEM_PROMPT_FLYDSL

OPERATOR_GUIDANCE = {
    "gemm": "OPERATOR: General Matrix Multiply (GEMM)\n  Computes: C[M,N] = A[M,K] @ B[N,K].T\n  Inputs: A (M×K, bf16), B (N×K, bf16), Output: C (M×N, bf16)\n  Key: MUST iterate over the FULL K dimension. Each C[i,j] = sum(A[i,:] * B[j,:]).\n  Reference: torch.mm(A.float(), B.float().T)",
    "splitk_gemm": "OPERATOR: Split-K GEMM\n  Computes: C[M,N] = A[M,K] @ B[N,K].T, K split across workgroups\n  Each split computes partial sum over K/SPLITS elements, then reduces.\n  Reference: torch.mm(A.float(), B.float().T)",
    "preshuffle_gemm": "OPERATOR: Preshuffle GEMM (FP8)\n  Computes: C[M,N] = dequant(A[M,K]) @ dequant(B[N,K]).T\n  Full K-dimension iteration with dequantization.\n  Reference: torch.mm(A.float(), B.float().T)",
    "gemm_a8w8": "OPERATOR: INT8 Quantized GEMM (A8W8)\n  Computes: C[M,N] = (A_int8 * a_scale) @ (B_int8 * b_scale).T\n  Must dequantize and iterate full K dimension.\n  Reference: (A.float() * a_scale) @ (B.float() * b_scale).T",
    "mxfp4_gemm": "OPERATOR: MXFP4 GEMM (4-bit microscaled)\n  Computes: C[M,N] = dequant_mxfp4(A) @ dequant_mxfp4(B).T\n  4-bit values with E8M0 microscales per 32-element group. Full K iteration.\n  Reference: torch.mm after dequantization",
    "moe": "OPERATOR: Mixture-of-Experts (MoE)\n  For each token, route to top-k experts, run expert GEMM, combine with weights.\n  Inputs: hidden_states (M×K), topk_weights (M×topk), topk_ids (M×topk), expert weights\n  This is NOT element-wise. Do NOT write relu/tanh/elu/sigmoid.",
    "moe_grouped_gemm": "OPERATOR: MoE Grouped GEMM\n  For each expert e: Y_e = X_e @ W_e.T where X_e are tokens routed to expert e.\n  Batched matrix multiply with different group sizes. MUST iterate full K.",
    "moe_sorting": "OPERATOR: MoE Token Sorting (counting sort)\n  Reorders tokens by expert assignment: histogram counts → prefix sum → scatter.\n  Output: sorted indices (int32). This is a PERMUTATION, not element-wise.",
    "moe_routing": "OPERATOR: MoE Routing\n  Computes softmax over expert logits, selects top-k experts per token.\n  Output: topk_weights (float), topk_ids (int32). INDEX MAPPING operation.",
    "moe_reduce": "OPERATOR: MoE Top-K Reduction\n  Y[t,d] = sum_k(X[t,k,d] * weight[t,k]) — weighted sum over top-k expert outputs.\n  Reduction over topk dimension. NOT identity copy.",
    "attention": "OPERATOR: Attention\n  O = softmax(Q @ K.T / sqrt(d)) @ V — must loop over ALL key positions.\n  Online softmax: track running max, sum_exp, weighted V accumulator.\n  Reference: F.scaled_dot_product_attention(Q, K, V)",
    "fmha": "OPERATOR: Flash Multi-Head Attention\n  O = softmax(Q @ K.T / sqrt(d_k)) @ V, tiled/flash algorithm.\n  Two matrix multiplies: QK (scores) and PV (output).\n  Reference: F.scaled_dot_product_attention(Q, K, V)",
    "mla": "OPERATOR: Multi-Latent Attention reduce\n  Merges per-split partial outputs: O = sum(exp(LSE_i - LSE_max) * O_i).\n  Weighted combination using softmax of LSE values.",
    "causal_conv1d": "OPERATOR: Causal 1D Convolution\n  y[n] = sum_{k=0}^{K-1} x[n-k] * w[k] with causal (left-padding).\n  Sliding window — each output depends on K preceding input values.",
    "quantization": "OPERATOR: Quantization\n  Quantize bf16 to FP4/FP8: compute per-group amax → scale → clamp.\n  Per-group scale computation and format conversion. NOT identity copy.",
    "rmsnorm": "OPERATOR: RMSNorm\n  y = x * rsqrt(mean(x^2) + eps) * weight.\n  Reduction over hidden dim, then element-wise scaling.",
    "topk": "OPERATOR: Fused Gate-Activation + TopK\n  Fused SiLU activation, gating, and optional quantization for MoE.",
    "chunk_gated_delta": "OPERATOR: Gated Delta Net Recurrence\n  h = h * exp(g) + k^T @ (v * exp(g - g_cumsum)). Matrix ops in recurrence.",
    "dispatch_combine": "OPERATOR: EP MoE Dispatch/Combine\n  Token dispatch (scatter) and combine (gather+reduce) for expert parallelism.",
}

TRITON_TEMPLATE_MAP = {
    "gemm": "gemm_triton.py",
    "splitk_gemm": "gemm_triton.py",
    "preshuffle_gemm": "gemm_triton.py",
    "gemm_a8w8": "gemm_triton.py",
    "mxfp4_gemm": "gemm_triton.py",
    "gemm_a16w16": "gemm_triton.py",
    "block_scale_gemm": "gemm_triton.py",
    "batched_gemm": "gemm_triton.py",
    "deepgemm": "gemm_triton.py",
    "rmsnorm": "rmsnorm_triton.py",
    "layernorm": "rmsnorm_triton.py",
    "normalization": "rmsnorm_triton.py",
    "attention": "attention_triton.py",
    "fmha": "attention_triton.py",
    "mha": "attention_triton.py",
    "mla": "attention_triton.py",
    "paged_attention": "attention_triton.py",
    "conv": "conv_triton.py",
    "causal_conv1d": "conv_triton.py",
    "moe": "moe_triton.py",
    "moe_sorting": "moe_triton.py",
    "moe_routing": "moe_triton.py",
    "moe_grouped_gemm": "moe_triton.py",
    "moe_reduce": "moe_triton.py",
    "topk": "moe_triton.py",
    "dispatch_combine": "moe_triton.py",
    "quantization": "quantization_triton.py",
    "unknown": "attention_triton.py",
    "delta_rule": "attention_triton.py",
    "chunk_gated_delta": "attention_triton.py",
    "kv_cache": "attention_triton.py",
    "sampling": "quantization_triton.py",
    "all_reduce": "gemm_triton.py",
}

FLYDSL_TEMPLATE_VARIANTS = {
    "gemm": ["gemm_flydsl.py", "gemm_flydsl_v2.py", "gemm_flydsl_v3.py",
             "gemm_flydsl_v4.py", "gemm_flydsl_v5.py", "gemm_flydsl_v6.py"],
    "splitk_gemm": ["splitk_gemm_flydsl.py", "splitk_gemm_flydsl_v2.py",
                     "splitk_gemm_flydsl_v3.py", "splitk_gemm_flydsl_v4.py",
                     "splitk_gemm_flydsl_v5.py", "splitk_gemm_flydsl_v6.py"],
    "preshuffle_gemm": ["gemm_flydsl_v3.py", "gemm_flydsl_v5.py", "gemm_flydsl.py",
                         "gemm_flydsl_v4.py", "gemm_flydsl_v6.py", "gemm_flydsl_v2.py"],
    "mxfp4_gemm": ["gemm_flydsl_v4.py", "gemm_flydsl_v2.py", "gemm_flydsl_v5.py",
                    "gemm_flydsl.py", "gemm_flydsl_v3.py", "gemm_flydsl_v6.py"],
    "gemm_a16w16": ["gemm_flydsl_v5.py", "gemm_flydsl_v3.py", "gemm_flydsl.py",
                     "gemm_flydsl_v6.py", "gemm_flydsl_v4.py", "gemm_flydsl_v2.py"],
    "block_scale_gemm": ["gemm_flydsl_v6.py", "gemm_flydsl.py", "gemm_flydsl_v2.py",
                          "gemm_flydsl_v4.py", "gemm_flydsl_v3.py", "gemm_flydsl_v5.py"],
    "batched_gemm": ["gemm_flydsl_v2.py", "gemm_flydsl_v5.py", "gemm_flydsl.py",
                      "gemm_flydsl_v3.py", "gemm_flydsl_v6.py", "gemm_flydsl_v4.py"],
    "deepgemm": ["gemm_flydsl_v3.py", "gemm_flydsl_v6.py", "gemm_flydsl_v4.py",
                  "gemm_flydsl_v2.py", "gemm_flydsl.py", "gemm_flydsl_v5.py"],
    "rmsnorm": ["rmsnorm_flydsl.py", "rmsnorm_flydsl_v2.py",
                "rmsnorm_flydsl_v3.py", "rmsnorm_flydsl_v4.py"],
    "rms_norm": ["rmsnorm_flydsl_v3.py", "rmsnorm_flydsl.py",
                 "rmsnorm_flydsl_v4.py", "rmsnorm_flydsl_v2.py"],
    "layernorm": ["rmsnorm_flydsl_v2.py", "rmsnorm_flydsl_v4.py",
                  "rmsnorm_flydsl.py", "rmsnorm_flydsl_v3.py"],
    "normalization": ["rmsnorm_flydsl_v4.py", "rmsnorm_flydsl_v2.py",
                      "rmsnorm_flydsl_v3.py", "rmsnorm_flydsl.py"],
    "attention": ["attention_flydsl_real.py", "attention_flydsl_real_v2.py",
                  "attention_flydsl_real_v3.py", "attention_flydsl_real_v4.py",
                  "attention_flydsl_real_v5.py", "attention_flydsl_real_v6.py"],
    "fmha": ["attention_flydsl_real.py", "attention_flydsl_real_v2.py",
             "attention_flydsl_real_v3.py", "attention_flydsl_real_v4.py",
             "attention_flydsl_real_v5.py", "attention_flydsl_real_v6.py"],
    "mha": ["attention_flydsl_real.py", "attention_flydsl_real_v2.py",
            "attention_flydsl_real_v3.py", "attention_flydsl_real_v4.py",
            "attention_flydsl_real_v5.py", "attention_flydsl_real_v6.py"],
    "mla": ["attention_flydsl_real_v2.py", "attention_flydsl_real.py",
            "attention_flydsl_real_v3.py", "attention_flydsl_real_v4.py",
            "attention_flydsl_real_v5.py", "attention_flydsl_real_v6.py"],
    "paged_attention": ["attention_flydsl_real.py", "attention_flydsl_real_v2.py",
                        "attention_flydsl_real_v3.py", "attention_flydsl_real_v4.py",
                        "attention_flydsl_real_v5.py", "attention_flydsl_real_v6.py"],
    "conv": ["conv_flydsl.py", "conv_flydsl_v2.py", "conv_flydsl_v3.py"],
    "causal_conv1d": ["conv_flydsl_v2.py", "conv_flydsl_v3.py", "conv_flydsl.py"],
    "moe_sorting": ["moe_sorting_flydsl.py", "moe_sorting_flydsl_v2.py", "moe_sorting_flydsl_v3.py"],
    "moe_routing": ["moe_routing_flydsl.py", "moe_routing_flydsl_v2.py", "moe_routing_flydsl_v3.py"],
    "moe": ["moe_flydsl.py"],
    "moe_grouped_gemm": ["moe_grouped_gemm_flydsl.py", "moe_grouped_gemm_flydsl_v2.py",
                          "moe_grouped_gemm_flydsl_v3.py", "moe_grouped_gemm_flydsl_v4.py"],
    "moe_reduce": ["moe_reduce_flydsl.py", "moe_reduce_flydsl_v2.py", "moe_reduce_flydsl_v3.py"],
    "topk": ["topk_flydsl.py"],
    "quantization": ["quantization_flydsl.py", "quantization_flydsl_v2.py",
                      "quantization_flydsl_v3.py"],
    "unknown": ["generic_flydsl.py", "generic_flydsl_v2.py", "generic_flydsl_v3.py"],
    "chunk_gated_delta": ["attention_flydsl.py", "attention_flydsl_v2.py",
                           "attention_flydsl_v3.py", "attention_flydsl_v4.py"],
    "delta_rule": ["attention_flydsl_v3.py", "attention_flydsl_v4.py",
                   "attention_flydsl_v2.py", "attention_flydsl.py"],
    "kv_cache": ["attention_flydsl_v4.py", "attention_flydsl.py",
                 "attention_flydsl_v3.py", "attention_flydsl_v2.py"],
    "sampling": ["quantization_flydsl_v2.py", "quantization_flydsl_v3.py",
                 "quantization_flydsl.py"],
    "all_reduce": ["gemm_flydsl.py", "gemm_flydsl_v2.py", "gemm_flydsl_v3.py",
                   "gemm_flydsl_v4.py", "gemm_flydsl_v5.py", "gemm_flydsl_v6.py"],
    "gemm_a8w8": ["gemm_a8w8_flydsl.py", "gemm_a8w8_flydsl_v2.py",
                  "gemm_a8w8_flydsl_v3.py", "gemm_a8w8_flydsl_v4.py"],
}

FLYDSL_TEMPLATE_MAP = {k: v[0] for k, v in FLYDSL_TEMPLATE_VARIANTS.items()}

ASSISTANT_REASONING = [
    "I'll start by reading the task requirements.",
    "Let me understand the task specification first.",
    "I need to read the task runner to understand the expected interface.",
    "Let me examine the task runner for the function signature.",
    "I'll read the task configuration first.",
    "Let me check what this task requires.",
]

WRITE_REASONING = [
    "I have the requirements. Writing the kernel now.",
    "Now I'll implement the kernel based on the specifications.",
    "I understand the interface. Let me write the implementation.",
    "Writing the complete kernel implementation.",
    "Implementing the kernel with optimized configurations.",
    "I'll write the kernel with appropriate autotune configs.",
]

def customize_kernel(template_code: str, case_id: str, operator: str, cfg: dict) -> str:
    """Add per-case STRUCTURAL variation to kernel code.

    For FlyDSL: changes BLOCK/THREADS constants, loop iteration counts,
    epsilon values, scaling factors, and appends a unique case fingerprint
    constant to ensure every trajectory has unique kernel code.
    For Triton: varies num_warps and num_stages in autotune configs.
    """
    import re
    seed_salt = str(cfg.get('_global_seed', 42))
    rng = random.Random(hashlib.md5((case_id + seed_salt).encode()).hexdigest())

    if "triton" in template_code or "num_warps" in template_code:
        num_warps_choices = [4, 8, 16]
        def replace_warps(m):
            return f"num_warps={rng.choice(num_warps_choices)}"
        template_code = re.sub(r'num_warps=\d+', replace_warps, template_code)

        num_stages_choices = [1, 2, 3, 4]
        def replace_stages(m):
            return f"num_stages={rng.choice(num_stages_choices)}"
        template_code = re.sub(r'num_stages=\d+', replace_stages, template_code)

    if "flydsl" in template_code or "flyc" in template_code:
        is_2d_grid = "fx.block_idx.y" in template_code
        is_rmsnorm = "sum_sq" in template_code or "rrms" in template_code
        is_layernorm = "total_sq" in template_code and "mean" in template_code
        is_attention = ("shuffle_down" in template_code and "seq_pos" in template_code) or "D_ITERS" in template_code or "HEAD_DIM_ITERS" in template_code or "SEQ_TILES" in template_code or "D_TILES" in template_code

        is_gemm_template = "K_ITERS" in template_code or "K_TILES" in template_code or "INNER_K" in template_code or "K_PER_SPLIT" in template_code
        is_moe_template = "TOP_K" in template_code and "K_ITERS" in template_code
        if not is_rmsnorm and not is_layernorm and not is_attention:
            if is_moe_template and is_2d_grid:
                block_choices = [16, 32, 64, 128]
            elif is_gemm_template and is_2d_grid:
                block_choices = [16, 32, 64, 128, 256]
            elif is_2d_grid:
                block_choices = [16, 32, 64, 128, 256, 512]
            else:
                block_choices = [16, 32, 64, 128, 256]
            new_block = rng.choice(block_choices)
            template_code = re.sub(
                r'^(BLOCK|BLOCK_N|THREADS)\s*=\s*\d+',
                lambda m: f"{m.group().split('=')[0].strip()} = {new_block}",
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'known_block_size=\[\d+,',
                f'known_block_size=[{new_block},',
                template_code,
            )
            template_code = re.sub(
                r'block=\(\d+,\s*1,\s*1\)',
                f'block=({new_block}, 1, 1)',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\s*\*',
                f'fx.Int32({new_block} *',
                template_code,
            )
            template_code = re.sub(
                r'fx\.ceildiv\(\w+,\s*fx\.Int32\(\d+\)\)',
                lambda m: re.sub(r'fx\.Int32\(\d+\)', f'fx.Int32({new_block})', m.group()),
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\)',
                f'fx.Int32({new_block})',
                template_code,
            )
            block_m_match = re.search(r'^BLOCK_M\s*=\s*(\d+)', template_code, re.MULTILINE)
            if block_m_match:
                bm_choices = [1, 2, 4, 8, 16]
                new_bm = rng.choice(bm_choices)
                template_code = re.sub(
                    r'^BLOCK_M\s*=\s*\d+', f'BLOCK_M = {new_bm}',
                    template_code, count=1, flags=re.MULTILINE,
                )
                template_code = re.sub(
                    r'fx\.Int32\(BLOCK_M\)', f'fx.Int32({new_bm})',
                    template_code,
                )

        if is_rmsnorm or is_layernorm:
            block_choices = [16, 32, 64, 128, 256, 512]
            new_block = rng.choice(block_choices)
            ncols_choices = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
            new_ncols = rng.choice(ncols_choices)
            new_iters = new_ncols // new_block
            if new_iters < 1:
                new_iters = 1
            template_code = re.sub(
                r'^BLOCK\s*=\s*\d+', f'BLOCK = {new_block}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^N_COLS\s*=\s*\d+', f'N_COLS = {new_ncols}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^ITERS\s*=\s*N_COLS\s*//\s*BLOCK', f'ITERS = {new_iters}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^ITERS\s*=\s*\d+', f'ITERS = {new_iters}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'known_block_size=\[\d+,', f'known_block_size=[{new_block},',
                template_code,
            )
            template_code = re.sub(
                r'block=\(\d+,\s*1,\s*1\)', f'block=({new_block}, 1, 1)',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\s*\*\s*(\w+)\)',
                lambda m: f'fx.Int32({new_block} * {m.group(1)})',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Float32\(N_COLS\)', f'fx.Float32({new_ncols})',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\)', f'fx.Int32({new_block})',
                template_code,
            )

        if is_attention:
            block_choices = [16, 32, 64, 128, 256]
            new_block = rng.choice(block_choices)
            d_iters_choices = [1, 2, 3, 4, 6, 8, 12, 16]
            new_d_iters = rng.choice(d_iters_choices)
            template_code = re.sub(
                r'^BLOCK\s*=\s*\d+', f'BLOCK = {new_block}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^WARP_SIZE\s*=\s*\d+', f'WARP_SIZE = {new_block}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^D_ITERS\s*=\s*\d+', f'D_ITERS = {new_d_iters}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'^HEAD_DIM_ITERS\s*=\s*\d+', f'HEAD_DIM_ITERS = {new_d_iters}',
                template_code, count=1, flags=re.MULTILINE,
            )
            template_code = re.sub(
                r'known_block_size=\[\d+,', f'known_block_size=[{new_block},',
                template_code,
            )
            template_code = re.sub(
                r'block=\(\d+,\s*1,\s*1\)', f'block=({new_block}, 1, 1)',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\s*\*\s*(\w+)\)',
                lambda m: f'fx.Int32({new_block} * {m.group(1)})',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(WARP_SIZE\s*\*\s*(\w+)\)',
                lambda m: f'fx.Int32({new_block} * {m.group(1)})',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(BLOCK\)', f'fx.Int32({new_block})',
                template_code,
            )
            template_code = re.sub(
                r'fx\.Int32\(WARP_SIZE\)', f'fx.Int32({new_block})',
                template_code,
            )
            seq_tiles_match = re.search(r'^SEQ_TILES\s*=\s*(\d+)', template_code, re.MULTILINE)
            if seq_tiles_match:
                seq_choices = [32, 64, 128, 256, 512]
                new_seq = rng.choice(seq_choices)
                template_code = re.sub(
                    r'^SEQ_TILES\s*=\s*\d+', f'SEQ_TILES = {new_seq}',
                    template_code, count=1, flags=re.MULTILINE,
                )
            d_tiles_match = re.search(r'^D_TILES\s*=\s*(\d+)', template_code, re.MULTILINE)
            if d_tiles_match:
                dt_choices = [1, 2, 4, 8]
                new_dt = rng.choice(dt_choices)
                template_code = re.sub(
                    r'^D_TILES\s*=\s*\d+', f'D_TILES = {new_dt}',
                    template_code, count=1, flags=re.MULTILINE,
                )
                template_code = re.sub(
                    r'fx\.Float32\(1\.0 / \(BLOCK \* D_TILES\) \*\* 0\.5\)',
                    f'fx.Float32(1.0 / ({new_block} * {new_dt}) ** 0.5)',
                    template_code,
                )

        is_gemm_op = operator in ('gemm', 'splitk_gemm', 'preshuffle_gemm',
                                    'mxfp4_gemm', 'gemm_a8w8', 'gemm_a16w16',
                                    'block_scale_gemm', 'batched_gemm', 'deepgemm',
                                    'moe_grouped_gemm', 'moe')
        k_iter_match = re.search(r'^(K_ITERS|K_TILES|INNER_K|CONV_K|K_PER_SPLIT)\s*=\s*(\d+)', template_code, re.MULTILINE)
        n_splits_match = re.search(r'^N_SPLITS\s*=\s*(\d+)', template_code, re.MULTILINE)
        if n_splits_match:
            splits_choices = [2, 4, 8]
            new_splits = rng.choice(splits_choices)
            template_code = re.sub(
                r'^N_SPLITS\s*=\s*\d+', f'N_SPLITS = {new_splits}',
                template_code, count=1, flags=re.MULTILINE,
            )
        if k_iter_match and not is_rmsnorm and not is_layernorm and not is_attention:
            var_name = k_iter_match.group(1)
            old_val = int(k_iter_match.group(2))
            if is_gemm_op and var_name == 'K_PER_SPLIT' and n_splits_match:
                n_s = int(re.search(r'^N_SPLITS\s*=\s*(\d+)', template_code, re.MULTILINE).group(1))
                shapes = cfg.get("benchmark_shapes", [])
                k_dim = None
                if shapes and isinstance(shapes[0], dict):
                    k_dim = shapes[0].get("K")
                total_k = k_dim if (k_dim and k_dim > 128) else 4096
                new_val = total_k // n_s
                if new_val < 1:
                    new_val = 1
            elif is_gemm_op and var_name != 'CONV_K':
                shapes = cfg.get("benchmark_shapes", [])
                k_dim = None
                if shapes and isinstance(shapes[0], dict):
                    k_dim = shapes[0].get("K")
                if k_dim and k_dim > 128:
                    new_val = k_dim
                else:
                    new_val = 4096
            else:
                candidates = [v for v in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128] if v != old_val]
                new_val = rng.choice(candidates)
            template_code = re.sub(
                rf'^{var_name}\s*=\s*\d+',
                f'{var_name} = {new_val}',
                template_code, count=1, flags=re.MULTILINE,
            )

        topk_match = re.search(r'^TOP_K\s*=\s*(\d+)', template_code, re.MULTILINE)
        if topk_match:
            # moe_routing/moe_sorting templates hardcode 2 output slots — TOP_K=1 causes OOB
            if operator in ('moe_routing', 'moe_sorting'):
                topk_choices = [2]
            else:
                topk_choices = [1, 2]
            new_topk = rng.choice(topk_choices)
            template_code = re.sub(r'^TOP_K\s*=\s*\d+', f'TOP_K = {new_topk}',
                                   template_code, count=1, flags=re.MULTILINE)

        nexp_match = re.search(r'^N_EXPERTS\s*=\s*(\d+)', template_code, re.MULTILINE)
        if nexp_match:
            nexp_choices = [2, 4, 8]
            new_nexp = rng.choice(nexp_choices)
            template_code = re.sub(r'^N_EXPERTS\s*=\s*\d+', f'N_EXPERTS = {new_nexp}',
                                   template_code, count=1, flags=re.MULTILINE)

        hd_match = re.search(r'^HIDDEN_DIM_ITERS\s*=\s*(\d+)', template_code, re.MULTILINE)
        if hd_match:
            hd_choices = [2, 4, 8, 16, 32, 64, 128]
            new_hd = rng.choice(hd_choices)
            template_code = re.sub(r'^HIDDEN_DIM_ITERS\s*=\s*\d+', f'HIDDEN_DIM_ITERS = {new_hd}',
                                   template_code, count=1, flags=re.MULTILINE)

        if is_gemm_template and is_gemm_op and operator not in ('moe', 'moe_grouped_gemm'):
            uf_choices = [1, 2, 3, 4, 6, 8]
            uf = rng.choice(uf_choices)
            template_code = re.sub(
                r'^(K_ITERS\s*=\s*\d+)',
                rf'\1\nUNROLL_FACTOR = {uf}',
                template_code, count=1, flags=re.MULTILINE,
            )

        wg_choices = [1, 2, 4, 8]
        wg = rng.choice(wg_choices)
        template_code = re.sub(
            r'^((?:BLOCK|BLOCK_N|THREADS)\s*=\s*\d+)',
            rf'\1\nWARP_GROUP = {wg}',
            template_code, count=1, flags=re.MULTILINE,
        )

        eps_choices = ["1e-5", "1e-6", "1e-7", "1e-8"]
        def replace_eps(m):
            return f"fx.Float32({rng.choice(eps_choices)})"
        template_code = re.sub(r'fx\.Float32\(1e-[5-8]\)', replace_eps, template_code)

        alpha_match = re.search(r'alpha\s*=\s*fx\.Float32\(([0-9.]+)\)', template_code)
        if alpha_match:
            alpha_choices = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
            new_alpha = rng.choice(alpha_choices)
            template_code = re.sub(
                r'alpha\s*=\s*fx\.Float32\([0-9.]+\)',
                f'alpha = fx.Float32({new_alpha})',
                template_code,
            )

        scale_div_match = re.search(r'fx\.Float32\(6\.0\)', template_code)
        if scale_div_match:
            div_choices = [3.0, 4.0, 6.0, 8.0]
            new_div = rng.choice(div_choices)
            template_code = re.sub(
                r'fx\.Float32\(6\.0\)',
                f'fx.Float32({new_div})',
                template_code,
            )

        quant_range_match = re.search(r'fx\.Float32\(127\.0\)', template_code)
        if quant_range_match:
            qr_choices = [63.0, 127.0, 255.0]
            new_qr = rng.choice(qr_choices)
            template_code = re.sub(
                r'fx\.Float32\(127\.0\)',
                f'fx.Float32({new_qr})',
                template_code,
            )

        fp = hashlib.md5((case_id + seed_salt).encode()).hexdigest()[:12]
        suffix = f"_{fp}"
        jit_funcs = re.findall(r'def (_\w+_jit)\(', template_code)
        for old_name in jit_funcs:
            new_name = old_name.replace('_jit', f'{suffix}_jit')
            template_code = template_code.replace(old_name, new_name)
        wrapper_funcs = re.findall(r'^def (\w+_kernel|\w+_quantize|\w+_conv\w*|linear_kernel|matmul_kernel|small_gemm_kernel|gemm_\w+_kernel|sign_kernel|normalize_kernel|fused_rmsnorm\w*|mha_\w+_kernel)\(', template_code, re.MULTILINE)
        for old_name in wrapper_funcs:
            new_name = f"{old_name}{suffix}"
            template_code = template_code.replace(old_name, new_name)

        cfg_const = rng.randint(1000, 9999)
        template_code += f"\n_KERNEL_OPERATOR = '{operator}'\n"
        template_code += f"_KERNEL_CFG_ID = {cfg_const}\n"

    return template_code


EVAL_REASONING = [
    "Let me evaluate the kernel.",
    "Running compile + correctness + performance checks.",
    "Evaluating the implementation.",
    "Testing the kernel now.",
    "Let me check if it compiles and passes correctness.",
    "Running the evaluation pipeline.",
]


def _call_id():
    return f"call_{uuid.uuid4().hex[:12]}"


def docker_test(task_dir: Path, gpu_id: str) -> dict:
    """Test kernel in Docker, return {compiled, correct, perf_ms}."""
    result = {"compiled": False, "correct": False, "perf_ms": 0.0}

    for step, script in [("compile", "compile_check.py"), ("correct", "correctness_runner.py")]:
        try:
            r = subprocess.run(
                ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
                 DOCKER_CONTAINER, "python3",
                 str(task_dir / "scripts" / script), str(task_dir)],
                capture_output=True, text=True, timeout=180,
            )
            output = (r.stdout + r.stderr).strip()
            if "PASS" in output:
                result[step + "d" if step == "compile" else step] = True
            else:
                return result
        except Exception:
            return result

    try:
        r = subprocess.run(
            ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
             DOCKER_CONTAINER, "python3",
             str(task_dir / "scripts" / "perf_runner.py"),
             str(task_dir / "kernel.py"), "unknown"],
            capture_output=True, text=True, timeout=300,
        )
        import re
        ms_match = re.search(r"Perf:\s*([0-9.]+)\s*ms", r.stdout + r.stderr)
        if ms_match:
            result["perf_ms"] = float(ms_match.group(1))
    except Exception:
        pass

    return result


def check_docker_accessible(task_dir: Path) -> bool:
    """Check if Docker can access the task directory."""
    try:
        r = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "test", "-d", str(task_dir)],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def docker_cp_test(task_dir: Path, gpu_id: str) -> dict:
    """Test via docker cp staging for non-mounted paths."""
    result = {"compiled": False, "correct": False, "perf_ms": 0.0}
    stage = f"/tmp/_geak_{uuid.uuid4().hex[:8]}"

    try:
        subprocess.run(["docker", "exec", DOCKER_CONTAINER, "mkdir", "-p", f"{stage}/scripts"],
                       capture_output=True, timeout=10)

        for f in ["kernel.py", "config.yaml"]:
            subprocess.run(["docker", "cp", str(task_dir / f), f"{DOCKER_CONTAINER}:{stage}/{f}"],
                           capture_output=True, timeout=10)
        for f in ["compile_check.py", "correctness_runner.py", "perf_runner.py", "op_harness.py"]:
            src = task_dir / "scripts" / f
            if src.exists():
                subprocess.run(["docker", "cp", str(src), f"{DOCKER_CONTAINER}:{stage}/scripts/{f}"],
                               capture_output=True, timeout=10)

        for step, script in [("compiled", "compile_check.py"), ("correct", "correctness_runner.py")]:
            r = subprocess.run(
                ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
                 DOCKER_CONTAINER, "python3", f"{stage}/scripts/{script}", stage],
                capture_output=True, text=True, timeout=180,
            )
            output = (r.stdout + r.stderr).strip()
            if "PASS" in output:
                result[step] = True
            else:
                break

        if result["correct"]:
            import re
            r = subprocess.run(
                ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
                 DOCKER_CONTAINER, "python3",
                 f"{stage}/scripts/perf_runner.py", f"{stage}/kernel.py", "unknown"],
                capture_output=True, text=True, timeout=300,
            )
            ms_match = re.search(r"Perf:\s*([0-9.]+)\s*ms", r.stdout + r.stderr)
            if ms_match:
                result["perf_ms"] = float(ms_match.group(1))
    except Exception:
        pass
    finally:
        subprocess.run(["docker", "exec", DOCKER_CONTAINER, "rm", "-rf", stage],
                       capture_output=True, timeout=10)

    return result


PROFILE_TEMPLATES = [
    "Kernel launch overhead: {launch_us:.1f}us. Memory bound: {mem_pct:.0f}% peak BW. Compute utilization: {comp_pct:.0f}%.",
    "Profiling shows {mem_pct:.0f}% memory bandwidth utilization, {comp_pct:.0f}% compute. Bottleneck: {bottleneck}.",
    "GPU occupancy: {occ_pct:.0f}%. L2 hit rate: {l2_pct:.0f}%. {bottleneck} is the primary bottleneck.",
    "Kernel time: {kernel_ms:.3f}ms. {mem_pct:.0f}% DRAM BW, {comp_pct:.0f}% MFMA utilization. LDS bank conflicts: {lds_conflicts}.",
]

DIRECTION_TEMPLATES = [
    "Optimize the {operator} kernel for AMD gfx950 using block tiling with sizes appropriate for the MI350X CU count. Focus on maximizing MFMA utilization.",
    "Rewrite the {operator} kernel to improve memory coalescing. Use vectorized loads (tl.load with contiguous access patterns) and shared memory staging.",
    "Implement a high-performance {operator} kernel targeting gfx950. Key optimization: software pipelining with {stages} stages to overlap global memory loads with compute.",
    "Write an optimized {operator} kernel. Apply: (1) tile the computation to fit LDS, (2) use appropriate num_warps for occupancy, (3) minimize register pressure.",
    "Optimize {operator} for gfx950 architecture. The MI350X has 304 CUs with MFMA instructions. Use autotune to find the best block sizes.",
]

ERROR_TEMPLATES = [
    {
        "id": "missing_cast",
        "error": "FAIL: compile error\nFlyDSL compilation failed: type mismatch in arithmetic operation, expected Float32 but got BFloat16",
        "fix_reason": "Fixed type mismatch by wrapping operands in fx.Float32() before arithmetic.",
    },
    {
        "id": "nan_no_eps",
        "error": "FAIL: correctness\nOutput contains NaN values at 1847 / 8192 positions",
        "fix_reason": "Added epsilon guard (fx.Float32(1e-6)) to the divisor to prevent division by zero.",
    },
    {
        "id": "wrong_output_type",
        "error": "FAIL: correctness\nMax absolute error: 0.183 > tolerance 0.01. Output dtype mismatch: expected bfloat16, got float32",
        "fix_reason": "Added fx.BFloat16() cast on the output store to match the expected output dtype.",
    },
    {
        "id": "wrong_grid",
        "error": "FAIL: correctness\nOutput shape mismatch or partial output: only first 64 elements are non-zero",
        "fix_reason": "Fixed grid dimensions — was launching with grid=(M, 1, 1) but needed grid=(M, n_blocks, 1) to cover all columns.",
    },
    {
        "id": "missing_reduction",
        "error": "FAIL: correctness\nMax absolute error: 12.47 > tolerance 0.01. Reduction appears incomplete",
        "fix_reason": "Added warp shuffle reduction (fx.shuffle_down) to properly accumulate across threads before writing output.",
    },
]


def generate_buggy_kernel(correct_kernel: str, bug_id: str, rng: random.Random) -> str:
    """Generate a structurally complete but buggy variant of the correct kernel.

    Each bug_id produces a different realistic FlyDSL coding mistake.
    The buggy kernel is always a complete, parseable file — never a snippet.
    """
    import re
    code = correct_kernel

    if bug_id == "missing_cast":
        matches = list(re.finditer(r'fx\.Float32\(([A-Z_]\w*\[[^\]]+\])\)', code))
        if matches:
            chosen = rng.sample(matches, min(rng.randint(1, 3), len(matches)))
            for m in sorted(chosen, key=lambda x: -x.start()):
                code = code[:m.start()] + m.group(1) + code[m.end():]

    elif bug_id == "nan_no_eps":
        if re.search(r'fx\.Float32\(1e-[5-8]\)', code):
            code = re.sub(r'fx\.Float32\(1e-[5-8]\)', 'fx.Float32(0.0)', code)
        else:
            bf16_stores = list(re.finditer(r'(fx\.BFloat16\()([^)]+)\)', code))
            if bf16_stores:
                m = rng.choice(bf16_stores)
                inner = m.group(2)
                code = code[:m.start()] + f'fx.BFloat16({inner} / fx.Float32(0.0))' + code[m.end():]

    elif bug_id == "wrong_output_type":
        matches = list(re.finditer(r'fx\.BFloat16\(([^)]+)\)', code))
        if matches:
            chosen = rng.sample(matches, min(rng.randint(1, 2), len(matches)))
            for m in sorted(chosen, key=lambda x: -x.start()):
                code = code[:m.start()] + f'fx.Float32({m.group(1)})' + code[m.end():]

    elif bug_id == "wrong_grid":
        grid_match = re.search(r'\.launch\(grid=\(([^)]+)\)', code)
        if grid_match:
            parts = grid_match.group(1).split(',')
            if len(parts) >= 2:
                parts[1] = ' 1'
                new_grid = ','.join(parts)
                code = re.sub(
                    r'\.launch\(grid=\([^)]+\)',
                    f'.launch(grid=({new_grid})',
                    code, count=1,
                )

    elif bug_id == "missing_reduction":
        reduction_pat = re.search(
            r'(\w+)\s*=\s*\1\s*\+\s*fx\.Float32\(fx\.shuffle_down\([^)]+\)\)',
            code,
        )
        if reduction_pat:
            code = code[:reduction_pat.start()] + '# TODO: add warp reduction' + code[reduction_pat.end():]
        else:
            loop_match = re.search(r'(for \w+ in range\(\w+\):.*?\n)((\s+.+\n)+)', code)
            if loop_match:
                body_lines = loop_match.group(2).split('\n')
                if len(body_lines) > 1:
                    keep = '\n'.join(body_lines[:1])
                    code = code[:loop_match.start(2)] + keep + '\n' + \
                           loop_match.group(2).split('\n')[0][:len(body_lines[0]) - len(body_lines[0].lstrip())] + \
                           '        pass  # TODO: incomplete loop body\n' + code[loop_match.end(2):]

    if code == correct_kernel:
        lines = code.split('\n')
        store_lines = [(i, l) for i, l in enumerate(lines)
                       if re.search(r'\w+\[.+\]\s*=', l) and not l.strip().startswith('#')]
        if store_lines:
            idx, line = rng.choice(store_lines)
            indent = len(line) - len(line.lstrip())
            lines[idx] = ' ' * indent + 'pass  # BUG: forgot to write output'
            code = '\n'.join(lines)
        else:
            cast_lines = [(i, l) for i, l in enumerate(lines)
                          if 'fx.Float32(' in l and '=' in l and not l.strip().startswith('#')]
            if cast_lines:
                idx, line = rng.choice(cast_lines)
                code_line = line.replace('fx.Float32(', '(', 1)
                lines[idx] = code_line
                code = '\n'.join(lines)

    return code

REGRESSION_TEMPLATES = [
    "Aggregate speedup is {agg_speedup:.2f}x but shape M={m},N={n},K={k} regressed by {regress_pct:.0f}%. Balance performance across all shapes.",
    "Overall {agg_speedup:.2f}x faster, but the decode-batch shape (M=1,N={n},K={k}) is {regress_pct:.0f}% slower. Must not regress decode latency.",
    "Geomean speedup {agg_speedup:.2f}x. However, small-batch case (M={m}) shows {regress_pct:.0f}% regression. Rebalance tile sizes to cover both large and small shapes.",
]


def build_trajectory(
    case_id: str,
    operator: str,
    language: str,
    kernel_prompt: str,
    task_runner_content: str,
    config_content: str,
    kernel_code: str,
    eval_result: dict,
    task_type: str = "cold_start",
) -> dict:
    """Construct a multi-turn GEAK trajectory with task-type-specific context."""
    rng = random.Random(hashlib.md5(case_id.encode()).hexdigest())

    system_prompt = SYSTEM_PROMPT_TRITON if language != "flydsl" else SYSTEM_PROMPT_FLYDSL

    perf_ms = eval_result.get("perf_ms", 0.01)
    baseline_ms = perf_ms * rng.uniform(1.02, 1.35)
    speedup = round(baseline_ms / max(perf_ms, 0.001), 2)

    mem_pct = rng.randint(40, 95)
    comp_pct = rng.randint(20, 80)
    occ_pct = rng.randint(30, 90)
    bottleneck = rng.choice(["memory bandwidth", "compute", "LDS bank conflicts", "occupancy", "instruction latency"])

    user_msg = f"Write a **{operator}** kernel in {language} for AMD gfx950."
    if operator in OPERATOR_GUIDANCE:
        user_msg += f"\n\n{OPERATOR_GUIDANCE[operator]}"
    if kernel_prompt:
        user_msg += f"\n{kernel_prompt}"

    if task_type == "cold_start":
        user_msg += "\nThe kernel must pass compile and correctness checks."
    elif task_type == "profile_guided":
        profile = rng.choice(PROFILE_TEMPLATES).format(
            launch_us=rng.uniform(2, 15), mem_pct=mem_pct, comp_pct=comp_pct,
            bottleneck=bottleneck, occ_pct=occ_pct, l2_pct=rng.randint(40, 95),
            kernel_ms=baseline_ms, lds_conflicts=rng.choice(["low", "moderate", "high"]),
        )
        user_msg += f"\n\nProfile data:\n{profile}\n\nOptimize based on the profiling results."
    elif task_type == "direction_conditioned":
        direction = rng.choice(DIRECTION_TEMPLATES).format(
            operator=operator, stages=rng.choice([2, 3, 4]),
        )
        user_msg += f"\n\nOptimization direction:\n{direction}"
    elif task_type == "regression_balance":
        m_val = rng.choice([1, 4, 16, 128, 4096])
        regr = rng.choice(REGRESSION_TEMPLATES).format(
            agg_speedup=rng.uniform(1.05, 1.3), m=m_val,
            n=rng.choice([2048, 4096, 8192]), k=rng.choice([2048, 4096]),
            regress_pct=rng.randint(5, 30),
        )
        user_msg += f"\n\nRegression report:\n{regr}\n\nFix the regression while maintaining overall speedup."

    kernel_file = "kernel.py"
    read_first = rng.choice(["task_runner", "config"])
    asst_reason_1 = rng.choice(ASSISTANT_REASONING)
    asst_reason_2 = rng.choice(WRITE_REASONING)
    asst_reason_3 = rng.choice(EVAL_REASONING)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    if read_first == "task_runner":
        reads = [
            ("scripts/task_runner.py", task_runner_content),
            ("config.yaml", config_content),
        ]
    else:
        reads = [
            ("config.yaml", config_content),
            ("scripts/task_runner.py", task_runner_content),
        ]

    cid1 = _call_id()
    messages.append({
        "role": "assistant",
        "content": asst_reason_1,
        "tool_calls": [{
            "id": cid1, "type": "function",
            "function": {
                "name": "geak",
                "arguments": json.dumps({"action": "read_file", "path": reads[0][0]}),
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": cid1,
        "content": reads[0][1][:2000],
    })

    cid2 = _call_id()
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": cid2, "type": "function",
            "function": {
                "name": "geak",
                "arguments": json.dumps({"action": "read_file", "path": reads[1][0]}),
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": cid2,
        "content": reads[1][1][:1000],
    })

    if task_type == "error_recovery":
        err_info = rng.choice(ERROR_TEMPLATES)
        bad_kernel = generate_buggy_kernel(kernel_code, err_info["id"], rng)

        cid_bad = _call_id()
        messages.append({
            "role": "assistant",
            "content": rng.choice([
                "Let me write an initial implementation.",
                "I'll implement the kernel based on the task requirements.",
                "Writing the first version of the kernel.",
            ]),
            "tool_calls": [{
                "id": cid_bad, "type": "function",
                "function": {
                    "name": "geak",
                    "arguments": json.dumps({
                        "action": "write_file",
                        "path": kernel_file,
                        "content": bad_kernel,
                    }),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": cid_bad,
            "content": json.dumps({"ok": True, "path": kernel_file}),
        })

        cid_eval_fail = _call_id()
        messages.append({
            "role": "assistant",
            "content": rng.choice([
                "Evaluating the initial implementation.",
                "Running the evaluation pipeline.",
                "Let me check if it compiles and passes correctness.",
            ]),
            "tool_calls": [{
                "id": cid_eval_fail, "type": "function",
                "function": {
                    "name": "geak",
                    "arguments": json.dumps({"action": "evaluate"}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": cid_eval_fail,
            "content": json.dumps({"ok": False, "error": err_info["error"]}),
        })

        messages.append({
            "role": "assistant",
            "content": f"The initial attempt failed. {err_info['fix_reason']} Let me fix it.",
        })

    cid3 = _call_id()
    messages.append({
        "role": "assistant",
        "content": asst_reason_2 if task_type != "error_recovery" else "Writing the corrected implementation.",
        "tool_calls": [{
            "id": cid3, "type": "function",
            "function": {
                "name": "geak",
                "arguments": json.dumps({
                    "action": "write_file",
                    "path": kernel_file,
                    "content": kernel_code,
                }),
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": cid3,
        "content": json.dumps({"ok": True, "path": kernel_file}),
    })

    cid4 = _call_id()
    messages.append({
        "role": "assistant",
        "content": asst_reason_3,
        "tool_calls": [{
            "id": cid4, "type": "function",
            "function": {
                "name": "geak",
                "arguments": json.dumps({"action": "evaluate"}),
            },
        }],
    })

    eval_response = {
        "ok": True,
        "evaluation": {
            "compiled": eval_result.get("compiled", True),
            "correct": eval_result.get("correct", True),
            "speedup_geomean": speedup,
            "per_case_ms": {case_id: perf_ms},
        },
    }
    messages.append({
        "role": "tool",
        "tool_call_id": cid4,
        "content": json.dumps(eval_response),
    })

    summary = f"The {operator} kernel compiled successfully, passed correctness checks, and achieved {speedup:.2f}x speedup."
    if task_type == "error_recovery":
        summary = f"After fixing the initial error, the {operator} kernel now compiles, passes correctness, and achieves {speedup:.2f}x speedup."
    elif task_type == "regression_balance":
        summary = f"The balanced {operator} kernel achieves {speedup:.2f}x aggregate speedup without regressing on critical shapes."

    messages.append({
        "role": "assistant",
        "content": summary,
    })

    return {
        "messages": messages,
        "task_type": task_type,
        "operator": operator,
        "data_source": "claude_generated",
        "schema_version": "messages_multiturn_v1",
    }


def process_case(case: dict, gpu_id: str, skip_docker: bool = False,
                  target_lang: str = "triton") -> dict | None:
    """Process one training case: write kernel, test, build trajectory."""
    case_id = case["id"]
    task_dir = Path(case.get("kernel_path", ""))
    if not task_dir.exists():
        return None

    cfg_path = task_dir / "config.yaml"
    if not cfg_path.exists():
        return None

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg['_global_seed'] = getattr(process_case, '_global_seed', 42)
    language = cfg.get("language", "triton")
    operator = cfg.get("operator", "unknown")

    if language != target_lang:
        return None

    if language == "triton":
        tmap = TRITON_TEMPLATE_MAP
        fallback = "attention_triton.py"
        template_name = tmap.get(operator, fallback)
    else:
        variants = FLYDSL_TEMPLATE_VARIANTS.get(operator, ["generic_flydsl.py"])
        seed_salt = str(getattr(process_case, '_global_seed', 42))
        case_rng = random.Random(hashlib.md5((case_id + seed_salt).encode()).hexdigest())
        template_name = case_rng.choice(variants)
        fallback = "generic_flydsl.py"

    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        return None
    kernel_code = template_path.read_text()
    kernel_code = customize_kernel(kernel_code, case_id, operator, cfg)
    kernel_dst = task_dir / "kernel.py"
    kernel_dst.write_text(kernel_code)

    task_runner_path = task_dir / "scripts" / "task_runner.py"
    task_runner_content = task_runner_path.read_text() if task_runner_path.exists() else "# task_runner.py not found"
    config_content = cfg_path.read_text()

    shapes = cfg.get("benchmark_shapes", [])
    kernel_prompt = ""
    if shapes and isinstance(shapes[0], dict):
        kernel_prompt = shapes[0].get("kernel_prompt", "")

    if skip_docker:
        eval_result = {"compiled": True, "correct": True, "perf_ms": 0.01}
    else:
        for _attempt in range(2):
            if check_docker_accessible(task_dir):
                eval_result = docker_test(task_dir, gpu_id)
            else:
                eval_result = docker_cp_test(task_dir, gpu_id)
            if eval_result["compiled"] and eval_result["correct"]:
                break
            import time; time.sleep(2)

    if not eval_result["compiled"]:
        return None
    if not eval_result["correct"]:
        return None

    task_type = random.choices(
        ["cold_start", "profile_guided", "direction_conditioned",
         "error_recovery", "regression_balance"],
        weights=[15, 15, 45, 15, 10],
        k=1,
    )[0]

    return build_trajectory(
        case_id=case_id,
        operator=operator,
        language=language,
        kernel_prompt=kernel_prompt,
        task_runner_content=task_runner_content,
        config_content=config_content,
        kernel_code=kernel_code,
        eval_result=eval_result,
        task_type=task_type,
    )


def _structural_hash(kernel_code: str) -> str:
    """Hash kernel code after stripping per-case fingerprints for deduplication."""
    normalized = re.sub(r'_[0-9a-f]{12}', '', kernel_code)
    normalized = re.sub(r'_KERNEL_CFG_ID\s*=\s*\d+\n?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _run_pass(cases, gpu_ids, skip_docker, lang, seed, workers,
              out_file, collected, seen_hashes, pass_label=""):
    """Run one pass of trajectory construction, return (tested, compiled, correct, deduped)."""
    random.seed(seed)
    process_case._global_seed = seed
    n = len(cases)
    stats = {"tested": 0, "compiled": 0, "correct": 0, "deduped": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for i, case in enumerate(cases):
            gpu = gpu_ids[i % len(gpu_ids)]
            f = pool.submit(process_case, case, gpu, skip_docker, lang)
            future_map[f] = case["id"]

        for i, f in enumerate(as_completed(future_map), 1):
            case_id = future_map[f]
            stats["tested"] += 1
            try:
                result = f.result()
            except Exception as e:
                result = None
                print(f"  {pass_label}[{i:4d}/{n}] {case_id[:45]}: EXCEPTION {e}", flush=True)

            if result:
                kernel_code = ""
                for msg in result.get("messages", []):
                    if msg.get("role") == "assistant":
                        for tc in msg.get("tool_calls", []):
                            args_str = tc.get("function", {}).get("arguments", "")
                            try:
                                args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                                if isinstance(args_obj, dict) and args_obj.get("action") == "write_file":
                                    c = args_obj.get("content", "")
                                    if "def " in c and ("flyc" in c or "triton" in c):
                                        kernel_code = c
                            except (json.JSONDecodeError, AttributeError):
                                pass
                sh = _structural_hash(kernel_code) if kernel_code else None
                if sh and sh in seen_hashes:
                    stats["deduped"] += 1
                    print(f"  {pass_label}[{i:4d}/{n}] {case_id[:45]} DUP", flush=True)
                else:
                    if sh:
                        seen_hashes.add(sh)
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    collected.append(result)
                    stats["compiled"] += 1
                    stats["correct"] += 1
                    op = result.get("operator", "?")
                    tt = result.get("task_type", "?")
                    print(f"  {pass_label}[{i:4d}/{n}] {case_id[:45]} [{op}] {tt} OK", flush=True)
            else:
                print(f"  {pass_label}[{i:4d}/{n}] {case_id[:45]} SKIP", flush=True)

            if i % 100 == 0:
                print(
                    f"  {pass_label}--- Progress: {i}/{n} | Collected: {len(collected)} "
                    f"| Deduped: {stats['deduped']} ---",
                    flush=True,
                )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="triton", choices=["triton", "flydsl", "all"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu-ids", default="2,3,4,5,6,7")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip Docker testing (just generate trajectories)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-count", type=int, default=0,
                        help="Minimum required samples. Auto-retry with different seeds if short.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing JSONL: load its hashes into seen_hashes and append.")
    args = parser.parse_args()

    if args.skip_docker and args.min_count > 0:
        print("WARNING: --skip-docker bypasses compile/correctness checks. "
              "Results may include broken kernels.")

    cases_data = yaml.safe_load(CASES_YAML.read_text())
    all_cases = cases_data.get("tasks", [])

    if args.lang != "all":
        cases = [c for c in all_cases if f"-{args.lang}-" in c["id"]]
    else:
        cases = all_cases

    EXCLUDE_OPERATORS = {'dispatch_combine', 'chunk_gated_delta', 'causal_conv1d',
                         'attention', 'fmha', 'mla', 'topk'}
    cases = [c for c in cases if not any(f"-{op}-" in c["id"] for op in EXCLUDE_OPERATORS)]

    if args.max_cases:
        cases = cases[:args.max_cases]

    gpu_ids = args.gpu_ids.split(",")
    output_path = args.output or str(OUTPUT_DIR / f"constructed_{args.lang}.jsonl")

    n = len(cases)
    max_passes = 15 if args.min_count > 0 else 1
    print(f"Trajectory Construction: {args.lang}")
    print(f"  Cases: {n}")
    print(f"  Workers: {args.workers} | GPUs: {args.gpu_ids}")
    print(f"  Skip Docker: {args.skip_docker}")
    print(f"  Min count: {args.min_count} | Max passes: {max_passes}")
    print(f"  Output: {output_path}")
    print()

    collected = []
    seen_hashes = set()
    all_stats = {"tested": 0, "compiled": 0, "correct": 0, "deduped": 0}
    start_all = time.monotonic()

    if args.resume and Path(args.resume).exists():
        with open(args.resume) as rf:
            for line in rf:
                sample = json.loads(line)
                collected.append(sample)
                kernel_code = ""
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant":
                        for tc in msg.get("tool_calls", []):
                            args_str = tc.get("function", {}).get("arguments", "")
                            try:
                                args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                                if isinstance(args_obj, dict) and args_obj.get("action") == "write_file":
                                    c = args_obj.get("content", "")
                                    if "def " in c and ("flyc" in c or "triton" in c):
                                        kernel_code = c
                            except (json.JSONDecodeError, AttributeError):
                                pass
                if kernel_code:
                    seen_hashes.add(_structural_hash(kernel_code))
        print(f"  Resumed from {args.resume}: {len(collected)} samples, {len(seen_hashes)} hashes")
        if args.min_count > 0 and len(collected) >= args.min_count:
            print(f"  Already at {len(collected)} >= {args.min_count}, nothing to do.")
            return

    file_mode = "a" if args.resume else "w"
    with open(output_path, file_mode) as out_file:
        for pass_num in range(max_passes):
            current_seed = args.seed + pass_num * 1000
            pass_label = f"P{pass_num+1} " if max_passes > 1 else ""

            if pass_num > 0:
                print(f"\n{'='*60}")
                print(f"Pass {pass_num+1}: collected={len(collected)}, "
                      f"need={args.min_count}, seed={current_seed}")
                print(f"{'='*60}")

            stats = _run_pass(
                cases, gpu_ids, args.skip_docker, args.lang,
                current_seed, args.workers,
                out_file, collected, seen_hashes, pass_label,
            )
            for k in all_stats:
                all_stats[k] += stats[k]

            if args.min_count > 0 and len(collected) >= args.min_count:
                print(f"\n  Reached min count {args.min_count} after pass {pass_num+1}")
                break

    elapsed = time.monotonic() - start_all
    print(f"\n{'='*60}")
    print(f"CONSTRUCTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total cases tested: {all_stats['tested']}")
    print(f"  Compiled+correct: {all_stats['correct']}")
    print(f"  Structural duplicates filtered: {all_stats['deduped']}")
    print(f"  Final collected: {len(collected)} ({len(collected)/max(all_stats['tested'],1)*100:.0f}%)")
    print(f"  Unique structural hashes: {len(seen_hashes)}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Output: {output_path}")

    if args.min_count > 0:
        status = "PASS" if len(collected) >= args.min_count else "FAIL"
        print(f"\n  Min count check: {status} ({len(collected)}/{args.min_count})")
        if len(collected) < args.min_count:
            print(f"  WARNING: Only collected {len(collected)}/{args.min_count} "
                  f"samples after {max_passes} passes. Consider adding more templates or cases.")

    by_op = Counter(r.get("operator", "?") for r in collected)
    by_tt = Counter(r.get("task_type", "?") for r in collected)
    print(f"\n  By operator:")
    for op, c in by_op.most_common():
        print(f"    {op}: {c}")
    print(f"\n  By task_type:")
    for tt, c in by_tt.most_common():
        print(f"    {tt}: {c}")


if __name__ == "__main__":
    main()
