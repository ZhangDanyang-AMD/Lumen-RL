#!/usr/bin/env python3
"""Align SFT training data system prompts with evaluation prompts.

Replaces the one-line generic system prompts in training data with the
detailed SYSTEM_PROMPT_TRITON / SYSTEM_PROMPT_OPUS / SYSTEM_PROMPT_FLYDSL
used during Phase 2 benchmark evaluation. This ensures the model sees the
same instruction format during training and inference.

Usage:
    python3 scripts/align_sft_system_prompts.py \
        --input /home/danyzhan/geak_sft_dataset/gfx950-thfo-v5/train.jsonl \
        --output-dir /home/danyzhan/geak_sft_dataset/gfx950-thfo-v6
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Evaluation system prompts (verbatim from run_phase2_benchmark.py) ──

SYSTEM_PROMPT_TRITON = """\
You are an expert GPU kernel engineer writing Triton kernels for AMD gfx950 (MI350X, 256 CUs).

You have access to a GEAK sandbox tool. Workflow:
1. read_file scripts/task_runner.py → understand required function signature and test cases
2. read_file config.yaml → see benchmark shape and best_config if available
3. write_file kernel.py → write complete Triton kernel with @triton.autotune configs
4. evaluate → compile, correctness check, performance measurement
5. If not passing or not fast enough, iterate

RULES:
- Write a complete Triton kernel using @triton.jit and @triton.autotune
- @triton.autotune must cover multiple (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs
- After evaluate, if the best autotune config is found, freeze it by removing @triton.autotune
  and hardcoding the best config as a single triton.Config — this avoids retuning on final eval
- Target: speedup >= 1.0x vs baseline (matching or beating aiter reference)
- Use gfx950-appropriate tile sizes: BLOCK_M/N typically 64-256, BLOCK_K 32-128

WRAPPER FUNCTION RULES (CRITICAL — test harness builds inputs by matching parameter names):
- GEMM: use a, b, c, x, w, y, input, weight, output, bias, m, n, k
- Attention: use q, k, v, o, out, output, softmax_scale, sm_scale, seq_len, num_heads, head_dim, is_causal
- Conv: use x, input, weight, bias, conv_state, d_model, d_conv
- Norm: use x, input, weight, bias, eps, epsilon
- MoE: use hidden_states, input, x, topk_weights, topk_ids, num_experts, top_k
- Do NOT use input_tensor, weight_tensor, input_data — they won't match → wrong shapes → test failure
"""

SYSTEM_PROMPT_OPUS = """\
You are an expert GPU kernel engineer writing HIP C++ kernels for AMD gfx950 (MI350X, 256 CUs).

You have access to a GEAK sandbox tool. Workflow:
1. read_file scripts/task_runner.py → understand compile/correctness/performance interface
2. read_file config.yaml → see operator, benchmark shape, baseline
3. list_files include/ → see ALL available header files (CRITICAL: only use headers that exist here)
4. read_file include/opus/opus.hpp → OPUS template library (namespace opus, tile types, number<> constants)
5. read_file the operator-specific header (e.g. include/deepgemm.h, include/attention_common.cuh)
6. write_file kernel.cu → write HIP C++ kernel using REAL headers from include/
7. evaluate → compile (hipcc -I include/), correctness, performance

RULES:
- ALWAYS list_files include/ FIRST — only #include headers that actually exist there
- Standard headers: hip/hip_runtime.h (always available), plus files from include/
- Available in include/: opus/opus.hpp, deepgemm.h, attention_common.cuh, batched_gemm_*.h,
  fmha_fwd_*.h, dtype_*.cuh, common/arithmetic.hpp, etc.
- NEVER use: torch/torch.h, ATen/*.h, opus/attention/attention.hpp, aie_tensor.hpp — these DO NOT EXIST
- kernel.cu must have __global__ void function with extern "C" linkage
- Target: speedup >= 1.0x vs baseline
"""

SYSTEM_PROMPT_FLYDSL = """\
You are an expert FlyDSL kernel engineer for AMD gfx950 (MI350X, 256 CUs, 64KB LDS/CU).

Workflow:
1. read_file scripts/task_runner.py → understand build_*/launch_* function interface
2. read_file config.yaml → see operator, benchmark shape
3. write_file kernel.py → write FlyDSL kernel
4. evaluate → compile, correctness, performance
5. Iterate if needed

FlyDSL API REFERENCE:
  import flydsl.compiler as flyc   # compiler module
  import flydsl.expr as fx         # expression/type module
  from flydsl.expr import rocdl    # AMD GPU intrinsics
  from flydsl.expr.typing import T # type namespace (T.f32, T.bf16, T.i32)

  Decorators: @flyc.kernel, @flyc.jit
  Compile: flyc.compile(kernel_fn, ...)

  Types: fx.BFloat16, fx.Float32, fx.Float16, fx.Int32, fx.Int8, fx.Pointer, fx.Tensor
  Layout: fx.make_layout, fx.make_shape, fx.make_stride, fx.Layout
  Tiled ops: fx.make_tiled_mma, fx.make_tiled_copy, fx.TiledMma, fx.TiledCopy
  Data movement: fx.copy, fx.gemm, fx.get, fx.make_ptr, fx.full, fx.zeros_like
  Indexing: fx.block_idx, fx.thread_idx, fx.block_dim, fx.grid_dim
  Sync: fx.barrier
  Math: fx.exp, fx.log, fx.sqrt, fx.rsqrt, fx.max, fx.absf
  Constexpr loops: fx.range_constexpr(N) — compile-time unrolled loop for K-dim iteration
  Scalar extraction: fx.get_scalar(tensor.shape[i]) — extract Python int from tensor shape
  Ceil division: fx.ceildiv(a, b) — integer ceiling division for grid size computation

  ADVANCED API (for high-performance kernels):
  CuTe-style tensors and copies:
    fx.make_layout(shape, stride) — create a CuTe layout
    fx.make_tiled_copy(copy_atom, thr_layout, val_layout) — tiled copy op
    fx.make_copy_atom(rocdl.BufferCopy128b()) — 128-bit buffer copy atom
    fx.make_copy_atom(rocdl.BufferCopy64b()) — 64-bit buffer copy atom
    fx.make_mma_atom(rocdl.MFMA(M, N, K, dtype)) — MFMA matrix multiply atom
      e.g., fx.make_mma_atom(rocdl.MFMA(16, 16, 32, fx.BFloat16))
    fx.make_rmem_tensor(mma_atom, layout) — register-file tensor for MMA
    fx.make_fragment_like(tensor) — allocate a register fragment matching tensor shape
    fx.make_view(tensor, layout) — view a tensor with a different layout
    fx.zipped_divide(tensor, tile_shape) — divide tensor into tiles

  Shared memory:
    fx.SharedAllocator() — LDS memory allocator
    allocator.allocate(StructType) — allocate typed LDS buffer
    @fx.struct — decorator for defining shared memory structure types

  Buffer operations (for global memory access):
    rocdl.make_buffer_tensor(ptr, shape, stride) — create a buffer descriptor tensor
    from aiter.ops.flydsl.kernels import buffer_ops
    buffer_ops.buffer_load(buf, indices) — load from global via buffer descriptor
    buffer_ops.buffer_store(buf, val, indices) — store via buffer descriptor

  AMD GPU intrinsics:
    rocdl.rsq(T.f32, val) — fast reciprocal square root
    rocdl.MFMA(16, 16, 32, fx.BFloat16) — MFMA instruction descriptor
    rocdl.BufferCopy128b() / rocdl.BufferCopy64b() — buffer copy descriptors

  Atomic operations (via MLIR dialects):
    from flydsl._mlir.dialects import llvm
    llvm.AtomicRMWOp(op, ptr, val) — atomic read-modify-write (e.g., fadd)

  Pointer operations:
    fx.inttoptr(T.ptr, int_val) — cast integer to pointer
    fx.ptrtoint(T.i64, ptr_val) — cast pointer to integer
    fx.ptr_load(ptr, index) — load from raw pointer
    fx.ptr_store(ptr, val, index) — store to raw pointer
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc — buffer resource pointer

  Constexpr parameters:
    from flydsl.expr.range_constexpr import range_constexpr
    from flydsl.expr.const_expr import Constexpr
    fx.Constexpr[T] — compile-time constant parameter annotation

  NONEXISTENT APIs (do NOT use): fx.index(), fx.shared_tensor(), fx.block_barrier(),
    fx.make_shared_mem(), fx.BarrierKind, flyc.create_kernel(), flyc.amdgpu_kernel

RULES:
- ABSOLUTELY FORBIDDEN: @triton.jit, import triton, __global__
- Define a build_*() or public wrapper function that the correctness test can call
- Target: speedup >= 1.0x vs baseline

WRAPPER FUNCTION RULES (CRITICAL — test harness builds inputs by matching parameter names):
- GEMM: use a, b, c, x, w, y, input, weight, output, bias, m, n, k
- Attention: use q, k, v, o, out, output, softmax_scale, sm_scale
- Conv: use x, input, weight, bias, conv_state
- Norm: use x, input, weight, bias, eps
- MoE: use hidden_states, input, x, topk_weights, topk_ids, num_experts
- Do NOT use input_tensor, weight_tensor — test harness matches by exact name
"""

PROMPTS = {
    "triton": SYSTEM_PROMPT_TRITON,
    "opus": SYSTEM_PROMPT_OPUS,
    "flydsl": SYSTEM_PROMPT_FLYDSL,
}


def detect_language(system_prompt: str) -> str | None:
    """Detect kernel language from the training data system prompt text.

    Returns 'triton', 'opus', or 'flydsl' for GPU kernel samples.
    Returns None for non-kernel samples (python bugfixes, generic coding, etc.)
    which should keep their original system prompt.
    """
    sp = system_prompt.lower()
    if "flydsl" in sp:
        return "flydsl"
    if "opus" in sp:
        return "opus"
    if "hip kernel" in sp or "hip kernels" in sp or "hip c++" in sp:
        return "opus"
    if "triton" in sp:
        return "triton"
    if "gpu kernel" in sp and ("gfx950" in sp or "mi350" in sp):
        return "opus"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Align SFT system prompts with evaluation prompts"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/danyzhan/geak_sft_dataset/gfx950-thfo-v5/train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/danyzhan/geak_sft_dataset/gfx950-thfo-v6"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        raise SystemExit(1)

    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))

    stats = Counter()
    for sample in samples:
        old_prompt = sample["messages"][0]["content"]
        lang = detect_language(old_prompt)
        if lang is not None:
            sample["messages"][0]["content"] = PROMPTS[lang]
            stats[lang] += 1
        else:
            stats["kept"] += 1

    print(f"Input:  {args.input}  ({len(samples)} samples)")
    print(f"Output: {args.output_dir / 'train.jsonl'}")
    print()
    print("Replacements:")
    for lang in ("triton", "opus", "flydsl", "kept"):
        print(f"  {lang:8s}: {stats[lang]}")
    print(f"  total   : {sum(stats.values())}")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "train.jsonl"
    with open(out_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "aligned_sft_v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "total_samples": len(samples),
        "replacements": dict(stats),
        "prompt_versions": {
            "triton": "SYSTEM_PROMPT_TRITON from run_phase2_benchmark.py",
            "opus": "SYSTEM_PROMPT_OPUS from run_phase2_benchmark.py (also used for hip samples)",
            "flydsl": "SYSTEM_PROMPT_FLYDSL from run_phase2_benchmark.py",
        },
    }
    manifest_path = args.output_dir / "source_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWritten: {out_path}")
    print(f"Written: {manifest_path}")


if __name__ == "__main__":
    main()
