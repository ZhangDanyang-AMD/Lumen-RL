#!/usr/bin/env python3
"""Phase2 GEAK agent benchmark for SFT thfo-r3 models.

Runs 150 held-out tasks (50 triton / 50 opus / 50 flydsl) via GEAKStatefulTool.
Each task: model writes kernel from scratch via tool loop, then measured vs baseline.

Usage:
    python3 run_phase2_benchmark_r3.py \
        --label sft-thfo-r3-2epoch \
        --base-url http://127.0.0.1:8000/v1 \
        --workers 6 \
        --max-turns 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MULTI_TUNE = Path("/home/danyzhan/Lumen-RL/experiments/multi-tune-agent")
sys.path.insert(0, str(MULTI_TUNE / "src"))
sys.path.insert(0, str(MULTI_TUNE))

from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.geak_tool import GEAKStatefulTool, GEAKToolEnvironment
from multi_tune_agent.runtime import OpenAIModelBackend, ToolAgentLoop

HELD_OUT = Path("/home/danyzhan/agent-phase2-held-out-aiter")
CASES_YAML = HELD_OUT / "receipts" / "phase2_benchmark_cases.yaml"
BASELINES = HELD_OUT / "receipts" / "baseline_overrides.json"

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
    fx.buffer_load(buf, indices) — load from global via buffer descriptor
    fx.buffer_store(val, buf, indices) — store via buffer descriptor

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

  Constexpr parameters:
    from flydsl.expr.range_constexpr import range_constexpr
    from flydsl.expr.const_expr import Constexpr
    fx.Constexpr[T] — compile-time constant parameter annotation

  NONEXISTENT APIs (do NOT use): fx.index(), fx.shared_tensor(), fx.block_barrier(),
    fx.make_shared_mem(), fx.BarrierKind, flyc.create_kernel(), flyc.amdgpu_kernel

RULES:
- ABSOLUTELY FORBIDDEN: @triton.jit, import triton, __global__
- ABSOLUTELY FORBIDDEN: import aiter, from aiter import ..., import mori, from mori import ...
  Your kernel MUST be standalone. Only import from: torch, flydsl.compiler, flydsl.expr, flydsl.expr.typing, flydsl.expr.rocdl, flydsl._mlir.dialects
  Do NOT copy code from aiter source. Write self-contained kernels using only the FlyDSL API above.
- Define a build_*() or public wrapper function that the correctness test can call
  The wrapper must accept standard torch.Tensor arguments (not fx.Pointer or fx.Tensor)
  and return torch.Tensor output. The test harness calls: output = wrapper(a, b, ...) with PyTorch tensors.
- Target: speedup >= 1.0x vs baseline

CORRECTNESS RULES (your kernel WILL be rejected if it violates these):
- The correctness test compares your output against a torch reference (e.g., torch.mm for GEMM,
  scaled_dot_product_attention for attention). Writing a different operator will FAIL.
- For GEMM operators: your kernel MUST iterate over the full K dimension. Using K_ITERS, K_TILES,
  or range_constexpr with small values (< 32) for partial computation will FAIL.
- Your kernel function name must reflect the actual operator. Do NOT name functions after
  activation functions (moe_relu, moe_elu, moe_tanh, etc.) unless the task IS an activation op.
- ALL operators: the test checks that output is NOT trivial. Identity copies (OUT=IN), all-zeros,
  relu(input), tanh(input), sigmoid(input), or any simple element-wise transform will FAIL.
- For MoE/sorting/routing/topk: you must implement the actual algorithm, not an approximation.
- The test verifies output shape and numerical values match the expected operator behavior.

WRAPPER FUNCTION RULES (CRITICAL — test harness builds inputs by matching parameter names):
- GEMM: use a, b, c, x, w, y, input, weight, output, bias, m, n, k
- Attention: use q, k, v, o, out, output, softmax_scale, sm_scale
- Conv: use x, input, weight, bias, conv_state
- Norm: use x, input, weight, bias, eps
- MoE: use hidden_states, input, x, topk_weights, topk_ids, num_experts
- Do NOT use input_tensor, weight_tensor — test harness matches by exact name
"""

OPERATOR_GUIDANCE = {
    "gemm": """\
OPERATOR: General Matrix Multiply (GEMM)
  Computes: C[M,N] = A[M,K] @ B[N,K].T  (A times B-transpose)
  Inputs: A (M×K, bf16), B (N×K, bf16)
  Output: C (M×N, bf16)
  Key: MUST iterate over the FULL K dimension (K=4096 typically). Each output element C[i,j] = sum(A[i,:] * B[j,:]).
  FlyDSL pattern: Use tiled approach — BLOCK_M×BLOCK_N output tile, accumulate in fp32 over K tiles.
    acc = fx.Float32(0.0)
    for ki in fx.range_constexpr(K // TILE_K):   # MUST cover full K
        a_val = fx.Float32(A[row, ki * TILE_K + lane])
        b_val = fx.Float32(B[col, ki * TILE_K + lane])
        acc += a_val * b_val  # or use fx.gemm for tiled MMA
  Reference check: torch.mm(A.float(), B.float().T)""",

    "splitk_gemm": """\
OPERATOR: Split-K GEMM
  Computes: C[M,N] = A[M,K] @ B[N,K].T, with K dimension split across multiple workgroups
  Inputs: A (M×K, bf16), B (N×K, bf16)
  Output: C (M×N, bf16)
  Key: Same as GEMM but K is partitioned across workgroups. Each split computes partial sum, then atomically reduces.
  Reference check: torch.mm(A.float(), B.float().T)""",

    "preshuffle_gemm": """\
OPERATOR: Preshuffle GEMM (FP8 with preshuffled scales)
  Computes: C[M,N] = dequant(A[M,K]) @ dequant(B[N,K]).T
  Inputs: A (M×K, fp8/int8), B (N×K, fp8/int8), scales
  Output: C (M×N, bf16)
  Key: Full K-dimension iteration with dequantization. Uses MFMA (matrix fused multiply-add) instructions.
  FlyDSL: Use fx.gemm with tiled MMA, buffer_ops for global loads.
  Reference check: torch.mm(A.float(), B.float().T)""",

    "gemm_a8w8": """\
OPERATOR: INT8 Quantized GEMM (A8W8)
  Computes: C[M,N] = (A_int8[M,K] * a_scale) @ (B_int8[N,K] * b_scale).T
  Inputs: A (M×K, int8), B (N×K, int8), a_scale (M×K//128, fp32), b_scale (N×K//128, fp32)
  Output: C (M×N, bf16)
  Key: Must dequantize (multiply by block scales) and iterate full K dimension.
  Reference check: (A.float() * a_scale) @ (B.float() * b_scale).T""",

    "mxfp4_gemm": """\
OPERATOR: MXFP4 GEMM (4-bit microscaled floating point)
  Computes: C[M,N] = dequant_mxfp4(A[M,K]) @ dequant_mxfp4(B[N,K]).T
  Inputs: A (M×K, mxfp4 packed), B (N×K, mxfp4 packed), E8M0 scales
  Output: C (M×N, bf16)
  Key: 4-bit values with E8M0 microscales per 32-element group. Full K iteration required.
  FlyDSL: Use MFMA fp4 instructions (rocdl.cvt_scalef32_pk_fp4_f32 for conversion).
  Reference check: torch.mm after dequantization""",

    "moe": """\
OPERATOR: Mixture-of-Experts (MoE)
  Computes: For each token, route to top-k experts, run expert GEMM, combine results.
  Typical MoE pipeline: gate → topk → sort → grouped_gemm → reduce
  This kernel may implement a specific stage. Read the initial kernel.py docstring carefully.
  Inputs: hidden_states (M×K, bf16), topk_weights (M×2, fp32), topk_ids (M×2, int32), expert weights
  Output: depends on stage — typically (M×K, bf16) or (M×N, bf16)
  Key: This is NOT an element-wise operation. MoE involves routing tokens to experts and computing GEMMs.
  Do NOT write relu/tanh/elu/sigmoid — those are activation functions, not MoE operations.""",

    "moe_grouped_gemm": """\
OPERATOR: MoE Grouped GEMM
  Computes: For each expert e, compute Y_e = X_e @ W_e.T where X_e are the tokens routed to expert e.
  This is a batched/grouped matrix multiply where different groups have different sizes.
  Inputs: A (total_tokens×K, bf16), B (num_experts×N×K, bf16), group sizes
  Output: C (total_tokens×N, bf16)
  Key: MUST iterate over FULL K dimension for each expert's GEMM. This is a matrix multiply, not element-wise.
  FlyDSL pattern: Persistent GEMM with expert-aware tile scheduling.
  Reference check: torch.mm per expert group""",

    "moe_sorting": """\
OPERATOR: MoE Token Sorting (counting sort)
  Computes: Reorders tokens by expert assignment for efficient batched expert GEMM.
  Algorithm: counting sort — histogram expert counts → prefix sum → scatter tokens to grouped positions.
  Inputs: hidden_states (M×K, bf16), topk_ids (M×topk, int32), topk_weights (M×topk, fp32)
  Output: sorted hidden_states (M×K, bf16), sorted indices, expert offsets
  Key: This is a SORTING/PERMUTATION operation, NOT element-wise. Must actually reorder tokens.
  The output should be a permutation of the input tokens, grouped by expert.
  FlyDSL pattern: LDS-based histogram → prefix sum → scatter.""",

    "moe_routing": """\
OPERATOR: MoE Routing (route map construction)
  Computes: Maps route indices to grouped row positions via per-expert atomic counters.
  Creates topids_to_rows and rows_to_tokens mappings for the grouped GEMM scheduler.
  Inputs: topk_ids (M×topk, int32), num_experts, expert counts
  Output: route-to-row mapping arrays (int32)
  Key: This is an INDEX MAPPING operation using atomic operations, NOT element-wise math.
  Each route i gets: expert = topk_ids[i], slot = atomicAdd(counter[expert], 1), row = base[expert] + slot.""",

    "moe_reduce": """\
OPERATOR: MoE Top-K Reduction
  Computes: Y[t,d] = sum_k( X[t,k,d] * weight[t,k] ) — weighted sum over top-k expert outputs per token.
  Inputs: X (tokens×topk×dim, bf16), topk_weights (tokens×topk, fp32)
  Output: Y (tokens×dim, bf16)
  Key: Reduction over the topk dimension with gating weights. NOT identity copy.""",

    "attention": """\
OPERATOR: Fused Attention (compress/boundary kernel)
  Computes: Online-softmax attention with RMSNorm and RoPE, specific to V4 architecture.
  Read the initial kernel.py docstring for the exact computation.
  Inputs: Q, K, V tensors, position info, norms
  Output: attention output
  Key: Must implement softmax(Q@K.T/sqrt(d)) @ V with the fused operations described.
  Reference check: torch.nn.functional.scaled_dot_product_attention""",

    "fmha": """\
OPERATOR: Flash Multi-Head Attention (forward)
  Computes: O = softmax(Q @ K.T / sqrt(d_k)) @ V, using tiled/flash algorithm
  Inputs: Q (B×H×S×D, bf16), K (B×HK×S×D, bf16), V (B×HK×S×D, bf16)
  Output: O (B×H×S×D, bf16)
  Key: Tiled attention — GEMM1 (QK), online softmax, GEMM2 (PV). Two matrix multiplies.
  Reference check: F.scaled_dot_product_attention(Q, K, V)""",

    "mla": """\
OPERATOR: Multi-Latent Attention decode reduce
  Computes: Merges per-split partial outputs via online softmax — O = sum(exp(LSE_i - LSE_max) * O_i).
  Inputs: partial_output (fp32), partial_lse (fp32), reduce indices
  Output: final output (bf16)
  Key: Weighted combination using softmax of LSE values, NOT element-wise copy.""",

    "causal_conv1d": """\
OPERATOR: Causal 1D Convolution
  Computes: y[n] = sum_{k=0}^{K-1} x[n-k] * w[k] with causal (left-padding) semantics
  Inputs: x (B×C×L, bf16), weight (C×1×K, bf16), optional bias, conv_state
  Output: y (B×C×L, bf16) with optional split q/k/v
  Key: Sliding window convolution — each output depends on K preceding input values.
  Reference check: torch.nn.functional.conv1d with left padding""",

    "chunk_gated_delta": """\
OPERATOR: Gated Delta Net Hidden-State Recurrence
  Computes: Serially over chunks: h = h * exp(g_last) + k^T @ (v_new * exp(g_last - g_cumsum))
  This is a recurrence with matrix operations (k^T @ v_new involves MFMA).
  Inputs: q, k, v, gate/g, beta tensors
  Output: hidden state updates
  Key: Involves matrix multiply (MFMA) in the recurrence. NOT element-wise.""",

    "dispatch_combine": """\
OPERATOR: EP MoE Dispatch/Combine
  Computes: Intranode token dispatch (scatter) and combine (gather+reduce) for expert parallelism.
  Inputs: tokens, routing info, expert assignments
  Output: dispatched/combined token buffers
  Key: Data movement + reduction across expert parallel ranks. Complex scatter/gather.""",

    "quantization": """\
OPERATOR: MXFP4/MXFP8 Quantization
  Computes: Quantize bf16 input to FP4/FP8 with E8M0 microscales per 32-element group.
  For each group of 32 elements: compute amax → e8m0 scale → scale and clamp to target format.
  Inputs: x (M×N, bf16)
  Output: quantized values (fp4/fp8) + scales (e8m0)
  Key: Per-group scale computation and format conversion. NOT identity copy.""",

    "topk": """\
OPERATOR: Fused Gate-Activation-Mul + Quantization
  Computes: Fused SiLU activation, gating, and optional quantization for MoE stage1 post-processing.
  Inputs: tmp_out (tokens×topk, inter_dim×2, bf16), routing info
  Output: activated + quantized output (fp4/fp8/bf16)
  Key: SiLU(x) * gate pattern with quantization. Read initial kernel.py for details.""",
}


def get_system_prompt(language: str) -> str:
    if language == "opus":
        return SYSTEM_PROMPT_OPUS
    if language == "flydsl":
        return SYSTEM_PROMPT_FLYDSL
    return SYSTEM_PROMPT_TRITON


def run_single_task(
    task_id: str,
    config: MultiTuneConfig,
    baseline: dict,
    max_turns: int,
    gpu_id: str,
) -> dict:
    """Run one task with a fresh OpenAIModelBackend (thread-safe)."""
    start = time.monotonic()

    # Each thread gets its own backend to avoid shared session issues
    backend = OpenAIModelBackend(
        base_url=config.base_url,
        model=config.model,
        timeout=config.request_timeout,
    )

    environment = GEAKToolEnvironment(config)
    tool = GEAKStatefulTool(environment)

    # Get language and operator for system prompt
    language = "triton"
    operator = "unknown"
    kernel_prompt = ""
    task_dir = HELD_OUT / "artifacts" / "kernel" / task_id / "initial"
    cfg_path = task_dir / "config.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        language = cfg.get("language", "triton")
        operator = cfg.get("operator", "unknown")
        shapes = cfg.get("benchmark_shapes", [])
        if shapes and isinstance(shapes[0], dict):
            kernel_prompt = shapes[0].get("kernel_prompt", "")

    system_prompt = get_system_prompt(language)

    # Baseline override
    per_case_ms = baseline.get("per_case_ms", {"default": 1.0})
    geomean_ms = baseline.get("geomean_ms", 1.0)

    baseline_override = {
        "per_case_ms": per_case_ms,
        "geomean_ms": geomean_ms,
    }

    # ToolAgentLoop: messages includes system prompt + task-specific user message
    import asyncio

    task_desc = f"You must write a **{operator}** kernel in {language} for AMD gfx950."
    if kernel_prompt:
        task_desc += f"\n{kernel_prompt}"

    guidance = OPERATOR_GUIDANCE.get(operator, "")
    if guidance:
        task_desc += f"\n\n{guidance}"

    task_desc += (
        f"\n\nDo NOT write a different operator — the correctness test compares your output "
        f"against a torch reference (e.g., torch.mm for GEMM). Writing an activation function "
        f"(relu, tanh, elu, etc.) instead of {operator} will FAIL correctness."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
        {"role": "assistant", "content": f"I will write a {operator} kernel in {language}. Let me start by reading the task runner and config to understand the required interface."},
    ]
    create_kwargs = {
        "case_id":            task_id,
        "role":               "engineer",
        "establish_baseline": False,
        "baseline_override":  baseline_override,
        "clear_kernel":       True,
    }

    loop = ToolAgentLoop(
        backend=backend,
        tool=tool,
        max_assistant_turns=max_turns,
        retain_session=True,
    )

    try:
        output = asyncio.run(loop.run(messages, create_kwargs=create_kwargs))
    except Exception as e:
        return {
            "task_id":      task_id,
            "language":     language,
            "error":        f"loop_failed: {e}",
            "compiled":     False,
            "correct":      False,
            "best_speedup": 0.0,
            "turns_used":   0,
            "tool_calls":   0,
        }

    # Extract result from output + session state
    sid = output.session_id if output else ""
    last_eval = {}
    best_speedup = 0.0
    compiled = False
    correct = False

    try:
        state = environment.get(sid)
        best_speedup = state.best_speedup if state.best_speedup > 0 else 0.0
        last_eval = state.last_evaluation or {}
        compiled = last_eval.get("compiled", False)
        correct  = last_eval.get("correct",  False)
        # If model ever achieved correct + speedup, report that even if last eval regressed
        if best_speedup > 0:
            compiled = True
            correct = True
    except Exception:
        pass

    result = {
        "task_id":           task_id,
        "language":          language,
        "compiled":          compiled,
        "correct":           correct,
        "best_speedup":      best_speedup,
        "best_candidate_ms": last_eval.get("candidate_ms", {}),
        "turns_used":        output.metrics.assistant_turns if output and output.metrics else 0,
        "tool_calls":        output.metrics.tool_calls      if output and output.metrics else 0,
        "model_seconds":     output.metrics.model_seconds   if output and output.metrics else 0.0,
        "elapsed_s":         time.monotonic() - start,
        "session_id":        sid,
        "reward_score":      output.reward_score if output else 0.0,
    }

    # Release session
    if sid:
        try:
            tool.release(sid)
        except Exception:
            pass
    return result


def get_baseline_for_task(task_id: str, baselines: dict) -> dict:
    """Get baseline timing for a task, falling back to 1.0ms."""
    if task_id in baselines:
        return baselines[task_id]
    return {"per_case_ms": {"default": 1.0}, "geomean_ms": 1.0}


def _apply_artifact_patches() -> None:
    """Idempotent patches on all 150 artifact task scripts before benchmark.

    1. task_runner.py — remove opus correctness/perf shortcut
    2. op_harness.py — add missing operator aliases
    3. op_harness.py — warmup=360, run=50, independent profiler contexts, median
    4. correctness_runner.py — deploy FlyDSL build_* aware version
    """
    artifact_root = HELD_OUT / "artifacts" / "kernel"
    if not artifact_root.is_dir():
        return

    # ── Replacement strings for task_runner.py ──
    # Opus correctness: v0 (original) and v1 (compile-only) both get replaced
    _OLD_OPUS_CORRECT_V0 = (
        '    # Opus shortcut if no Python kernel\n'
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        print("PASS: correctness (opus kernel present)"); return 0'
    )
    _OLD_OPUS_CORRECT_V1 = (
        '    # Opus: must compile successfully (no free pass)\n'
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        include_dir = task_dir / "include"\n'
        '        inc_flags = ["-I", str(include_dir)] if include_dir.is_dir() else []\n'
        '        r = subprocess.run(\n'
        '            ["hipcc", "--genco", "-O3", "--offload-arch=gfx950"] + inc_flags +\n'
        '            [str(kernel_cu), "-o", "/dev/null"],\n'
        '            capture_output=True, text=True, timeout=120)\n'
        '        if r.returncode != 0:\n'
        '            print(f"FAIL: opus kernel compile error\\n{r.stderr[:500]}")\n'
        '            return 1\n'
        '        print("PASS: correctness (opus kernel compiles)")\n'
        '        return 0'
    )
    _NEW_OPUS_CORRECT = (
        '    # Opus: real compile+load+call via opus_runner.py\n'
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        runner = task_dir / "scripts" / "opus_runner.py"\n'
        '        if runner.exists():\n'
        '            r = subprocess.run(\n'
        '                [sys.executable, str(runner), str(task_dir), "correctness"],\n'
        '                capture_output=True, text=True, timeout=180)\n'
        '            output = (r.stdout + r.stderr).strip()\n'
        '            for line in output.splitlines():\n'
        '                if line.strip(): print(line)\n'
        '            return r.returncode\n'
        '        print("FAIL: opus_runner.py missing"); return 1'
    )
    # Opus perf: v0 and v1 both get replaced
    _OLD_OPUS_PERF_V0 = (
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        print("Perf: 1.000 ms (opus_placeholder)"); return 0'
    )
    _OLD_OPUS_PERF_V1 = (
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        print("Perf: 0.0 ms (opus, no python wrapper)")\n'
        '        return 0'
    )
    _NEW_OPUS_PERF = (
        '    # Opus: real benchmark via opus_runner.py\n'
        '    if kernel_cu.exists() and not kernel_py.exists():\n'
        '        runner = task_dir / "scripts" / "opus_runner.py"\n'
        '        if runner.exists():\n'
        '            r = subprocess.run(\n'
        '                [sys.executable, str(runner), str(task_dir), "performance"],\n'
        '                capture_output=True, text=True, timeout=300)\n'
        '            output = (r.stdout + r.stderr).strip()\n'
        '            for line in output.splitlines():\n'
        '                if line.strip(): print(line)\n'
        '            return r.returncode\n'
        '        print("Perf: 0.0 ms (opus_runner_missing)"); return 0'
    )

    # ── Replacement strings for op_harness.py profiler ──
    _OLD_DEFAULTS = "    n_warmup: int = 5\n    n_iter: int = 20"
    _NEW_DEFAULTS = "    n_warmup: int = 360\n    n_iter: int = 50"
    _OLD_PROFILE = (
        'def _profile(func: Callable, args: list, n_warmup: int = 5, n_iter: int = 20) -> float:\n'
        '    """Return total GPU kernel time per call in ms via torch.profiler."""\n'
        '    try:\n'
        '        for _ in range(n_warmup):\n'
        '            func(*args)\n'
        '        torch.cuda.synchronize()\n'
        '    except Exception:\n'
        '        return 0.0\n'
        '\n'
        '    try:\n'
        '        with torch.profiler.profile(\n'
        '            activities=[torch.profiler.ProfilerActivity.CUDA],\n'
        '            record_shapes=False,\n'
        '            with_stack=False,\n'
        '        ) as prof:\n'
        '            for _ in range(n_iter):\n'
        '                func(*args)\n'
        '            torch.cuda.synchronize()\n'
        '    except Exception:\n'
        '        return 0.0\n'
        '\n'
        '    events = [e for e in prof.key_averages() if e.device_time_total > 0]\n'
        '    if not events:\n'
        '        return 0.0\n'
        '    total_us = sum(e.device_time_total for e in events)\n'
        '    return total_us / 1000.0 / n_iter'
    )
    _NEW_PROFILE = (
        'def _profile(func: Callable, args: list, n_warmup: int = 360, n_iter: int = 50) -> float:\n'
        '    """Return median GPU kernel time per call in ms via torch.profiler.\n'
        '\n'
        '    Each of the n_iter iterations runs in its own profiler context to get\n'
        '    independent device-time measurements.  The median is reported.\n'
        '    """\n'
        '    try:\n'
        '        for _ in range(n_warmup):\n'
        '            func(*args)\n'
        '        torch.cuda.synchronize()\n'
        '    except Exception:\n'
        '        return 0.0\n'
        '\n'
        '    times_ms: list[float] = []\n'
        '    for _ in range(n_iter):\n'
        '        try:\n'
        '            with torch.profiler.profile(\n'
        '                activities=[torch.profiler.ProfilerActivity.CUDA],\n'
        '                record_shapes=False,\n'
        '                with_stack=False,\n'
        '            ) as prof:\n'
        '                func(*args)\n'
        '                torch.cuda.synchronize()\n'
        '        except Exception:\n'
        '            continue\n'
        '        events = [e for e in prof.key_averages() if e.device_time_total > 0]\n'
        '        if events:\n'
        '            total_us = sum(e.device_time_total for e in events)\n'
        '            times_ms.append(total_us / 1000.0)\n'
        '\n'
        '    if not times_ms:\n'
        '        return 0.0\n'
        '    times_ms.sort()\n'
        '    mid = len(times_ms) // 2\n'
        '    return times_ms[mid]'
    )

    # ── op_harness.py alias block (inserted before fallback) ──
    _ALIAS_MARKER = "causal_conv1d"
    _ALIAS_BLOCK = '''
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

    # ── Correctness runner v3 source (reference-based validation) ──
    cr_v3 = MULTI_TUNE / "scripts" / "correctness_runner_v3.py"
    cr_v2 = MULTI_TUNE / "scripts" / "correctness_runner_v2.py"
    cr_src = cr_v3 if cr_v3.exists() else cr_v2
    cr_v2_content = cr_src.read_text() if cr_src.exists() else None

    # ── Compile check source (real JIT compile, not just import/ast.parse) ──
    cc_src = MULTI_TUNE / "scripts" / "compile_check.py"
    cc_content = cc_src.read_text() if cc_src.exists() else None

    # ── Opus runner source (real compile→load→call→verify→time) ──
    or_src = MULTI_TUNE / "scripts" / "opus_runner.py"
    or_content = or_src.read_text() if or_src.exists() else None

    # ── OLD run_compile() bodies for all 3 language variants ──
    # They share the same structure, only the language literal and kernel ext differ.
    _NEW_RUN_COMPILE = (
        'def run_compile(task_dir):\n'
        '    """Check kernel compiles by actually calling the wrapper function."""\n'
        '    r = subprocess.run(\n'
        '        [sys.executable, str(task_dir / "scripts" / "compile_check.py"), str(task_dir)],\n'
        '        capture_output=True, text=True, timeout=120)\n'
        '    output = (r.stdout + r.stderr).strip()\n'
        '    for line in output.splitlines():\n'
        '        if line.strip():\n'
        '            print(line)\n'
        '    return r.returncode'
    )

    def _build_old_run_compile(lang: str) -> str:
        ext = "kernel.cu" if lang == "opus" else "kernel.py"
        return (
            'def run_compile(task_dir):\n'
            '    """Check kernel compiles."""\n'
            f'    kernel = task_dir / "{ext}"\n'
            '    if not kernel.exists():\n'
            '        print("FAIL: kernel file missing"); return 1\n'
            f'    if "{lang}" == "triton":\n'
            '        r = subprocess.run([sys.executable, "-c",\n'
            '                              f"import importlib.util, sys; "\n'
            '                              f"spec=importlib.util.spec_from_file_location(\'kernel\',\'{kernel}\'); "\n'
            '                              f"mod=importlib.util.module_from_spec(spec); "\n'
            '                              f"spec.loader.exec_module(mod)"],\n'
            '                          capture_output=True, text=True, timeout=60)\n'
            f'    elif "{lang}" == "opus":\n'
            '        include_dir = task_dir / "include"\n'
            '        inc_flags = ["-I", str(include_dir)] if include_dir.is_dir() else []\n'
            '        r = subprocess.run(["hipcc", "--genco", "-O3", "--offload-arch=gfx950"] + inc_flags + [str(kernel), "-o", "/dev/null"],\n'
            '                          capture_output=True, text=True, timeout=120)\n'
            f'    elif "{lang}" == "flydsl":\n'
            '        r = subprocess.run([sys.executable, "-c", f"import ast; ast.parse(open(\'{kernel}\').read())"],\n'
            '                          capture_output=True, text=True, timeout=30)\n'
            '    else:\n'
            '        print("FAIL: unknown language"); return 1\n'
            '    if r.returncode != 0:\n'
            '        print(f"FAIL: {r.stderr[:500]}"); return 1\n'
            '    print("PASS: compile"); return 0'
        )

    _OLD_RUN_COMPILE_TRITON = _build_old_run_compile("triton")
    _OLD_RUN_COMPILE_FLYDSL = _build_old_run_compile("flydsl")
    _OLD_RUN_COMPILE_OPUS = _build_old_run_compile("opus")

    # ── Apply all patches ──
    counts = {"task_runner": 0, "harness_alias": 0, "harness_profiler": 0, "correctness": 0,
              "compile_check": 0, "run_compile": 0, "opus_runner": 0}

    for task_dir in sorted(artifact_root.iterdir()):
        scripts = task_dir / "initial" / "scripts"
        if not scripts.is_dir():
            continue

        # 1. task_runner.py — opus correctness/perf via opus_runner.py
        tr = scripts / "task_runner.py"
        if tr.exists():
            c = tr.read_text()
            orig = c
            c = c.replace(_OLD_OPUS_CORRECT_V0, _NEW_OPUS_CORRECT)
            c = c.replace(_OLD_OPUS_CORRECT_V1, _NEW_OPUS_CORRECT)
            c = c.replace(_OLD_OPUS_PERF_V0, _NEW_OPUS_PERF)
            c = c.replace(_OLD_OPUS_PERF_V1, _NEW_OPUS_PERF)
            if c != orig:
                tr.write_text(c)
                counts["task_runner"] += 1

        # 2+3. op_harness.py — aliases + profiler + moe-family registration
        oh = scripts / "op_harness.py"
        if oh.exists():
            c = oh.read_text()
            orig = c
            if _ALIAS_MARKER not in c and "# ── Generic fallback" in c:
                c = c.replace("# ── Generic fallback", _ALIAS_BLOCK + "\n# ── Generic fallback")
                counts["harness_alias"] += 1
            if _OLD_DEFAULTS in c:
                c = c.replace(_OLD_DEFAULTS, _NEW_DEFAULTS)
            if _OLD_PROFILE in c:
                c = c.replace(_OLD_PROFILE, _NEW_PROFILE)
                counts["harness_profiler"] += 1
            _OLD_MOE_REG = '_reg(["moe", "moe_sorting", "topk"], OpHarness('
            _NEW_MOE_REG = ('_reg(["moe", "moe_sorting", "topk", "moe_routing",'
                            ' "moe_grouped_gemm",\n'
                            '      "moe_reduce", "dispatch_combine"], OpHarness(')
            if _OLD_MOE_REG in c:
                c = c.replace(_OLD_MOE_REG, _NEW_MOE_REG)
                counts["harness_alias"] += 1
            if c != orig:
                oh.write_text(c)

        # 4. correctness_runner.py — reference-based validation (v3 with structural checks)
        if cr_v2_content:
            cr = scripts / "correctness_runner.py"
            if not cr.exists() or cr.read_text() != cr_v2_content:
                cr.write_text(cr_v2_content)
                counts["correctness"] += 1

        # 5. compile_check.py — deploy real compile checker
        if cc_content:
            cc_dst = scripts / "compile_check.py"
            if not cc_dst.exists() or cc_dst.read_text() != cc_content:
                cc_dst.write_text(cc_content)
                counts["compile_check"] += 1

        # 6. task_runner.py — replace fake run_compile() with compile_check.py call
        if tr.exists():
            c = tr.read_text()
            if "compile_check.py" not in c:
                orig_c = c
                for old_body in (_OLD_RUN_COMPILE_TRITON, _OLD_RUN_COMPILE_FLYDSL, _OLD_RUN_COMPILE_OPUS):
                    c = c.replace(old_body, _NEW_RUN_COMPILE)
                if c != orig_c:
                    tr.write_text(c)
                    counts["run_compile"] += 1

        # 7. opus_runner.py — deploy opus compile+load+call+verify+time
        if or_content:
            or_dst = scripts / "opus_runner.py"
            if not or_dst.exists() or or_dst.read_text() != or_content:
                or_dst.write_text(or_content)
                counts["opus_runner"] += 1

    parts = []
    for k, v in counts.items():
        if v:
            parts.append(f"{k}={v}")
    if parts:
        print(f"  [setup] Artifact patches: {', '.join(parts)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",    default="sft-thfo-r3-2epoch")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model",    default="model")
    parser.add_argument("--workers",  type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--gpu-ids",  default="2,3,4,5,6,7")
    parser.add_argument("--lang",     default=None, help="triton|opus|flydsl or None for all")
    parser.add_argument("--task",     default=None, help="run single task id")
    parser.add_argument("--docker-container", default="", help="Docker container for eval (e.g. aiter-mi400-regress)")
    args = parser.parse_args()

    # Load cases and baselines
    import yaml
    cases_data = yaml.safe_load(CASES_YAML.read_text())
    all_tasks = cases_data.get("tasks", [])

    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}

    # Filter tasks
    tasks = all_tasks
    if args.lang:
        # Filter by language from config.yaml
        filtered = []
        for t in tasks:
            cfg = (HELD_OUT / "artifacts" / "kernel" / t["id"] / "initial" / "config.yaml")
            if cfg.exists():
                cfg_data = yaml.safe_load(cfg.read_text())
                if cfg_data.get("language") == args.lang:
                    filtered.append(t)
        tasks = filtered
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]

    # ── Apply artifact patches (idempotent) before benchmark starts ────────
    _apply_artifact_patches()

    n = len(tasks)
    print(f"Phase2 GEAK benchmark: {args.label}")
    print(f"  Model: {args.base_url} ({args.model})")
    print(f"  Tasks: {n} | Workers: {args.workers} | Max turns: {args.max_turns}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print()

    # Config
    config = MultiTuneConfig(
        geak_root=Path("/home/danyzhan/GEAK"),
        cases_path=CASES_YAML,
        trajectory_root=HELD_OUT / "receipts" / f"geak-flow-{args.label}",
        base_url=args.base_url,
        model=args.model,
        gpu_ids=args.gpu_ids,
        request_timeout=600,
        request_max_tokens=4096,
        command_timeout=300,
        baseline_repeats=1,
        max_rounds=1,
        engineers_per_round=1,
        engineer_tool_rounds=args.max_turns,
        integrator_tool_rounds=10,
        candidate_floor=0.8,
        min_improvement=0.01,
        target_speedup=1.1,
        keep_sessions=True,
        bootstrap_enabled=False,
        bootstrap_auto_promote=False,
        aiter_root=Path("/home/danyzhan/aiter"),
        docker_container=args.docker_container,
        sft_enabled=False,
    )

    gpu_ids = args.gpu_ids.split(",")
    results = []
    start_all = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_task = {
            pool.submit(
                run_single_task,
                t["id"],
                config,
                get_baseline_for_task(t["id"], baselines),
                args.max_turns,
                gpu_ids[i % len(gpu_ids)],
            ): t
            for i, t in enumerate(tasks)
        }
        for i, f in enumerate(as_completed(future_to_task), 1):
            t = future_to_task[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"task_id": t["id"], "error": str(e), "compiled": False,
                     "correct": False, "best_speedup": 0.0, "turns_used": 0, "tool_calls": 0}
            results.append(r)
            sp = r.get("best_speedup", 0.0)
            compiled = r.get("compiled", False)
            correct = r.get("correct", False)
            turns = r.get("turns_used", 0)
            tools = r.get("tool_calls", 0)
            lang = r.get("language", "?")
            print(
                f"  [{i:3d}/{n}] [{lang:6s}] {r['task_id'][:45]:<45} "
                f"compile={compiled} correct={correct} speedup={sp:.2f}x "
                f"turns={turns} tools={tools}",
                flush=True,
            )

    elapsed = time.monotonic() - start_all

    # Summary
    n_compiled = sum(1 for r in results if r.get("compiled"))
    n_correct  = sum(1 for r in results if r.get("correct"))
    speedups   = [r.get("best_speedup", 0.0) for r in results if r.get("best_speedup", 0) > 0]
    fast09     = sum(1 for r in results if r.get("best_speedup", 0) >= 0.9)
    fast10     = sum(1 for r in results if r.get("best_speedup", 0) >= 1.0)
    fast11     = sum(1 for r in results if r.get("best_speedup", 0) >= 1.1)
    fast15     = sum(1 for r in results if r.get("best_speedup", 0) >= 1.5)
    geomean    = math.exp(sum(math.log(max(s, 1e-6)) for s in speedups) / len(speedups)) if speedups else 0.0

    # Per-language breakdown
    by_lang = defaultdict(lambda: {"total":0,"compiled":0,"correct":0,"fast10":0,"speedups":[]})
    for r in results:
        lang = r.get("language","?")
        by_lang[lang]["total"] += 1
        if r.get("compiled"): by_lang[lang]["compiled"] += 1
        if r.get("correct"):  by_lang[lang]["correct"]  += 1
        if r.get("best_speedup",0) >= 1.0: by_lang[lang]["fast10"] += 1
        if r.get("best_speedup",0) > 0: by_lang[lang]["speedups"].append(r["best_speedup"])

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE: {args.label}")
    print(f"{'='*70}")
    print(f"  Total tasks:    {n}")
    print(f"  Compiled:       {n_compiled}/{n} ({100*n_compiled//max(n,1)}%)")
    print(f"  Correct:        {n_correct}/{n} ({100*n_correct//max(n,1)}%)")
    print(f"  Speedup >=0.9x: {fast09}/{n}")
    print(f"  Speedup >=1.0x: {fast10}/{n}")
    print(f"  Speedup >=1.1x: {fast11}/{n}")
    print(f"  Speedup >=1.5x: {fast15}/{n}")
    print(f"  Geomean speedup: {geomean:.3f}x")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    print(f"\n  By language:")
    for lang, s in sorted(by_lang.items()):
        t = s["total"]
        sp_geo = math.exp(sum(math.log(max(v,1e-6)) for v in s["speedups"])/len(s["speedups"])) if s["speedups"] else 0
        print(f"    {lang:8s}: compile={s['compiled']}/{t} correct={s['correct']}/{t} "
              f"fast10={s['fast10']}/{t} geomean={sp_geo:.3f}x")

    # Save
    out_dir = HELD_OUT / "receipts" / f"geak-flow-{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "label": args.label, "model": args.model, "base_url": args.base_url,
        "total": n, "compiled": n_compiled, "correct": n_correct,
        "fast09": fast09, "fast10": fast10, "fast11": fast11, "fast15": fast15,
        "speedup_geomean": round(geomean, 4),
        "elapsed_s": round(elapsed, 1),
        "by_language": {
            lang: {
                "total": s["total"], "compiled": s["compiled"], "correct": s["correct"],
                "fast10": s["fast10"],
                "geomean": round(math.exp(sum(math.log(max(v,1e-6)) for v in s["speedups"])/len(s["speedups"])),4) if s["speedups"] else 0,
            }
            for lang, s in by_lang.items()
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out_dir}")


if __name__ == "__main__":
    main()
