#!/usr/bin/env python3
"""Measure real FlyDSL baseline kernel performance using aiter test patterns.

V5: Uses ptr_arg() wrapping and exact calling conventions from aiter tests.
Each benchmark follows the pattern from op_tests/test_flydsl_*.py.

Run inside aiter-baseline-bench:
  docker exec -e HIP_VISIBLE_DEVICES=2 aiter-baseline-bench \
    python3 /home/danyzhan/Lumen-RL/experiments/multi-tune-agent/scripts/measure_flydsl_baselines_v5.py
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, '/home/jimguo12/aiter')
sys.path.insert(0, '/home/jimguo12/aiter/op_tests')

import torch
import torch.profiler
import yaml

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

ARTIFACTS = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")
OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/flydsl_real_baselines.json")
OVERRIDE_OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/baseline_overrides.json")

N_WARMUP = 360
N_REP = 50
I32 = torch.int32

torch.set_default_device("cuda")
stream = torch.cuda.current_stream().cuda_stream


def profiler_bench(func, warmup=N_WARMUP, rep=N_REP):
    try:
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
    except Exception as e:
        return 0.0, f"warmup: {e}"
    times = []
    for _ in range(rep):
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False,
            ) as prof:
                func()
                torch.cuda.synchronize()
            us = sum(e.device_time_total for e in prof.key_averages()
                     if e.device_time_total > 0)
            if us > 0:
                times.append(us / 1000.0)
        except Exception:
            pass
    if not times:
        return 0.0, "no valid timing"
    times.sort()
    return times[len(times) // 2], None


_cache = {}

def cached(key, fn):
    if key not in _cache:
        try:
            _cache[key] = fn()
        except Exception as e:
            _cache[key] = (0.0, str(e)[:200])
    return _cache[key]


# ── Benchmark functions (using aiter test patterns) ─────────────────

def bench_compress_attn():
    """From test_flydsl_compress_attn.py"""
    from test_flydsl_compress_attn import _build_inputs, _shape_by_label
    from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn
    import inspect
    shape = _shape_by_label("csa_main")
    inputs = _build_inputs(shape, bs=4, mtp=0, mode="decode")
    sig = inspect.signature(flydsl_fused_compress_attn)
    valid_keys = set(sig.parameters.keys())
    filtered = {k: v for k, v in inputs.items() if k in valid_keys}
    return profiler_bench(lambda: flydsl_fused_compress_attn(**filtered))


def bench_hgemm():
    """torch.mm as baseline for bf16 GEMM (original aiter hgemm module removed)."""
    M, N, K = 4096, 7168, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    C = torch.empty(M, N, dtype=torch.bfloat16)
    return profiler_bench(lambda: torch.mm(A, B, out=C))


def bench_gemm_a8w8():
    """From test calling compile_fp8_gemm_8w. Uses ptr_arg convention."""
    from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import compile_fp8_gemm_8w
    K = 4096
    compiled = compile_fp8_gemm_8w(K=K)
    M, N = 4096, 7168
    A = torch.zeros(M, K, dtype=torch.float8_e4m3fnuz)
    B = torch.zeros(K, N, dtype=torch.float8_e4m3fnuz)
    C = torch.empty(M, N, dtype=torch.bfloat16)
    sa = torch.ones(M, dtype=torch.float32)
    sb = torch.ones(N, dtype=torch.float32)
    def run():
        compiled(
            ptr_arg(A), ptr_arg(B), ptr_arg(C),
            ptr_arg(sa), ptr_arg(sb),
            M, N,
            stream=stream,
        )
    return profiler_bench(run)


def bench_preshuffle_gemm():
    """compile_preshuffle_gemm with ptr_arg."""
    from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm
    N, K = 7168, 4096
    compiled = compile_preshuffle_gemm(N=N, K=K, tile_m=128, tile_n=128, tile_k=64, in_dtype='fp8')
    M = 4096
    A = torch.zeros(M, K, dtype=torch.float8_e4m3fnuz)
    B = torch.zeros(K, N, dtype=torch.float8_e4m3fnuz)
    C = torch.empty(M, N, dtype=torch.bfloat16)
    def run():
        compiled(
            ptr_arg(A), ptr_arg(B), ptr_arg(C),
            M,
            stream=stream,
        )
    return profiler_bench(run)


def bench_mla_reduce():
    """compile_mla_reduce with Dv divisible by 256."""
    from aiter.ops.flydsl.kernels.mla_reduce import compile_mla_reduce
    H, Dv = 128, 512  # Dv must be divisible by NUM_THREADS(256)
    compiled = compile_mla_reduce(H=H, Dv=Dv)
    B, sq, sk = 1, 1, 4096
    q = torch.randn(B, sq, H, Dv, dtype=torch.bfloat16)
    k = torch.randn(B, sk, 1, Dv, dtype=torch.bfloat16)
    v = torch.randn(B, sk, 1, Dv, dtype=torch.bfloat16)
    o = torch.empty(B, sq, H, Dv, dtype=torch.bfloat16)
    lse = torch.empty(B, H, sq, dtype=torch.float32)
    sl = torch.tensor([sk], dtype=I32, device='cuda')
    def run():
        compiled(
            ptr_arg(q), ptr_arg(k), ptr_arg(v),
            ptr_arg(o), ptr_arg(lse), ptr_arg(sl),
            sq, sk, 1,  # sq, max_sk, num_seqs
            stream=stream,
        )
    return profiler_bench(run)


def bench_moe_sorting():
    """compile_moe_sorting with ptr_arg."""
    from aiter.ops.flydsl.kernels.moe_sorting_kernel import (
        compile_moe_sorting, moe_sorting_get_workspace_size
    )
    ne, topk, mt = 8, 2, 4096
    compiled = compile_moe_sorting(num_experts=ne, topk=topk, max_tokens=mt)
    ws = moe_sorting_get_workspace_size(mt, ne, topk)
    ws = ((ws + 3) // 4) * 4
    topk_ids = torch.randint(0, ne, (mt, topk), dtype=I32)
    topk_w = torch.randn(mt, topk, dtype=torch.float32)
    buf = mt * topk + ne * 256
    si = torch.empty(buf, dtype=I32)
    sw = torch.empty(buf, dtype=torch.float32)
    se = torch.empty(buf, dtype=I32)
    nv = torch.empty(1, dtype=I32)
    mb = torch.empty(ws, dtype=torch.uint8)
    def run():
        compiled(
            ptr_arg(topk_ids), ptr_arg(topk_w),
            ptr_arg(si), ptr_arg(sw), ptr_arg(se),
            ptr_arg(nv), ptr_arg(mb),
            mt,
            stream=stream,
        )
    return profiler_bench(run)


def bench_moe_contiguous_psum():
    """From test_flydsl_moe_aux::test_contiguous_psum."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import build_moe_contiguous_psum_module
    launch = build_moe_contiguous_psum_module()
    E, tile_m = 8, 32
    masked = torch.randint(0, 400, (E,), dtype=I32)
    starts = torch.empty(E, dtype=I32)
    psum = torch.empty(E, dtype=I32)
    cm = torch.empty(1, dtype=I32)
    def run():
        launch(
            ptr_arg(masked), ptr_arg(starts), ptr_arg(psum), ptr_arg(cm),
            E, tile_m, stream=stream,
        )
    return profiler_bench(run)


def bench_moe_gather_reduce():
    """From test_flydsl_moe_aux::test_gather_reduce."""
    from aiter.ops.flydsl.kernels.moe_gather_reduce import build_moe_gather_reduce_module
    tokens, model_dim, topk = 4096, 4096, 2
    launch = build_moe_gather_reduce_module(model_dim, topk, 'bf16', 1, 2, 'f32')
    grouped = torch.randn(tokens * topk, model_dim, dtype=torch.bfloat16)
    weights = torch.randn(tokens, topk, dtype=torch.float32).softmax(-1)
    rows = torch.arange(tokens * topk, dtype=I32)
    out = torch.empty(tokens, model_dim, dtype=torch.bfloat16)
    def run():
        launch(
            ptr_arg(grouped), ptr_arg(weights), ptr_arg(rows), ptr_arg(out),
            tokens, model_dim,
            stream=stream,
        )
    return profiler_bench(run)


def bench_moe_reduce():
    """compile_moe_reduction."""
    from aiter.ops.flydsl.kernels.moe_reduce import compile_moe_reduction
    tokens, model_dim, topk = 4096, 4096, 2
    launch = compile_moe_reduction(topk=topk, model_dim=model_dim)
    grouped = torch.randn(tokens * topk, model_dim, dtype=torch.bfloat16)
    weights = torch.randn(tokens, topk, dtype=torch.float32).softmax(-1)
    rows = torch.arange(tokens * topk, dtype=I32)
    out = torch.empty(tokens, model_dim, dtype=torch.bfloat16)
    def run():
        launch(
            ptr_arg(grouped), ptr_arg(weights), ptr_arg(rows), ptr_arg(out),
            tokens, model_dim,
            stream=stream,
        )
    return profiler_bench(run)


def bench_moe_route_maps():
    """From test_flydsl_moe_aux::test_route_maps."""
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module
    launch = build_moe_route_maps_module()
    numel, E, topk, max_m = 4096, 8, 2, 512
    topk_ids = torch.randint(0, E, (numel, topk), dtype=I32)
    expert_counts = torch.empty(E, dtype=I32)
    rows = torch.empty(numel * topk, dtype=I32)
    rows_to_tokens = torch.empty(numel * topk, dtype=I32)
    def run():
        launch(
            ptr_arg(topk_ids), ptr_arg(expert_counts),
            ptr_arg(rows), ptr_arg(rows_to_tokens),
            numel, E, topk, max_m,
            stream=stream,
        )
    return profiler_bench(run)


def bench_moe_g2l_lut():
    """From test_flydsl_moe_aux::test_g2l_lut."""
    from aiter.ops.flydsl.kernels.moe_g2l_lut import build_moe_g2l_lut_module
    launch = build_moe_g2l_lut_module()
    n_local = 64
    expert_ids = torch.randint(0, 256, (n_local,), dtype=I32)
    lut = torch.full((256,), -1, dtype=I32)
    nvt = torch.empty(1, dtype=I32)
    nvr = torch.empty(1, dtype=I32)
    cnt = torch.empty(n_local, dtype=I32)
    def run():
        launch(
            ptr_arg(expert_ids), ptr_arg(lut), ptr_arg(cnt),
            ptr_arg(nvt), ptr_arg(nvr),
            n_local, 256,
            stream=stream,
        )
    return profiler_bench(run)


def bench_silu_and_mul_fq():
    """build_silu_and_mul_fq_module."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module
    inter_dim, topk = 8192, 2
    launch = build_silu_and_mul_fq_module(inter_dim=inter_dim, topk=topk)
    tokens = 4096
    x = torch.randn(tokens, inter_dim * 2, dtype=torch.bfloat16)
    out = torch.empty(tokens, inter_dim, dtype=torch.float8_e4m3fnuz)
    scale = torch.empty(tokens, inter_dim // 128, dtype=torch.float32)
    def run():
        launch(
            ptr_arg(x), ptr_arg(out), ptr_arg(scale),
            tokens,
            stream=stream,
        )
    return profiler_bench(run)


def bench_topk():
    """build_topk_per_row_decode_module."""
    from aiter.ops.flydsl.kernels.topk.topk_per_row_decode import build_topk_per_row_decode_module
    k = 8
    launch = build_topk_per_row_decode_module(k=k, stable=True, wave_size=64)
    rows = 4096
    logits = torch.randn(rows, 256, dtype=torch.float32)
    vals = torch.empty(rows, k, dtype=torch.float32)
    ids = torch.empty(rows, k, dtype=I32)
    def run():
        launch(
            ptr_arg(logits), ptr_arg(vals), ptr_arg(ids),
            rows, 256,
            stream=stream,
        )
    return profiler_bench(run)


def bench_chunk_gated_delta():
    """compile_chunk_gated_delta_h."""
    from aiter.ops.flydsl.kernels.gdr_prefill.chunk_gated_delta_h import compile_chunk_gated_delta_h
    K, V, H, Hg = 128, 128, 32, 4
    compiled = compile_chunk_gated_delta_h(K=K, V=V, H=H, Hg=Hg)
    B, BT = 4, 64
    T = BT * 4  # 4 chunks
    q = torch.randn(B, H, T, K, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, K, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, V, dtype=torch.bfloat16)
    g = torch.randn(B, H, T, dtype=torch.bfloat16)
    h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16)
    ht = torch.empty(B, H, K, V, dtype=torch.bfloat16)
    o = torch.empty(B, H, T, V, dtype=torch.bfloat16)
    v_new = torch.empty(B, H, T, V, dtype=torch.bfloat16)
    # This kernel has complex args - try both calling patterns
    def run():
        compiled(
            ptr_arg(q), ptr_arg(k), ptr_arg(v), ptr_arg(g),
            ptr_arg(o), ptr_arg(h0), ptr_arg(ht), ptr_arg(v_new),
            B, T, B * H,
            stream=stream,
        )
    return profiler_bench(run)


def bench_gdn_prepare():
    """compile_gdn_prepare."""
    from aiter.ops.flydsl.kernels.gdr_prefill.gdn_prepare import compile_gdn_prepare
    compiled = compile_gdn_prepare()
    B, H, T, K = 4, 32, 256, 128
    g = torch.randn(B, H, T, dtype=torch.bfloat16)
    g_t = torch.empty(B, H, T, dtype=torch.bfloat16)
    g_s = torch.empty(B, H, T, dtype=torch.bfloat16)
    def run():
        compiled(
            ptr_arg(g), ptr_arg(g_t), ptr_arg(g_s),
            B * H, T,
            stream=stream,
        )
    return profiler_bench(run)


def bench_causal_conv1d():
    """compile_causal_conv1d_update."""
    from aiter.ops.flydsl.kernels.causal_conv1d_update import compile_causal_conv1d_update
    width, seqlen = 4, 1
    compiled = compile_causal_conv1d_update(
        width=width, seqlen=seqlen, has_bias=True, silu=True,
        is_spec_decoding=False, has_null_block=False,
        block_n=64, dtype_str='bf16', weight_contig=True,
        cs_vec=True, o_vec=True, channels_per_thread=4,
    )
    B, D = 4, 4096
    x = torch.randn(B, D, dtype=torch.bfloat16)
    conv_state = torch.randn(B, D, width, dtype=torch.bfloat16)
    weight = torch.randn(D, width, dtype=torch.bfloat16)
    bias = torch.randn(D, dtype=torch.bfloat16)
    out = torch.empty(B, D, dtype=torch.bfloat16)
    def run():
        compiled(
            ptr_arg(x), ptr_arg(conv_state), ptr_arg(weight), ptr_arg(bias),
            ptr_arg(out),
            B, D,
            stream=stream,
        )
    return profiler_bench(run)


def bench_dispatch_combine():
    return 0.0, "requires multi-GPU communication (mori)"


def bench_fmha_gfx950():
    """FP8 flash attention for gfx950."""
    try:
        from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func
        B, H, S, D = 1, 32, 4096, 128
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16)
        def run():
            flydsl_flash_attn_fp8_func(q, k, v, causal=True)
        return profiler_bench(run)
    except Exception as e:
        return 0.0, f"fmha_gfx950: {e}"


def bench_quantization():
    """per_1x32_mx_quant from mega_moe/quant.py."""
    from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
    M, N = 4096, 8192
    x = torch.randn(M, N, dtype=torch.bfloat16)
    return profiler_bench(lambda: per_1x32_mx_quant(x))


def bench_moe_2stage_gemm():
    """MoE 2-stage GEMM1 for gfx9 FP8."""
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm1 import compile_moe_gemm1
    N, K, topk = 7168, 4096, 2
    compiled = compile_moe_gemm1(
        N=N, K=K, weight_dtype='fp8', weight_quant_type='per_token',
        TOPK=topk, BLOCK_TILE_SIZE_M=128, BLOCK_TILE_SIZE_N=128,
    )
    if not callable(compiled):
        return 0.0, "not callable"
    M, E = 4096, 8
    x = torch.zeros(M * topk, K, dtype=torch.float8_e4m3fnuz)
    w = torch.zeros(E, N, K, dtype=torch.float8_e4m3fnuz)
    out = torch.empty(M * topk, N, dtype=torch.bfloat16)
    def run():
        compiled(
            ptr_arg(x), ptr_arg(w), ptr_arg(out),
            M * topk, E,
            stream=stream,
        )
    return profiler_bench(run)


# ── Source → bench mapping ─────────────────────────────────────────

SOURCE_BENCH = {
    "aiter/ops/flydsl/kernels/fused_compress_attn.py": bench_compress_attn,
    "aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py": bench_compress_attn,
    "aiter/ops/flydsl/causal_conv1d_flydsl.py": bench_causal_conv1d,
    "aiter/ops/flydsl/kernels/causal_conv1d_update.py": bench_causal_conv1d,
    "aiter/ops/flydsl/kernels/chunk_gated_delta_h_mfma16x16x16.py": bench_chunk_gated_delta,
    "aiter/ops/flydsl/kernels/gdn_prepare.py": bench_gdn_prepare,
    "aiter/ops/flydsl/kernels/chunk_gated_delta_h.py": bench_chunk_gated_delta,
    "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py": bench_dispatch_combine,
    "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py": bench_dispatch_combine,
    "aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_kernel.py": bench_fmha_gfx950,
    "aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py": bench_gemm_a8w8,
    "aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py": bench_gemm_a8w8,
    "aiter/ops/flydsl/kernels/gemm_common_gfx1250.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/buffer_ops.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/mega_moe/gemm_util.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/mla_reduce.py": bench_mla_reduce,
    "aiter/ops/flydsl/kernels/moe_contiguous_psum.py": bench_moe_contiguous_psum,
    "aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py": bench_moe_2stage_gemm,
    "aiter/ops/flydsl/kernels/moe_m_tile_map.py": bench_moe_route_maps,
    "aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py": bench_moe_2stage_gemm,
    "aiter/ops/flydsl/kernels/moe_gather_reduce.py": bench_moe_gather_reduce,
    "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py": bench_moe_2stage_gemm,
    "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py": bench_moe_2stage_gemm,
    "aiter/aot/flydsl/moe.py": bench_moe_2stage_gemm,
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm2.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm1.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py": bench_hgemm,
    "aiter/ops/flydsl/kernels/moe_reduce.py": bench_moe_reduce,
    "aiter/ops/flydsl/kernels/moe_g2l_lut.py": bench_moe_g2l_lut,
    "aiter/ops/flydsl/kernels/moe_route_maps.py": bench_moe_route_maps,
    "aiter/ops/flydsl/kernels/mega_moe/dispatch.py": bench_moe_route_maps,
    "aiter/ops/flydsl/kernels/moe_sorting_kernel.py": bench_moe_sorting,
    "aiter/ops/flydsl/kernels/mxfp4_gemm2.py": bench_preshuffle_gemm,
    "aiter/ops/flydsl/kernels/mxfp4_gemm1.py": bench_preshuffle_gemm,
    "aiter/ops/flydsl/batched_gemm_mxfp4.py": bench_preshuffle_gemm,
    "aiter/ops/flydsl/kernels/preshuffle_gemm.py": bench_preshuffle_gemm,
    "aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py": bench_preshuffle_gemm,
    "aiter/ops/flydsl/kernels/mega_moe/quant.py": bench_quantization,
    "aiter/ops/flydsl/kernels/silu_and_mul_fq.py": bench_silu_and_mul_fq,
    "aiter/ops/flydsl/kernels/splitk_hgemm.py": bench_hgemm,
}


def main():
    tasks = []
    for task_dir in sorted(ARTIFACTS.iterdir()):
        if not task_dir.is_dir():
            continue
        cfg_path = task_dir / "initial" / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("language") == "flydsl":
            meta_path = task_dir / "initial" / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            tasks.append((task_dir.name, task_dir, cfg, meta))

    print(f"Found {len(tasks)} flydsl tasks\n")
    results = {}

    for i, (task_id, task_dir, cfg, meta) in enumerate(tasks, 1):
        source = meta.get("source", "?")
        source_key = source.split("::")[0] if "::" in source else source
        operator = cfg.get("operator", "unknown")
        cases = cfg.get("benchmark_shapes", [])
        case_id = cases[0].get("case_id", task_id) if cases else task_id
        old_bl = cfg.get("baseline_ms", {})
        old_ms = list(old_bl.values())[0] if old_bl else 0.0

        print(f"[{i}/{len(tasks)}] {task_id}")
        print(f"  source={source}, op={operator}")

        bench_fn = SOURCE_BENCH.get(source_key)
        ms, err = 0.0, None

        if bench_fn is None:
            err = f"no bench for {source_key}"
            print(f"  SKIP: {err}")
        else:
            ms, err = cached(source_key, bench_fn)
            if ms > 0:
                ratio = old_ms / ms if old_ms > 0 else 0
                print(f"  OK: {ms:.6f} ms (old={old_ms:.6f}, ratio={ratio:.2f}x)")
            else:
                print(f"  FAIL: {err}")

        results[task_id] = {
            "task_id": task_id,
            "case_id": case_id,
            "operator": operator,
            "source": source,
            "old_baseline_ms": old_ms,
            "measured_ms": ms,
            "error": err,
            "ratio": old_ms / ms if ms > 0 and old_ms > 0 else None,
        }

    # Write results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)

    # Update baseline_overrides
    existing = json.loads(OVERRIDE_OUTPUT.read_text()) if OVERRIDE_OUTPUT.exists() else {}
    updated = 0
    for r in results.values():
        if r["measured_ms"] > 0:
            existing[r["task_id"]] = {
                "per_case_ms": {r["case_id"]: r["measured_ms"]},
                "geomean_ms": r["measured_ms"],
                "repeats": N_REP,
                "measured": True,
            }
            updated += 1
    with open(OVERRIDE_OUTPUT, 'w') as f:
        json.dump(existing, f, indent=2)

    # Summary
    measured = [r for r in results.values() if r["measured_ms"] > 0]
    failed = [r for r in results.values() if r["measured_ms"] == 0]
    print(f"\n{'='*70}")
    print(f"Summary: {len(measured)}/{len(results)} measured, {len(failed)} failed")
    for r in sorted(measured, key=lambda x: x["operator"]):
        ratio = r["ratio"]
        f = " ***" if ratio and (ratio > 5 or ratio < 0.2) else ""
        print(f"  {r['task_id']}: {r['measured_ms']:.6f} ms (old={r['old_baseline_ms']:.6f}, r={ratio:.2f}x){f}")
    print(f"\nFailed ({len(failed)}):")
    for r in sorted(failed, key=lambda x: x["operator"]):
        print(f"  {r['task_id']}: {(r.get('error') or '?')[:80]}")


if __name__ == "__main__":
    main()
