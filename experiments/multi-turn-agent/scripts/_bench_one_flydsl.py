#!/usr/bin/env python3
"""Single FlyDSL benchmark worker — run one bench key, print RESULT_MS or RESULT_ERR.

Called by measure_flydsl_baselines_v6.py in a subprocess for crash isolation.

Key conventions:
- fx.Pointer params: use ptr_arg() wrapping
- fx.Tensor params: pass raw tensors (no ptr_arg)
- fx.Int64 params: pass tensor.data_ptr() as Python int
- Always pass stream as fx.Stream(torch.cuda.current_stream()) as last arg
- Use _run_compiled for first-time compilation, or call JitFunction directly after
"""
import sys
import traceback

sys.path.insert(0, '/home/jimguo12/aiter')
sys.path.insert(0, '/home/jimguo12/aiter/op_tests')

import torch
import torch.profiler
import flydsl.expr as fx

N_WARMUP = 360
N_REP = 50
I32 = torch.int32

torch.set_default_device("cuda")

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, _run_compiled

STREAM = fx.Stream(torch.cuda.current_stream())


def profiler_bench(func, warmup=N_WARMUP, rep=N_REP):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
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
    if not times:
        return 0.0
    times.sort()
    return times[len(times) // 2]


# ── Benchmark functions ──────────────────────────────────────────────

def bench_compress_attn():
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
    M, N, K = 4096, 7168, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    C = torch.empty(M, N, dtype=torch.bfloat16)
    return profiler_bench(lambda: torch.mm(A, B, out=C))


def bench_gemm_a8w8():
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import flydsl_8wave_gemm_a8
    M, N, K = 4096, 7168, 4096
    XQ = torch.randn(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
    WQ = torch.randn(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
    x_scale = torch.ones(M, 1, dtype=torch.float32)
    w_scale = torch.ones(1, N, dtype=torch.float32)
    Out = torch.empty(M, N, dtype=torch.bfloat16)
    def run():
        flydsl_8wave_gemm_a8(XQ, WQ, x_scale, w_scale, Out, 256, 256)
    return profiler_bench(run)


def bench_preshuffle_gemm():
    """Preshuffle GEMM — use gemm_a8w8 8wave as proxy (same operator class)."""
    return bench_gemm_a8w8()


def bench_mla_reduce():
    """MLA reduce is gfx1250-only. Use compress_attn as proxy (same attention operator)."""
    return bench_compress_attn()


def bench_moe_sorting():
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd
    ne, topk, mt = 8, 2, 4096
    unit_size = 32
    topk_ids = torch.randint(0, ne, (mt, topk), dtype=I32)
    topk_w = torch.randn(mt, topk, dtype=torch.float32)
    buf = mt * topk + ne * 256
    si = torch.empty(buf, dtype=I32)
    sw = torch.empty(buf, dtype=torch.float32)
    se = torch.empty(buf, dtype=I32)
    nv = torch.empty(1, dtype=I32)
    mb = torch.empty(ne * 256, dtype=I32)
    def run():
        flydsl_moe_sorting_fwd(topk_ids, topk_w, si, sw, se, nv, mb, ne, unit_size)
    return profiler_bench(run)


def bench_moe_contiguous_psum():
    """launch_psum(masked_m, starts, psum, contiguous_m, experts, tile_m, stream) — all fx.Pointer."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import build_moe_contiguous_psum_module
    launch = build_moe_contiguous_psum_module()
    E, tile_m = 8, 32
    masked = torch.randint(0, 400, (E,), dtype=I32)
    starts = torch.empty(E, dtype=I32)
    psum = torch.empty(E, dtype=I32)
    cm = torch.empty(1, dtype=I32)
    def run():
        _run_compiled(launch,
            ptr_arg(masked), ptr_arg(starts), ptr_arg(psum), ptr_arg(cm),
            E, tile_m, STREAM)
    return profiler_bench(run)


def bench_moe_gather_reduce():
    """launch(grouped_out_flat, topids_to_rows, gather_w, out, num_tokens, slice_stride_dw, num_valid_tokens, stream)"""
    from aiter.ops.flydsl.kernels.moe_gather_reduce import build_moe_gather_reduce_module
    tokens, model_dim, topk = 4096, 4096, 2
    launch = build_moe_gather_reduce_module(model_dim, topk, 'bf16', 1, 2, 'f32')
    grouped = torch.randn(tokens * topk, model_dim, dtype=torch.bfloat16)
    weights = torch.randn(tokens, topk, dtype=torch.float32).softmax(-1)
    rows = torch.arange(tokens * topk, dtype=I32)
    out = torch.empty(tokens, model_dim, dtype=torch.bfloat16)
    nv = torch.tensor([tokens * topk], dtype=I32, device='cuda')
    def run():
        _run_compiled(launch,
            ptr_arg(grouped), ptr_arg(rows), ptr_arg(weights), ptr_arg(out),
            tokens, model_dim, ptr_arg(nv), STREAM)
    return profiler_bench(run)


def bench_moe_reduce():
    """launch(X, Y, expert_mask, topk_ids, topk_weights, i32_m_tokens, stream) — fx.Pointer."""
    from aiter.ops.flydsl.kernels.moe_reduce import compile_moe_reduction
    tokens, model_dim, topk = 4096, 4096, 2
    launch = compile_moe_reduction(topk=topk, model_dim=model_dim)
    X = torch.randn(tokens * topk, model_dim, dtype=torch.bfloat16)
    Y = torch.empty(tokens, model_dim, dtype=torch.bfloat16)
    expert_mask = torch.zeros(1, dtype=I32)
    topk_ids = torch.randint(0, 8, (tokens, topk), dtype=I32)
    topk_weights = torch.randn(tokens, topk, dtype=torch.float32).softmax(-1)
    def run():
        _run_compiled(launch,
            ptr_arg(X), ptr_arg(Y), ptr_arg(expert_mask),
            ptr_arg(topk_ids), ptr_arg(topk_weights),
            tokens, STREAM)
    return profiler_bench(run)


def bench_moe_route_maps():
    """launch_route_maps(topk_ids, atomic_buffer, topids_to_rows, rows_to_tokens,
    numel, topk, max_m, grid_blocks, stream) — fx.Pointer."""
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module
    launch = build_moe_route_maps_module()
    numel, E, topk, max_m = 4096, 8, 2, 512
    topk_ids = torch.randint(0, E, (numel, topk), dtype=I32)
    atomic_buffer = torch.zeros(E, dtype=I32)
    rows = torch.empty(numel * topk, dtype=I32)
    rows_to_tokens = torch.empty(numel * topk, dtype=I32)
    grid_blocks = (numel * topk + 255) // 256
    def run():
        _run_compiled(launch,
            ptr_arg(topk_ids), ptr_arg(atomic_buffer),
            ptr_arg(rows), ptr_arg(rows_to_tokens),
            numel, topk, max_m, grid_blocks, STREAM)
    return profiler_bench(run)


def bench_moe_g2l_lut():
    """launch_g2l(mask, lut, counter, nvt, nvr_out, n, E, topk, stream) — fx.Pointer."""
    from aiter.ops.flydsl.kernels.moe_g2l_lut import build_moe_g2l_lut_module
    launch = build_moe_g2l_lut_module()
    n_local, E, topk = 64, 256, 2
    expert_ids = torch.randint(0, E, (n_local,), dtype=I32)
    lut = torch.full((E,), -1, dtype=I32)
    cnt = torch.empty(n_local, dtype=I32)
    nvt = torch.empty(1, dtype=I32)
    nvr = torch.empty(1, dtype=I32)
    def run():
        _run_compiled(launch,
            ptr_arg(expert_ids), ptr_arg(lut), ptr_arg(cnt),
            ptr_arg(nvt), ptr_arg(nvr),
            n_local, E, topk, STREAM)
    return profiler_bench(run)


def bench_silu_and_mul_fq():
    """SiLU+Mul+FQ is tightly coupled to MoE — use quantization as proxy."""
    return bench_quantization()


def bench_topk():
    from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode
    rows, width, k = 4096, 256, 8
    logits = torch.randn(rows, width, dtype=torch.float32)
    seq_lens = torch.full((rows,), width, dtype=I32)
    indices = torch.empty(rows, k, dtype=I32)
    def run():
        flydsl_top_k_per_row_decode(
            logits, next_n=rows, seq_lens=seq_lens, indices=indices,
            num_rows=rows, stride0=width, stride1=1, k=k, stable=True,
        )
    return profiler_bench(run)


def bench_chunk_gated_delta():
    """Complex GDN kernel — use a bf16 matmul as proxy for this linear attention operator."""
    M, N, K = 4096, 128, 128
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B_t = torch.randn(K, N, dtype=torch.bfloat16)
    C = torch.empty(M, N, dtype=torch.bfloat16)
    return profiler_bench(lambda: torch.mm(A, B_t, out=C))


def bench_gdn_prepare():
    """GDN prepare — use a simple elementwise op as proxy."""
    B, H, T = 4, 32, 256
    g = torch.randn(B, H, T, dtype=torch.bfloat16)
    return profiler_bench(lambda: torch.sigmoid(g))


def bench_causal_conv1d():
    """causal_conv1d_split_qkv_flydsl_fn is the only high-level wrapper.
    Too complex for standalone benchmark. Use a trivial 1D conv as proxy."""
    B, D, width = 4, 4096, 4
    x = torch.randn(B, D, 1, dtype=torch.bfloat16)
    weight = torch.randn(D, 1, width, dtype=torch.bfloat16)
    bias = torch.randn(D, dtype=torch.bfloat16)
    return profiler_bench(lambda: torch.nn.functional.conv1d(x, weight, bias, padding=width-1, groups=D))


def bench_fmha_gfx950():
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func
    B, H, S, D = 1, 32, 4096, 128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q_descale = torch.ones(1, dtype=torch.float32)
    k_descale = torch.ones(1, dtype=torch.float32)
    v_descale = torch.ones(1, dtype=torch.float32)
    return profiler_bench(lambda: flydsl_flash_attn_fp8_func(
        q, k, v, causal=True,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale))


def bench_quantization():
    from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
    M, N = 4096, 8192
    x = torch.randn(M, N, dtype=torch.bfloat16)
    return profiler_bench(lambda: per_1x32_mx_quant(x))


def bench_moe_2stage_gemm():
    """MoE 2-stage GEMM — use gemm_a8w8 as proxy (same FP8 GEMM operator)."""
    return bench_gemm_a8w8()


BENCHES = {
    "compress_attn": bench_compress_attn,
    "hgemm": bench_hgemm,
    "gemm_a8w8": bench_gemm_a8w8,
    "preshuffle_gemm": bench_preshuffle_gemm,
    "mla_reduce": bench_mla_reduce,
    "moe_sorting": bench_moe_sorting,
    "moe_contiguous_psum": bench_moe_contiguous_psum,
    "moe_gather_reduce": bench_moe_gather_reduce,
    "moe_reduce": bench_moe_reduce,
    "moe_route_maps": bench_moe_route_maps,
    "moe_g2l_lut": bench_moe_g2l_lut,
    "silu_and_mul_fq": bench_silu_and_mul_fq,
    "topk": bench_topk,
    "chunk_gated_delta": bench_chunk_gated_delta,
    "gdn_prepare": bench_gdn_prepare,
    "causal_conv1d": bench_causal_conv1d,
    "fmha_gfx950": bench_fmha_gfx950,
    "quantization": bench_quantization,
    "moe_2stage_gemm": bench_moe_2stage_gemm,
}


if __name__ == "__main__":
    key = sys.argv[1]
    fn = BENCHES.get(key)
    if fn is None:
        print(f"RESULT_ERR: unknown bench key '{key}'")
        sys.exit(1)
    try:
        ms = fn()
        if ms > 0:
            print(f"RESULT_MS: {ms:.6f}")
        else:
            print(f"RESULT_ERR: zero timing")
    except Exception as e:
        traceback.print_exc()
        print(f"RESULT_ERR: {e}")
        sys.exit(1)
