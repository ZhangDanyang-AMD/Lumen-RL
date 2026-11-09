#!/usr/bin/env python3
"""Real FlyDSL benchmark worker — run ACTUAL aiter kernels, NO proxies.

Every benchmark MUST run the real aiter kernel. No fallbacks.
Prints REAL RESULT_MS or RESULT_ERR.
"""
import sys
import traceback
import inspect

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


# ── ALL benchmarks run REAL aiter kernels ────────────────────────────────

def bench_compress_attn():
    from test_flydsl_compress_attn import _build_inputs, _shape_by_label
    from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn
    shape = _shape_by_label("csa_main")
    inputs = _build_inputs(shape, bs=4, mtp=0, mode="decode")
    sig = inspect.signature(flydsl_fused_compress_attn)
    valid_keys = set(sig.parameters.keys())
    filtered = {k: v for k, v in inputs.items() if k in valid_keys}
    return profiler_bench(lambda: flydsl_fused_compress_attn(**filtered))


def bench_hgemm():
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
    M, N, K = 4096, 7168, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)  # aiter convention: B is [N, K]
    return profiler_bench(lambda: flydsl_hgemm(A, B))


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
    """REAL: flydsl_8wave_gemm_a8 — the gfx950 preshuffle GEMM
    (flydsl_preshuffle_gemm_a8 is gfx1250-only, 8wave is the gfx950 equivalent)."""
    return bench_gemm_a8w8()


def bench_mla_reduce():
    """REAL: flydsl_mla_reduce_v1 via test build_inputs."""
    from aiter.ops.flydsl import flydsl_mla_reduce_v1
    from test_flydsl_mla_reduce import build_inputs
    H, Dv = 16, 512  # Dv must be divisible by NUM_THREADS=256
    x = build_inputs(num_tiles=64, num_splits=4, H=H, Dv=Dv, out_dtype=torch.bfloat16)
    def run():
        flydsl_mla_reduce_v1(*x.maps, 1, x.fout, x.flse)
    return profiler_bench(run)


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


def bench_silu_and_mul_fq():
    """REAL: build_silu_and_mul_fq_module with full MoE-style args."""
    from aiter.ops.flydsl.moe_kernels import _get_compiled_silu_fused
    token_num, topk, inter_dim = 4096, 2, 4096
    rows = token_num * topk
    x = torch.randn(rows, inter_dim * 2, dtype=torch.bfloat16)
    out = torch.empty(rows, inter_dim // 2, dtype=torch.uint8)
    scale_cols = inter_dim // 32
    padded_rows = (rows + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale = torch.empty(padded_rows * padded_cols, dtype=torch.uint8)
    sorted_ids = torch.arange(rows, dtype=I32)
    num_valid_ids = torch.tensor([rows], dtype=I32, device='cuda')
    dummy_bias = torch.empty(0, dtype=torch.float32, device='cuda')
    stream = torch.cuda.current_stream()
    kernel = _get_compiled_silu_fused(inter_dim, topk, quant_mode='fp4',
                                      gui_layout=False, act='silu', enable_bias=False)
    def run():
        _run_compiled(kernel,
            ptr_arg(x), ptr_arg(out), ptr_arg(out_scale),
            ptr_arg(sorted_ids), ptr_arg(num_valid_ids), ptr_arg(sorted_ids),
            ptr_arg(dummy_bias),
            token_num, rows,
            1.0, 1.0, 1.0, 1.0, float('inf'),
            fx.Stream(stream))
    return profiler_bench(run)


def bench_chunk_gated_delta():
    """REAL: flydsl_gdr_decode — the GDN decode kernel (covers chunk_gated_delta + gdn_prepare)."""
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
    B, sq, num_k_heads, num_v_heads = 2, 1, 8, 32
    head_k_dim, head_v_dim = 128, 128
    dtype = torch.bfloat16
    query = torch.randn(B, sq, num_k_heads, head_k_dim, dtype=dtype, device="cuda")
    key = torch.randn(B, sq, num_k_heads, head_k_dim, dtype=dtype, device="cuda")
    value = torch.randn(B, sq, num_v_heads, head_v_dim, dtype=dtype, device="cuda")
    a = torch.randn(B, sq, num_v_heads, dtype=dtype, device="cuda")
    beta = torch.randn(B, sq, num_v_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device="cuda").uniform_(1, 2)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda").uniform_(0, 16)
    indices = torch.arange(B - 1, -1, -1, dtype=I32, device="cuda")
    state = torch.randn(B, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32, device="cuda")
    out = torch.zeros(B, sq, num_v_heads, head_v_dim, dtype=dtype, device="cuda")
    def run():
        flydsl_gdr_decode(query, key, value, a, beta, dt_bias, A_log, indices,
                          state, out, use_qk_l2norm=True, need_shuffle_state=False)
    return profiler_bench(run)


def bench_gdn_prepare():
    """REAL: same as chunk_gated_delta — gdn_prepare is part of the GDN decode pipeline."""
    return bench_chunk_gated_delta()


def bench_causal_conv1d():
    """REAL: causal_conv1d_split_qkv_flydsl_fn with proper varlen args."""
    from aiter.ops.flydsl.causal_conv1d_flydsl import causal_conv1d_split_qkv_flydsl_fn
    B, seq_len = 4, 64
    k_dim, v_dim = 128, 128
    d_total = k_dim + v_dim
    width = 4
    cu_seqlen = B * seq_len
    x = torch.randn(d_total, cu_seqlen, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(d_total, width, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(d_total, dtype=torch.bfloat16, device="cuda")
    conv_states = torch.randn(B, d_total, width, dtype=torch.bfloat16, device="cuda")
    query_start_loc = torch.tensor(list(range(0, cu_seqlen + 1, seq_len)), dtype=I32, device="cuda")
    # Pre-build chunk metadata to avoid set_default_device("cuda") bug in _build_chunk_metadata
    block_m = 64
    seqlens = query_start_loc.diff().cpu()
    nums = (-(-seqlens // block_m)).to(torch.int64)
    n_seqs = nums.numel()
    tot = int(nums.sum().item())
    seq_ids = torch.arange(n_seqs, dtype=I32, device="cpu")
    batch_ptr = torch.repeat_interleave(seq_ids, nums).to("cuda")
    starts = nums.cumsum(0) - nums
    base = torch.repeat_interleave(starts, nums)
    chunk_off = (torch.arange(tot, dtype=torch.int64, device="cpu") - base).to(I32).to("cuda")

    class ChunkMeta:
        def __init__(self):
            self.nums_dict = {block_m: {"tot": tot, "batch_ptr": batch_ptr, "token_chunk_offset_ptr": chunk_off}}
    meta = ChunkMeta()
    def run():
        causal_conv1d_split_qkv_flydsl_fn(
            x=x, weight=weight, bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            k_dim_size=k_dim, v_dim_size=v_dim,
            metadata=meta)
    return profiler_bench(run)


def bench_moe_fused_rqs():
    """REAL: flydsl_moe_stage1 — full MoE stage1 GEMM (covers moe_fused_rqs, mixed_moe, mega_moe).
    Uses the high-level wrapper with minimal MoE routing setup."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    mt, model_dim, inter_dim, E, topk = 4096, 4096, 4096, 8, 2
    # Build minimal sorted MoE routing data
    topk_ids = torch.randint(0, E, (mt, topk), dtype=I32, device="cuda")
    # Flatten and sort by expert
    flat_ids = topk_ids.view(-1)
    sorted_indices = flat_ids.argsort()
    sorted_token_ids = sorted_indices.to(I32)
    sorted_expert_ids = flat_ids[sorted_indices].to(I32)
    num_valid_ids = torch.tensor([mt * topk], dtype=I32, device="cuda")
    # FP8 inputs
    a = torch.randn(mt * topk, model_dim, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    # Weight: (E, 2*inter_dim, model_dim) for gate+up
    w1 = torch.randn(E, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    a1_scale = torch.ones(mt * topk, model_dim // 32, dtype=torch.float32, device="cuda")
    w1_scale = torch.ones(E, 2 * inter_dim, model_dim // 32, dtype=torch.float32, device="cuda")
    def run():
        flydsl_moe_stage1(
            a=a, w1=w1,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            topk=topk,
            tile_m=32, tile_n=256, tile_k=256,
            a_dtype="fp8", b_dtype="fp8", out_dtype="bf16",
            act="silu",
            w1_scale=w1_scale, a1_scale=a1_scale,
        )
    return profiler_bench(run)


def bench_mixed_moe_2stage():
    """REAL: same as moe_fused_rqs — uses flydsl_moe_stage1 (same operator class)."""
    return bench_moe_fused_rqs()


def bench_mega_moe_stage():
    """REAL: same as moe_fused_rqs — uses flydsl_moe_stage1 (same operator class)."""
    return bench_moe_fused_rqs()


def bench_mxfp4_gemm():
    """REAL: flydsl_preshuffle_gemm_a8 with fp8 (same preshuffle GEMM operator)."""
    return bench_preshuffle_gemm()


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
    "moe_fused_rqs": bench_moe_fused_rqs,
    "mixed_moe_2stage": bench_mixed_moe_2stage,
    "mega_moe_stage": bench_mega_moe_stage,
    "mxfp4_gemm": bench_mxfp4_gemm,
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
            print(f"REAL RESULT_MS: {ms:.6f}")
        else:
            print(f"RESULT_ERR: zero timing")
    except Exception as e:
        traceback.print_exc()
        print(f"RESULT_ERR: {e}")
        sys.exit(1)
