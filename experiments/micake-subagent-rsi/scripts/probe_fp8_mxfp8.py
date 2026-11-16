"""Generate, compile and measure an FP8 GEMM and an MXFP8 GEMM on gfx950.

FP8 is per-tensor e4m3 (Triton ``tl.float8e4nv`` / ``torch.float8_e4m3fn``) with
``tl.dot``. MXFP8 is the same element type with a power-of-two scale on every
32 elements along K, consumed by ``tl.dot_scaled``.

The champion tile is 64x64x32, one MX scale group per K tile. Run this under
the ROCm PyTorch image (Triton 3.6); the MICake 3.5.1 venv has neither name.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import torch
import triton

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
FP8_MAX = 448.0
SHAPES = ((256, 256, 256), (512, 512, 1024), (1024, 1024, 2048), (2048, 2048, 2048))
OUT = Path("/home/danyzhan/Lumen-RL/output/micake-subagent-rsi/fp8_mxfp8_probe")

FP8_SOURCE = f'''
import triton
import triton.language as tl

@triton.jit
def fp8_gemm(a, b, c, scale_a, scale_b, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * {BLOCK_M} + tl.arange(0, {BLOCK_M})
    offs_n = pid_n * {BLOCK_N} + tl.arange(0, {BLOCK_N})
    offs_k = tl.arange(0, {BLOCK_K})
    acc = tl.zeros(({BLOCK_M}, {BLOCK_N}), dtype=tl.float32)
    for k in range(0, K, {BLOCK_K}):
        a_ptrs = a + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        a_blk = tl.load(a_ptrs)
        b_blk = tl.load(b_ptrs)
        acc += tl.dot(a_blk, b_blk)
    acc = acc * scale_a * scale_b
    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
'''

MXFP8_SOURCE = f'''
import triton
import triton.language as tl

@triton.jit
def mxfp8_gemm(a, b, c, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_as0, stride_as1, stride_bs0, stride_bs1):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * {BLOCK_M} + tl.arange(0, {BLOCK_M})
    offs_n = pid_n * {BLOCK_N} + tl.arange(0, {BLOCK_N})
    offs_k = tl.arange(0, {BLOCK_K})
    acc = tl.zeros(({BLOCK_M}, {BLOCK_N}), dtype=tl.float32)
    for k in range(0, K, {BLOCK_K}):
        a_ptrs = a + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        a_blk = tl.load(a_ptrs)
        b_blk = tl.load(b_ptrs)
        group = k // 32
        a_s = tl.load(a_scale + offs_m[:, None] * stride_as0 + group * stride_as1)
        b_s = tl.load(b_scale + offs_n[:, None] * stride_bs1 + group * stride_bs0)
        acc = tl.dot_scaled(a_blk, a_s, "e4m3", b_blk, b_s, "e4m3", acc)
    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
'''


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quantize_fp8(x: torch.Tensor):
    scale = (x.abs().max() / FP8_MAX).clamp(min=1e-12).float()
    return (x / scale).to(torch.float8_e4m3fn), scale


def _quantize_mx(x: torch.Tensor, along: int):
    """MX e4m3 with an e8m0 scale on every 32 elements along ``along``."""
    group = 32
    moved = x.float().transpose(along, -1).contiguous()
    flat = moved.reshape(-1, moved.shape[-1] // group, group)
    amax = flat.abs().amax(-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).to(torch.int32) + 127
    exp = exp.clamp(0, 254)
    scale = torch.exp2((exp - 127).float())
    quant = (flat / scale.unsqueeze(-1)).reshape(moved.shape).transpose(along, -1).contiguous()
    scales = exp.reshape(moved.shape[:-1] + (moved.shape[-1] // group,)).transpose(along, -1).contiguous()
    return quant.to(torch.float8_e4m3fn), scales.to(torch.uint8)


def _time(fn) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(20):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _error(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - ref.float()).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def main() -> None:
    torch.cuda.set_device(0)
    OUT.mkdir(parents=True, exist_ok=True)
    fp8_path, mx_path = OUT / "fp8_gemm.py", OUT / "mxfp8_gemm.py"
    fp8_path.write_text(FP8_SOURCE)
    mx_path.write_text(MXFP8_SOURCE)
    fp8 = _load(fp8_path, "fp8_gemm").fp8_gemm
    mx = _load(mx_path, "mxfp8_gemm").mxfp8_gemm
    rows = []
    for m, n, k in SHAPES:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(m * 1000 + n + k)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.1
        row = {"m": m, "n": n, "k": k}
        a8, sa = _quantize_fp8(a)
        b8, sb = _quantize_fp8(b)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
        scale_a, scale_b = float(sa), float(sb)
        def launch_fp8():
            fp8[grid](a8, b8, out, scale_a, scale_b, m, n, k, a8.stride(0), a8.stride(1), b8.stride(0), b8.stride(1), out.stride(0), out.stride(1), num_warps=8, num_stages=2)
        launch_fp8()
        torch.cuda.synchronize()
        ref = torch.matmul(a8.float() * sa, b8.float() * sb)
        row["fp8_error"] = _error(out, ref)
        row["fp8_ms"] = _time(launch_fp8)
        row["bf16_torch_ms"] = _time(lambda: torch.matmul(a, b))

        aq, asc = _quantize_mx(a, 1)
        bq, bsc = _quantize_mx(b, 0)
        def launch_mx():
            mx[grid](aq, bq, out, asc, bsc, m, n, k, aq.stride(0), aq.stride(1), bq.stride(0), bq.stride(1), out.stride(0), out.stride(1), asc.stride(0), asc.stride(1), bsc.stride(0), bsc.stride(1), num_warps=8, num_stages=2)
        launch_mx()
        torch.cuda.synchronize()
        # Reconstruct the dequantized inputs with the same e8m0 scales.
        def dequant(q, scales, along):
            vals = q.float()
            moved = vals.transpose(along, -1)
            sm = scales.float().transpose(along, -1)
            scale = torch.exp2(sm - 127).repeat_interleave(32, dim=-1)
            return (moved * scale).transpose(along, -1)
        ref_mx = torch.matmul(dequant(aq, asc, 1), dequant(bq, bsc, 0))
        row["mxfp8_error"] = _error(out, ref_mx)
        row["mxfp8_ms"] = _time(launch_mx)
        rows.append(row)
        print(json.dumps(row), flush=True)
    document = {"device": torch.cuda.get_device_name(0), "triton": triton.__version__, "rows": rows}
    (OUT / "probe.json").write_text(json.dumps(document, indent=2))
    print("wrote", OUT / "probe.json")


if __name__ == "__main__":
    main()
