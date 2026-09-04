#!/usr/bin/env python3
"""Compile, validate, or benchmark the gfx950 native MXFP4 dense GEMM task."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_ARCH = "gfx950"
QUANT_BLOCK_SIZE = 32
OUTPUT_DTYPE = torch.bfloat16

CANONICAL_CASES = [
    ("decode_m16", 16, 512, 1024),
    ("prefill_m128", 128, 1024, 1024),
    ("rectangular_m256", 256, 768, 2048),
]


def configured_cases():
    spec_path = TASK_DIR / "task_spec.json"
    if not spec_path.is_file():
        return CANONICAL_CASES
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("format") != "mxfp4":
        raise RuntimeError("task_spec.json format must be mxfp4")
    shapes = spec.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise RuntimeError("generated MXFP4 task requires exactly one shape")
    shape = shapes[0]
    return [
        (
            "m%d_n%d_k%d" % (shape["M"], shape["N"], shape["K"]),
            int(shape["M"]),
            int(shape["N"]),
            int(shape["K"]),
        )
    ]


CASES = configured_cases()


def runtime_arch() -> str:
    if not torch.cuda.is_available():
        return "no HIP device"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", "")
    return (arch or "unknown").split(":", 1)[0].lower()


def require_gfx950() -> None:
    detected = runtime_arch()
    if detected != SUPPORTED_ARCH:
        raise RuntimeError(
            f"native MXFP4 dense GEMM requires gfx950; detected {detected}"
        )


def load_kernel():
    """Import the editable wrapper only after the architecture gate."""

    sys.path.insert(0, str(TASK_DIR))
    os.chdir(TASK_DIR)
    return importlib.import_module("kernel").mxfp4_gemm


def make_case(index: int):
    """Quantize source tensors with AITER's production HIP MXFP4 API."""

    from aiter.ops.quant import quant_mxfp4_hip

    _, m, n, k = CASES[index]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1701 + index)
    a_source = torch.randn(
        (m, k), device="cuda", dtype=torch.float16, generator=generator
    )
    w_source = torch.randn(
        (n, k), device="cuda", dtype=torch.float16, generator=generator
    )
    a_packed, a_scales = quant_mxfp4_hip(
        a_source.contiguous(), group_size=QUANT_BLOCK_SIZE
    )
    w_packed, w_scales = quant_mxfp4_hip(
        w_source.contiguous(), group_size=QUANT_BLOCK_SIZE
    )
    return (
        a_packed.contiguous(),
        w_packed.contiguous(),
        a_scales.contiguous(),
        w_scales.contiguous(),
    )


def dequantized_reference(
    a_packed: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
) -> torch.Tensor:
    """Build the FP32 reference from the exact packed values sent to GEMM."""

    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    a = mxfp4_to_f32(a_packed)
    w = mxfp4_to_f32(w_packed)
    a_scale = e8m0_to_f32(a_scales).repeat_interleave(
        QUANT_BLOCK_SIZE, dim=-1
    )
    w_scale = e8m0_to_f32(w_scales).repeat_interleave(
        QUANT_BLOCK_SIZE, dim=-1
    )
    return torch.matmul(a * a_scale, (w * w_scale).T).to(OUTPUT_DTYPE)


def compile_kernel(mxfp4_gemm) -> None:
    operands = make_case(0)
    mxfp4_gemm(*operands)
    torch.cuda.synchronize()
    print(
        "Compilation: PASS "
        "(aiter.ops.triton.gemm.basic.gemm_afp4wfp4, gfx950 native E2M1)"
    )


def check_correctness(mxfp4_gemm) -> None:
    for index, (name, _, _, _) in enumerate(CASES):
        operands = make_case(index)
        expected = dequantized_reference(*operands)
        actual = mxfp4_gemm(*operands)
        torch.cuda.synchronize()
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name}: native GEMM produced non-finite output")
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=2.0e-2,
            atol=5.0e-1,
        )
        max_abs = (actual.float() - expected.float()).abs().max().item()
        print(f"Correctness: PASS ({name}, max_abs_error={max_abs:.6g})")


def time_ms(fn, warmup: int = 10, repeats: int = 40) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def benchmark(mxfp4_gemm) -> None:
    results = []
    for index, (name, m, n, k) in enumerate(CASES):
        operands = make_case(index)
        out = torch.empty((m, n), device="cuda", dtype=OUTPUT_DTYPE)
        latency_ms = time_ms(lambda: mxfp4_gemm(*operands, out=out))
        tflops = (2.0 * m * n * k) / (latency_ms * 1.0e9)
        result = {
            "test_case_id": name,
            "execution_time_ms": latency_ms,
            "logical_tflops": tflops,
            "params": {
                "M": m,
                "N": n,
                "K": k,
                "format": "mxfp4",
                "quant_block": QUANT_BLOCK_SIZE,
                "output_dtype": "bfloat16",
            },
        }
        results.append(result)
        print(f"Performance: {latency_ms:.6f} ms, {tflops:.3f} TFLOP/s ({name})")

    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    report = {
        "architecture": SUPPORTED_ARCH,
        "production_api": "aiter.ops.triton.gemm.basic.gemm_afp4wfp4",
        "timing_scope": "native GEMM only; excludes MXFP4 quantization and reference",
        "test_cases": results,
    }
    (build / "performance_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode

    # This is intentionally before importing kernel/AITER or allocating tensors.
    require_gfx950()
    mxfp4_gemm = load_kernel()

    if mode == "compile":
        compile_kernel(mxfp4_gemm)
    elif mode == "correctness":
        check_correctness(mxfp4_gemm)
    else:
        benchmark(mxfp4_gemm)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
