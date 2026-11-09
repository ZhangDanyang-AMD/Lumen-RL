#!/usr/bin/env python3
"""Compile, validate, and benchmark gfx942 FP8 A8W8 dense GEMM."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)

from kernel import FP8_DTYPE, fp8_a8w8_gemm, require_gfx942_fp8  # noqa: E402


# M/N cover decode, batched decode, prefill, and rectangular projection regimes.
# Every K is aligned to 256 (and therefore to the default Triton BLOCK_K).
CANONICAL_CASES = [
    ("decode_m1", 1, 4096, 4096),
    ("decode_m32_rectangular", 32, 6144, 4096),
    ("prefill_m512", 512, 4096, 4096),
    ("prefill_rectangular", 256, 3072, 5120),
]


def configured_cases():
    spec_path = TASK_DIR / "task_spec.json"
    if not spec_path.is_file():
        return CANONICAL_CASES
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("format") != "fp8":
        raise RuntimeError("task_spec.json format must be fp8")
    shapes = spec.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise RuntimeError("generated FP8 task requires exactly one shape")
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
K_ALIGNMENT = 256
FP16_RTOL = 5.0e-3
FP16_ATOL = 5.0e-3


def quantize_rows(source: torch.Tensor, dequantize: bool):
    """Symmetrically quantize contiguous FP16/FP32 rows to E4M3 FNUZ."""

    if FP8_DTYPE is None:
        raise RuntimeError(
            "torch.float8_e4m3fnuz is unavailable; use a ROCm PyTorch build "
            "with gfx942 FP8 support."
        )
    if source.ndim != 2 or source.dtype not in (torch.float16, torch.float32):
        raise TypeError("row quantization expects a 2D FP16 or FP32 source")

    source_fp32 = source.to(torch.float32)
    row_absmax = source_fp32.abs().amax(dim=1)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    scale = torch.where(
        row_absmax > 0,
        row_absmax / fp8_max,
        torch.ones_like(row_absmax),
    ).contiguous()
    normalized = torch.clamp(
        source_fp32 / scale[:, None], min=-fp8_max, max=fp8_max
    )
    quantized = normalized.to(FP8_DTYPE).contiguous()
    dequantized = (
        quantized.to(torch.float32) * scale[:, None] if dequantize else None
    )
    return quantized, scale, dequantized


def make_case(index: int, with_reference: bool):
    """Create deterministic row-quantized operands and an optional reference."""

    device = require_gfx942_fp8()
    name, m, n, k = CASES[index]
    seed = 1701 + index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Activations are FP16 [M,K]. Weights are FP32 [N,K], making row-wise
    # weight quantization exactly per-output-channel quantization.
    activation_source = (
        torch.randn((m, k), device=device, dtype=torch.float16) * 0.5
    )
    weight_source = torch.randn(
        (n, k), device=device, dtype=torch.float32
    ) * (0.5 / math.sqrt(k))

    a_quantized, a_scale, a_dequantized = quantize_rows(
        activation_source, dequantize=with_reference
    )
    weight_quantized, weight_scale, weight_dequantized = quantize_rows(
        weight_source, dequantize=with_reference
    )
    # B is [K,N]. Its stride-1 K dimension is useful for gfx942 matrix loads.
    b_quantized = weight_quantized.transpose(0, 1)

    reference = None
    if with_reference:
        # This is intentionally based on the quantized values, never the
        # original source tensors. Scaling and matmul are both FP32.
        reference = torch.matmul(
            a_dequantized, weight_dequantized.transpose(0, 1)
        ).to(torch.float16)
    return name, a_quantized, b_quantized, a_scale, weight_scale, reference


def time_ms(function, warmup: int = 10, repeats: int = 40) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def compile_kernel() -> None:
    _, a, b, a_scale, b_scale, _ = make_case(0, with_reference=False)
    fp8_a8w8_gemm(a, b, a_scale, b_scale)
    torch.cuda.synchronize()
    print("Compilation: PASS (gfx942, float8_e4m3fnuz)")


def check_correctness() -> None:
    for index, (case_name, _, _, _) in enumerate(CASES):
        name, a, b, a_scale, b_scale, reference = make_case(
            index, with_reference=True
        )
        assert name == case_name and reference is not None
        actual = fp8_a8w8_gemm(a, b, a_scale, b_scale)
        torch.cuda.synchronize()
        if actual.dtype != torch.float16:
            raise AssertionError("kernel output must be FP16")
        if not torch.isfinite(actual).all():
            raise AssertionError("kernel output contains non-finite values")
        torch.testing.assert_close(
            actual,
            reference,
            rtol=FP16_RTOL,
            atol=FP16_ATOL,
        )
        print("Correctness: PASS (%s)" % name)


def benchmark() -> None:
    results = []
    for index, (case_name, m, n, k) in enumerate(CASES):
        name, a, b, a_scale, b_scale, _ = make_case(
            index, with_reference=False
        )
        assert name == case_name
        out = torch.empty((m, n), device=a.device, dtype=torch.float16)

        # Validate and compile once. The timed closure launches only the GEMM;
        # source creation and FP8 quantization have already completed.
        fp8_a8w8_gemm(a, b, a_scale, b_scale, out=out)

        def launch():
            return fp8_a8w8_gemm(
                a,
                b,
                a_scale,
                b_scale,
                out=out,
                validate=False,
            )

        latency = time_ms(launch)
        results.append(
            {
                "test_case_id": name,
                "execution_time_ms": latency,
                "params": {
                    "M": m,
                    "N": n,
                    "K": k,
                    "input_dtype": "float8_e4m3fnuz",
                    "output_dtype": "float16",
                    "activation_scale": "per_token_fp32",
                    "weight_scale": "per_output_channel_fp32",
                },
            }
        )
        print("Perf: %.6f ms (%s)" % (latency, name))

    build_dir = TASK_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    (build_dir / "performance_report.json").write_text(
        json.dumps({"test_cases": results}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    if mode == "compile":
        compile_kernel()
    elif mode == "correctness":
        check_correctness()
    else:
        benchmark()


if __name__ == "__main__":
    main()
