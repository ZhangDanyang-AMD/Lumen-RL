#!/usr/bin/env python3
"""Compile, validate, and benchmark the single-launch grouped GEMM task."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)

from kernel import grouped_gemm  # noqa: E402


CASES = [
    ("decode_skew", 8, 64, 1024, 1024, [1, 1, 2, 3, 5, 8, 13, 64]),
    ("mixed_skew", 8, 128, 2048, 1024, [4, 7, 11, 19, 31, 53, 89, 128]),
    ("prefill_skew", 8, 256, 1024, 2048, [16, 24, 40, 63, 96, 144, 208, 256]),
]


def make_case(index):
    _, experts, m_max, k, n, routed_counts = CASES[index]
    torch.manual_seed(300 + index)
    activations = torch.randn(
        (experts, m_max, k), device="cuda", dtype=torch.float16
    )
    weights = torch.randn((experts, k, n), device="cuda", dtype=torch.float16)
    group_m = torch.tensor(routed_counts, device="cuda", dtype=torch.int32)
    return activations, weights, group_m


def time_ms(fn, warmup=10, repeats=25):
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


def compile_kernel():
    activations, weights, group_m = make_case(0)
    grouped_gemm(activations, weights, group_m)
    torch.cuda.synchronize()
    print("Compilation: PASS")


def check_correctness():
    for index, (name, experts, _, _, _, routed_counts) in enumerate(CASES):
        activations, weights, group_m = make_case(index)
        actual = grouped_gemm(activations, weights, group_m)
        for expert in range(experts):
            rows = routed_counts[expert]
            expected = torch.matmul(
                activations[expert, :rows], weights[expert]
            )
            torch.testing.assert_close(
                actual[expert, :rows], expected, rtol=2e-2, atol=2e-2
            )
        print("Correctness: PASS (%s)" % name)


def benchmark():
    results = []
    for index, (name, experts, m_max, k, n, routed_counts) in enumerate(CASES):
        activations, weights, group_m = make_case(index)
        latency = time_ms(lambda: grouped_gemm(activations, weights, group_m))
        results.append(
            {
                "test_case_id": name,
                "execution_time_ms": latency,
                "params": {
                    "experts": experts,
                    "M_max": m_max,
                    "N": n,
                    "K": k,
                    "group_m": routed_counts,
                    "dtype": "fp16",
                },
            }
        )
        print("Perf: %.6f ms (%s)" % (latency, name))
    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({"test_cases": results}, indent=2) + "\n"
    )


def main():
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
