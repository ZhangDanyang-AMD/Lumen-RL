#!/usr/bin/env python3
"""Compile, validate, and benchmark the Triton GEMM task."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)

from kernel import gemm  # noqa: E402


CASES = [
    ("decode_m32", 32, 4096, 4096),
    ("prefill_m512", 512, 4096, 4096),
    ("rectangular", 1024, 3072, 4096),
]


def make_case(index):
    _, m, n, k = CASES[index]
    torch.manual_seed(100 + index)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    return a, b


def time_ms(fn, warmup=10, repeats=40):
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
    a, b = make_case(0)
    gemm(a, b)
    torch.cuda.synchronize()
    print("Compilation: PASS")


def check_correctness():
    for index, (name, _, _, _) in enumerate(CASES):
        a, b = make_case(index)
        actual = gemm(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print("Correctness: PASS (%s)" % name)


def benchmark():
    results = []
    for index, (name, m, n, k) in enumerate(CASES):
        a, b = make_case(index)
        latency = time_ms(lambda: gemm(a, b))
        results.append(
            {
                "test_case_id": name,
                "execution_time_ms": latency,
                "params": {"M": m, "N": n, "K": k, "dtype": "fp16"},
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
