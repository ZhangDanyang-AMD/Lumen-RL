#!/usr/bin/env python3
"""Compile, validate, and benchmark the fused-attention task."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)

from kernel import fused_attention  # noqa: E402


CASES = [
    ("short_prefill", 1, 8, 128, 64),
    ("medium_prefill", 2, 16, 256, 64),
    ("long_prefill", 1, 16, 512, 64),
]


def make_case(index):
    _, batch, heads, sequence, dim = CASES[index]
    torch.manual_seed(200 + index)
    shape = (batch, heads, sequence, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.25
    return q, k, v


def time_ms(fn, warmup=10, repeats=30):
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
    q, k, v = make_case(0)
    fused_attention(q, k, v)
    torch.cuda.synchronize()
    print("Compilation: PASS")


def check_correctness():
    for index, (name, _, _, _, _) in enumerate(CASES):
        q, k, v = make_case(index)
        actual = fused_attention(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print("Correctness: PASS (%s)" % name)


def benchmark():
    results = []
    for index, (name, batch, heads, sequence, dim) in enumerate(CASES):
        q, k, v = make_case(index)
        latency = time_ms(lambda: fused_attention(q, k, v))
        results.append(
            {
                "test_case_id": name,
                "execution_time_ms": latency,
                "params": {
                    "B": batch,
                    "H": heads,
                    "S": sequence,
                    "D": dim,
                    "causal": True,
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
