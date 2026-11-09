"""Generic kernel performance runner.

Runs kernel.py as a subprocess and parses timing output.
Falls back to importing the module and running discovered benchmark functions.

Output format: "Perf: X.XXXX ms (case_name)" — one line per test case.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


_TIMING_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*ms\s*(?:per|/)\s*\w+|"
    r"Perf:\s*([0-9]+(?:\.[0-9]+)?)\s*ms|"
    r"GEAK_RESULT_LATENCY_MS\s*=\s*([0-9]+(?:\.[0-9]+)?)|"
    r"elapsed[_ ]time[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms|"
    r"latency[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms|"
    r"([0-9]+\.[0-9]{2,})\s*ms",
    re.IGNORECASE,
)


def _parse_any_timing(output: str) -> list[float]:
    """Extract timing values (ms) from any format of output."""
    values = []
    for m in _TIMING_RE.finditer(output):
        for g in m.groups():
            if g is not None:
                try:
                    v = float(g)
                    if 0.0001 < v < 100000:
                        values.append(v)
                except ValueError:
                    pass
                break
    return values


def benchmark_via_subprocess(
    kernel_path: str,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Run kernel.py as a subprocess and extract timing from stdout."""
    path = Path(kernel_path)
    if not path.exists():
        return []

    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(path.parent),
        )
    except subprocess.TimeoutExpired:
        return [{"test_case_id": "timeout", "execution_time_ms": 0.0}]
    except Exception as e:
        return [{"test_case_id": "error", "execution_time_ms": 0.0,
                 "error": str(e)[:300]}]

    combined = proc.stdout + "\n" + proc.stderr
    values = _parse_any_timing(combined)
    if not values:
        return []

    results = []
    for i, ms in enumerate(values):
        results.append({
            "test_case_id": f"case_{i}" if len(values) > 1 else "default",
            "execution_time_ms": ms,
        })
    return results


def benchmark_via_import(
    kernel_path: str,
    n_warmup: int = 10,
    n_iter: int = 100,
) -> list[dict[str, Any]]:
    """Import kernel module, find wrapper functions, run __main__-style benchmark."""
    import torch

    path = Path(kernel_path).resolve()
    if not path.exists():
        return []

    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("kernel_bench", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return []
    finally:
        sys.path = old_path

    bench_fn = getattr(mod, "benchmark", None) or getattr(mod, "run_benchmark", None)
    if bench_fn and callable(bench_fn):
        try:
            result = bench_fn()
            if isinstance(result, (int, float)) and result > 0:
                return [{"test_case_id": "default", "execution_time_ms": float(result)}]
            if isinstance(result, list):
                return result
        except Exception:
            pass

    return []


def run_performance(kernel_path: str, timeout: int = 120) -> list[dict[str, Any]]:
    """Main entry: try subprocess first, fall back to import-based."""
    results = benchmark_via_subprocess(kernel_path, timeout)
    if results and any(r.get("execution_time_ms", 0) > 0 for r in results):
        return results

    try:
        results = benchmark_via_import(kernel_path)
        if results and any(r.get("execution_time_ms", 0) > 0 for r in results):
            return results
    except Exception:
        pass

    return [{"test_case_id": "default", "execution_time_ms": 0.0,
             "error": "could not extract timing from kernel"}]


def main():
    """CLI: python kernel_benchmarker.py <kernel.py>"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    results = run_performance(args.kernel_path, args.timeout)
    for r in results:
        if r.get("error"):
            print(f"ERROR: {r['error']}", file=sys.stderr)
        ms = r.get("execution_time_ms", 0.0)
        cid = r.get("test_case_id", "unknown")
        print(f"Perf: {ms:.4f} ms ({cid})")


if __name__ == "__main__":
    main()
