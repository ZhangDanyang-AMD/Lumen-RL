#!/usr/bin/env python3
"""Candidate kernel performance benchmarker.

Reads benchmark_shapes from config.yaml, uses the same case_ids as the baseline,
and measures with torch.profiler (warmup=360, rep=50, median).

Outputs "Perf: X.XXXX ms (case_id)" lines matching the baseline case_ids.
"""
import sys, os, json, importlib.util, inspect
from pathlib import Path

import torch
import torch.profiler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, '/app/aiter-test')

N_WARMUP = 360
N_REP    = 50

def _profiler_bench(func, warmup=N_WARMUP, rep=N_REP):
    """warmup × func(), then rep independent profiler contexts → median device ms."""
    try:
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
    except Exception:
        return 0.0
    times = []
    for _ in range(rep):
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False,
            ) as prof:
                func()
                torch.cuda.synchronize()
            us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
            if us > 0:
                times.append(us / 1000.0)
        except Exception:
            pass
    if not times:
        return 0.0
    times.sort()
    return times[len(times) // 2]


def _load_kernel(kernel_path):
    spec = importlib.util.spec_from_file_location("kernel", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_wrappers(mod):
    """Return non-JIT public functions defined in this module."""
    return [
        (name, obj)
        for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if not name.startswith("_")
        and getattr(obj, "__module__", "") == "kernel"
        and not (hasattr(obj, "fn") and hasattr(obj, "run"))
    ]


def main():
    kernel_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not kernel_path:
        print("usage: perf_runner.py <kernel.py> [operator]", file=sys.stderr)
        sys.exit(1)

    # Load config.yaml to get case_id and benchmark_shapes
    config_path = ROOT / "config.yaml"
    case_id = None
    operator = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    if config_path.is_file():
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        cases = cfg.get("benchmark_shapes", [])
        if cases:
            case_id = cases[0].get("case_id", None)
        if not operator or operator == "unknown":
            operator = cfg.get("operator", "unknown")

    # Load op harness for fallback
    try:
        from op_harness import get_harness
        harness = get_harness(operator)
    except Exception:
        harness = None

    # Load kernel module
    try:
        mod = _load_kernel(kernel_path)
    except Exception as e:
        print(f"Perf: 0.0000 ms ({case_id or 'import_error'})", file=sys.stderr)
        sys.exit(0)

    wrappers = _find_wrappers(mod)

    if not wrappers:
        print(f"Perf: 0.0000 ms ({case_id or 'no_wrapper'})")
        sys.exit(0)

    reported = False
    for fname, func in wrappers:
        # Use harness to build args
        if harness is not None:
            try:
                args = harness.make_args(func)
                ms = _profiler_bench(lambda f=func, a=args: f(*a))
                if ms > 0:
                    out_name = case_id if case_id else fname
                    print(f"Perf: {ms:.6f} ms ({out_name})")
                    reported = True
                    continue
            except Exception:
                pass

        # Fallback: empty args
        try:
            ms = _profiler_bench(lambda f=func: f())
            if ms > 0:
                out_name = case_id if case_id else fname
                print(f"Perf: {ms:.6f} ms ({out_name})")
                reported = True
        except Exception:
            pass

    if not reported:
        print(f"Perf: 0.0000 ms ({case_id or 'failed'})")


if __name__ == "__main__":
    main()
