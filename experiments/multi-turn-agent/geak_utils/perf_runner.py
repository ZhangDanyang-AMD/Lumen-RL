#!/usr/bin/env python3
"""Standalone GPU performance benchmarker using torch.profiler.

Usage:
    python3 perf_runner.py <kernel_path> [operator_name]

Outputs "Perf: X.XXXX ms (case_name)" lines — one per wrapper function.
GPU time only (device_time_total from torch.profiler CUDA activities).
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import torch
import torch.profiler


def _is_jit(obj: object) -> bool:
    return hasattr(obj, "fn") and hasattr(obj, "run")


def _profile(func, args, n_warmup: int = 5, n_iter: int = 20) -> float:
    """Return per-call GPU kernel time in ms (device_time_total only)."""
    try:
        for _ in range(n_warmup):
            func(*args)
        torch.cuda.synchronize()
    except Exception:
        return 0.0

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(n_iter):
                func(*args)
            torch.cuda.synchronize()
    except Exception:
        return 0.0

    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    if not events:
        return 0.0
    total_us = sum(e.device_time_total for e in events)
    return total_us / 1000.0 / n_iter


def _load_module(kernel_path: str):
    spec = importlib.util.spec_from_file_location("kernel", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_operator(kernel_path: str) -> str:
    """Guess operator from config.yaml next to kernel_path."""
    try:
        task_dir = Path(kernel_path).parent
        for name in ("config.yaml", "config.yml"):
            cfg = task_dir / name
            if not cfg.exists():
                cfg = task_dir.parent / name
            if cfg.exists():
                import yaml
                data = yaml.safe_load(cfg.read_text())
                return data.get("operator", "unknown")
    except Exception:
        pass
    return "unknown"


def _find_wrappers(mod) -> list:
    """Return non-JIT public functions defined in the module."""
    return [
        (name, obj)
        for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if not name.startswith("_")
        and obj.__module__ == "kernel"
        and not _is_jit(obj)
    ]


def main():
    kernel_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not kernel_path:
        print("usage: perf_runner.py <kernel.py> [operator]", file=sys.stderr)
        sys.exit(1)

    # Optional operator override from argv
    operator = sys.argv[2] if len(sys.argv) > 2 else _get_operator(kernel_path)

    # Load op harness (op_harness.py lives next to this file)
    try:
        harness_dir = str(Path(__file__).parent)
        if harness_dir not in sys.path:
            sys.path.insert(0, harness_dir)
        from op_harness import get_harness
        harness = get_harness(operator)
    except Exception:
        harness = None

    # Load kernel module
    try:
        mod = _load_module(kernel_path)
    except Exception as e:
        print(f"Perf: 0.000 ms (import_error)", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(0)

    wrappers = _find_wrappers(mod)

    if not wrappers:
        print("Perf: 0.000 ms (no_wrapper)")
        sys.exit(0)

    for fname, func in wrappers:
        # Try harness first (operator-aware shapes), then generic fallback
        if harness is not None:
            try:
                args = harness.make_args(func)
                ms = _profile(func, args)
                if ms > 0:
                    print(f"Perf: {ms:.4f} ms ({fname})")
                    continue
            except Exception:
                pass

        # Generic fallback: try empty args, then shape-guessed args
        for arg_set in [[], _generic_args(func)]:
            try:
                ms = _profile(func, arg_set)
                if ms > 0:
                    print(f"Perf: {ms:.4f} ms ({fname})")
                    break
            except Exception:
                pass
        else:
            print(f"Perf: 0.000 ms ({fname}_failed)")


def _generic_args(func) -> list:
    """Last-resort argument builder using name heuristics."""
    sig = inspect.signature(func)
    args = []
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty:
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        nl = p.name.lower()
        if nl in ("m", "n", "k", "size", "dim", "batch_size", "seq_len",
                  "seqlen", "num_heads", "head_dim", "num_experts", "top_k"):
            args.append(128)
        elif nl in ("is_causal", "causal", "training", "use_bias"):
            args.append(False)
        elif nl in ("scale", "alpha", "beta", "eps", "epsilon", "softmax_scale"):
            args.append(1.0)
        elif nl in ("cu_seqlens_q", "cu_seqlens_k", "cu_seqlens"):
            args.append(None)
        else:
            args.append(torch.randn(128, 256, dtype=torch.bfloat16, device="cuda"))
    return args


if __name__ == "__main__":
    main()
