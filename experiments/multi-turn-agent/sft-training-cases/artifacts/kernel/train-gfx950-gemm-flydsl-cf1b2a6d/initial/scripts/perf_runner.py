#!/usr/bin/env python3
"""Candidate kernel performance benchmarker with FlyDSL tile config search.

For FlyDSL kernels:
  - Finds build_*/launch_*/compile_* functions
  - Sweeps tunable tile params (tile_m, tile_n, BM, TILE_M, split_k, block_threads)
  - Picks best config, profiles with torch.profiler
  - Outputs "Perf: X.XXXX ms (case_id)" matching baseline case_ids

For Triton kernels:
  - Finds wrapper functions, calls with harness args
  - 360 warmup runs trigger @triton.autotune to find best config
"""
import sys, os, json, importlib.util, inspect, re
from pathlib import Path

import torch
import torch.profiler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, '/app/aiter-test')

N_WARMUP = 360
N_REP    = 50
N_TUNE_W = 5    # warmup per tile config during search
N_TUNE_I = 3    # timed iters per config

def _profiler_bench(func, warmup=N_WARMUP, rep=N_REP):
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


def _quick_bench(func):
    """Fast measurement for config search (CUDA events)."""
    try:
        for _ in range(N_TUNE_W): func()
        torch.cuda.synchronize()
    except Exception:
        return float('inf')
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(N_TUNE_I):
        try:
            start.record(); func(); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        except Exception:
            pass
    return sorted(times)[len(times)//2] if times else float('inf')


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


def _find_build_fns(mod):
    """Return FlyDSL build_*/launch_*/compile_* functions."""
    return [
        (name, obj)
        for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if callable(obj) and re.match(r'(build_|get_flydsl_|compile_|launch_)', name)
    ]


# FlyDSL tile param search grids
TILE_GRIDS = {
    'tile_m_n': [(64,64), (64,128), (128,64), (128,128), (128,256), (256,128)],
    'BM':       [32, 64, 128, 256],
    'TILE_MN':  [(32,32), (64,64), (64,128), (128,64), (128,128)],
    'split_k':  [1, 2, 4, 8],
    'block_threads': [64, 128, 256],
}


def _try_build(build_fn, harness, extra_kwargs):
    """Call build_* with harness kwargs + extra, return callable or None."""
    sig = inspect.signature(build_fn)
    kwargs = {}
    for pname, param in sig.parameters.items():
        if pname in extra_kwargs:
            kwargs[pname] = extra_kwargs[pname]
        elif harness and hasattr(harness, 'build_kwargs') and pname in harness.build_kwargs:
            kwargs[pname] = harness.build_kwargs[pname]
        elif param.default is not inspect.Parameter.empty:
            kwargs[pname] = param.default
    try:
        result = build_fn(**kwargs)
        return result if callable(result) else None
    except Exception:
        return None


def _tune_flydsl_build(build_fn, harness, case_id):
    """Search over tile param configs for a FlyDSL build_* function."""
    sig = inspect.signature(build_fn)
    params = list(sig.parameters.keys())

    has_tile_mn  = 'tile_m' in params and 'tile_n' in params
    has_BM       = 'BM' in params and 'tile_m' not in params
    has_TILE_MN  = 'TILE_M' in params and 'tile_m' not in params
    has_split_k  = 'split_k' in params
    has_block_th = 'block_threads' in params

    best_ms  = float('inf')
    best_cfg = {}

    def try_cfg(extra):
        fn = _try_build(build_fn, harness, extra)
        if fn is None:
            return float('inf')
        args = []
        if harness:
            try:
                args = harness.make_launch_args(fn)
            except Exception:
                args = []
        try:
            return _quick_bench(lambda f=fn, a=args: f(*a) if a else f())
        except Exception:
            return float('inf')

    # Baseline (defaults)
    ms0 = try_cfg({})
    if ms0 < best_ms:
        best_ms = ms0
        best_cfg = {}

    # Sweep tile params
    if has_tile_mn:
        for tm, tn in TILE_GRIDS['tile_m_n']:
            ms = try_cfg({'tile_m': tm, 'tile_n': tn})
            if ms < best_ms:
                best_ms = ms
                best_cfg = {'tile_m': tm, 'tile_n': tn}

    if has_BM:
        for bm in TILE_GRIDS['BM']:
            ms = try_cfg({'BM': bm})
            if ms < best_ms:
                best_ms = ms
                best_cfg = {'BM': bm}

    if has_TILE_MN:
        for tm, tn in TILE_GRIDS['TILE_MN']:
            ms = try_cfg({'TILE_M': tm, 'TILE_N': tn})
            if ms < best_ms:
                best_ms = ms
                best_cfg = {'TILE_M': tm, 'TILE_N': tn}

    if has_split_k:
        for sk in TILE_GRIDS['split_k']:
            ms = try_cfg({'split_k': sk})
            if ms < best_ms:
                best_ms = ms
                best_cfg = {'split_k': sk}

    if has_block_th:
        for bt in TILE_GRIDS['block_threads']:
            ms = try_cfg({'block_threads': bt})
            if ms < best_ms:
                best_ms = ms
                best_cfg = {'block_threads': bt}

    if best_ms == float('inf'):
        return 0.0, {}

    # Final accurate measurement with best config
    fn_best = _try_build(build_fn, harness, best_cfg)
    if fn_best is None:
        return 0.0, {}
    args = []
    if harness:
        try:
            args = harness.make_launch_args(fn_best)
        except Exception:
            args = []
    final_ms = _profiler_bench(
        lambda f=fn_best, a=args: f(*a) if a else f()
    )
    return final_ms, best_cfg


def main():
    kernel_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not kernel_path:
        print("usage: perf_runner.py <kernel.py> [operator]", file=sys.stderr)
        sys.exit(1)

    config_path = ROOT / "config.yaml"
    case_id = None
    operator = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    language = "triton"

    if config_path.is_file():
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        cases = cfg.get("benchmark_shapes", [])
        if cases:
            case_id = cases[0].get("case_id", None)
        if not operator or operator == "unknown":
            operator = cfg.get("operator", "unknown")
        language = cfg.get("language", "triton")

    # Load op harness
    try:
        from op_harness import get_harness
        harness = get_harness(operator)
    except Exception:
        harness = None

    # Load kernel module
    try:
        mod = _load_kernel(kernel_path)
    except Exception as e:
        print(f"Perf: 0.0000 ms ({case_id or 'import_error'})")
        sys.exit(0)

    # ── FlyDSL: use build_* with tile search ─────────────────────────────────
    if language == "flydsl":
        build_fns = _find_build_fns(mod)
        if build_fns:
            fn_name, build_fn = build_fns[0]
            ms, best_cfg = _tune_flydsl_build(build_fn, harness, case_id)
            if ms > 0:
                cfg_str = str(best_cfg) if best_cfg else 'default'
                out_name = case_id if case_id else fn_name
                print(f"Perf: {ms:.6f} ms ({out_name}) [cfg={cfg_str}]")
                sys.exit(0)
        print(f"Perf: 0.0000 ms ({case_id or 'no_build_fn'})")
        sys.exit(0)

    # ── Triton / Opus: use wrapper functions ──────────────────────────────────
    wrappers = _find_wrappers(mod)
    if not wrappers:
        print(f"Perf: 0.0000 ms ({case_id or 'no_wrapper'})")
        sys.exit(0)

    reported = False
    for fname, func in wrappers:
        if harness is not None:
            try:
                args = harness.make_args(func)
                ms = _profiler_bench(lambda f=func, a=args: f(*a))
                if ms > 0:
                    print(f"Perf: {ms:.6f} ms ({case_id if case_id else fname})")
                    reported = True
                    continue
            except Exception:
                pass
        try:
            ms = _profiler_bench(lambda f=func: f())
            if ms > 0:
                print(f"Perf: {ms:.6f} ms ({case_id if case_id else fname})")
                reported = True
        except Exception:
            pass

    if not reported:
        print(f"Perf: 0.0000 ms ({case_id or 'failed'})")


if __name__ == "__main__":
    main()
