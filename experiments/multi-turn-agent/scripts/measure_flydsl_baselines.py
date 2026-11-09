#!/usr/bin/env python3
"""Measure real FlyDSL baseline kernel performance in aiter environment.

Run inside aiter-baseline-bench container:
  docker exec -e HIP_VISIBLE_DEVICES=2 aiter-baseline-bench \
    python3 /home/danyzhan/Lumen-RL/experiments/multi-tune-agent/scripts/measure_flydsl_baselines.py

For each flydsl held-out task:
1. Load the original aiter kernel via aiter package import
2. Load op_harness from the task's scripts/ to get standard test inputs
3. Run the kernel with profiler (warmup=360, rep=50, median)
4. Output per-task measured baseline to JSON
"""
import importlib
import importlib.util
import inspect
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, '/home/jimguo12/aiter')

import torch
import torch.profiler

HELD_OUT = Path("/home/danyzhan/agent-phase2-held-out-aiter")
ARTIFACTS = HELD_OUT / "artifacts" / "kernel"
OUTPUT = HELD_OUT / "receipts" / "flydsl_measured_baselines.json"

N_WARMUP = 360
N_REP = 50


def profiler_bench(func, warmup=N_WARMUP, rep=N_REP):
    try:
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
    except Exception as e:
        return 0.0, f"warmup error: {e}"

    times = []
    for _ in range(rep):
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False,
            ) as prof:
                func()
                torch.cuda.synchronize()
            us = sum(e.device_time_total for e in prof.key_averages()
                     if e.device_time_total > 0)
            if us > 0:
                times.append(us / 1000.0)
        except Exception:
            pass
    if not times:
        return 0.0, "no valid timing"
    times.sort()
    return times[len(times) // 2], None


def load_op_harness(task_dir):
    scripts = task_dir / "initial" / "scripts"
    harness_path = scripts / "op_harness.py"
    if not harness_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("op_harness", str(harness_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["op_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_task_config(task_dir):
    import yaml
    cfg_path = task_dir / "initial" / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {}


def load_metadata(task_dir):
    meta_path = task_dir / "initial" / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def measure_via_perf_runner(task_dir, task_id):
    """Use the task's own perf_runner.py on the original kernel."""
    initial = task_dir / "initial"
    kernel_py = initial / "kernel.py"
    perf_runner = initial / "scripts" / "perf_runner.py"
    if not perf_runner.exists() or not kernel_py.exists():
        return 0.0, "perf_runner or kernel.py missing"

    import subprocess
    cfg = load_task_config(task_dir)
    operator = cfg.get("operator", "unknown")

    r = subprocess.run(
        [sys.executable, str(perf_runner), str(kernel_py), operator],
        capture_output=True, text=True, timeout=300,
        cwd=str(initial),
    )
    output = r.stdout + r.stderr
    perf_re = re.compile(r'Perf:\s*([0-9]+(?:\.[0-9]+)?)\s*ms')
    match = perf_re.search(output)
    if match:
        ms = float(match.group(1))
        return ms, None
    return 0.0, f"no Perf line in output: {output[:500]}"


def measure_via_aiter_import(task_dir, task_id):
    """Import the original kernel via aiter package and benchmark."""
    meta = load_metadata(task_dir)
    source = meta.get("source", "")
    if not source:
        return 0.0, "no source in metadata"

    # Convert source path to module path
    # e.g. "aiter/ops/flydsl/kernels/buffer_ops.py" -> "aiter.ops.flydsl.kernels.buffer_ops"
    mod_path = source.replace("/", ".").replace(".py", "")
    # Remove ::function_name suffix if present
    if "::" in mod_path:
        mod_path = mod_path.split("::")[0]

    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        return 0.0, f"import {mod_path} failed: {e}"

    # Find callable functions
    fns = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith('_'):
            continue
        fns.append((name, obj))

    build_fns = [(n, f) for n, f in fns
                 if re.match(r'(build_|launch_|compile_|flydsl_)', n)]

    # Load op_harness
    harness_mod = load_op_harness(task_dir)
    cfg = load_task_config(task_dir)
    operator = cfg.get("operator", "unknown")

    harness = None
    if harness_mod:
        try:
            harness = harness_mod.get_harness(operator)
        except Exception:
            pass

    # Try build_* functions first (FlyDSL pattern)
    for fn_name, build_fn in build_fns:
        try:
            sig = inspect.signature(build_fn)
            kwargs = {}
            for pname, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    kwargs[pname] = param.default
            result = build_fn(**kwargs)
            if callable(result):
                args = []
                if harness:
                    try:
                        args = harness.make_launch_args(result)
                    except Exception:
                        try:
                            args = harness.make_args(result)
                        except Exception:
                            pass
                ms, err = profiler_bench(
                    lambda f=result, a=args: f(*a) if a else f()
                )
                if ms > 0:
                    return ms, None
        except Exception as e:
            continue

    # Try regular wrapper functions
    target_fns = build_fns if build_fns else fns
    for fn_name, fn in target_fns[:3]:
        if not harness:
            continue
        try:
            args = harness.make_args(fn)
            ms, err = profiler_bench(lambda f=fn, a=args: f(*a))
            if ms > 0:
                return ms, None
        except Exception:
            continue

    return 0.0, f"no callable function benchmarked. fns={[n for n,_ in fns]}"


def main():
    # Find all flydsl tasks
    flydsl_tasks = []
    for task_dir in sorted(ARTIFACTS.iterdir()):
        if not task_dir.is_dir():
            continue
        cfg = load_task_config(task_dir)
        if cfg.get("language") == "flydsl":
            flydsl_tasks.append((task_dir.name, task_dir))

    print(f"Found {len(flydsl_tasks)} flydsl tasks")
    results = {}

    for task_id, task_dir in flydsl_tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task_id}")

        meta = load_metadata(task_dir)
        source = meta.get("source", "?")
        old_baseline = meta.get("baseline_performance", {}).get("geomean_ms", 0)
        print(f"  Source: {source}")
        print(f"  Old baseline: {old_baseline:.6f} ms")

        # Method 1: Try perf_runner on the original kernel
        ms1, err1 = measure_via_perf_runner(task_dir, task_id)
        print(f"  perf_runner: {ms1:.6f} ms (err={err1})")

        # Method 2: Try aiter import path
        ms2, err2 = measure_via_aiter_import(task_dir, task_id)
        print(f"  aiter_import: {ms2:.6f} ms (err={err2})")

        # Use best non-zero measurement
        measured = ms1 if ms1 > 0 else ms2
        method = "perf_runner" if ms1 > 0 else ("aiter_import" if ms2 > 0 else "failed")

        cfg = load_task_config(task_dir)
        cases = cfg.get("benchmark_shapes", [])
        case_id = cases[0].get("case_id", task_id) if cases else task_id

        results[task_id] = {
            "task_id": task_id,
            "case_id": case_id,
            "source": source,
            "old_baseline_ms": old_baseline,
            "measured_ms": measured,
            "method": method,
            "perf_runner_ms": ms1,
            "perf_runner_err": err1,
            "aiter_import_ms": ms2,
            "aiter_import_err": err2,
            "ratio_old_new": old_baseline / measured if measured > 0 else None,
        }
        print(f"  RESULT: {measured:.6f} ms via {method} (old={old_baseline:.6f}, ratio={old_baseline/measured:.2f}x)" if measured > 0 else f"  RESULT: FAILED")

    # Write results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results written to {OUTPUT}")

    # Summary
    measured = [r for r in results.values() if r["measured_ms"] > 0]
    failed = [r for r in results.values() if r["measured_ms"] == 0]
    print(f"\nSummary: {len(measured)}/{len(results)} measured, {len(failed)} failed")
    for r in measured:
        ratio = r["ratio_old_new"]
        flag = " *** SUSPICIOUS" if ratio and (ratio > 5 or ratio < 0.2) else ""
        print(f"  {r['task_id']}: old={r['old_baseline_ms']:.4f} new={r['measured_ms']:.4f} ratio={ratio:.2f}x{flag}")


if __name__ == "__main__":
    main()
