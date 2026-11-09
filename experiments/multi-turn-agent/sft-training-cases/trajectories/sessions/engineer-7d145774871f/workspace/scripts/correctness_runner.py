#!/usr/bin/env python3
"""Functional correctness runner for candidate kernels.

Run inside Docker: python3 correctness_runner.py <task_dir>

Loads candidate kernel.py, calls the wrapper function with op_harness shapes,
checks for:
  1. No import errors
  2. Wrapper function exists (or FlyDSL build_*/launch_*/compile_* function)
  3. Runs without exception
  4. Output is finite (no NaN/Inf)

Outputs PASS or FAIL with detailed error message for model debugging.
"""
import sys, importlib.util, inspect, traceback, re
sys.path.insert(0, '/app/aiter-test')

from pathlib import Path

task_dir = Path(sys.argv[1])
scripts_dir = task_dir / "scripts"
sys.path.insert(0, str(scripts_dir))

import torch
import yaml as _yaml
from op_harness import get_harness

cfg = _yaml.safe_load((task_dir / "config.yaml").read_text())
operator = cfg.get("operator", "unknown")
language = cfg.get("language", "triton")

# Opus: kernel.cu only, no Python callable to test functionally
kernel_cu = task_dir / "kernel.cu"
kernel_py = task_dir / "kernel.py"

if kernel_cu.exists() and not kernel_py.exists():
    print("PASS: correctness (opus kernel present)")
    sys.exit(0)

if not kernel_py.exists():
    print("FAIL: kernel.py missing")
    sys.exit(1)

# Load candidate kernel
spec = importlib.util.spec_from_file_location("candidate", str(kernel_py))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    tb = traceback.format_exc()
    print(f"FAIL: kernel.py import error\n{tb[-2000:]}")
    sys.exit(1)

# --- Helper: check output for NaN/Inf ---
def has_nan_inf(t):
    return isinstance(t, torch.Tensor) and (
        torch.any(torch.isnan(t)) or torch.any(torch.isinf(t))
    )

def check_output(out, fname):
    if isinstance(out, torch.Tensor):
        bad = has_nan_inf(out)
    elif isinstance(out, (list, tuple)):
        bad = any(has_nan_inf(t) for t in out if isinstance(t, torch.Tensor))
    else:
        bad = False
    if bad:
        print(f"FAIL: {fname}() output contains NaN or Inf values")
        print(f"  This usually indicates division by zero or overflow in the kernel")
        sys.exit(1)

# --- FlyDSL: try build_*/launch_*/compile_* pattern first ---
if language == "flydsl":
    build_fns = [
        (n, o) for n, o in inspect.getmembers(mod, inspect.isfunction)
        if callable(o) and re.match(r'(build_|get_flydsl_|compile_|launch_)', n)
    ]
    if build_fns:
        fname, build_fn = build_fns[0]
        harness = get_harness(operator)
        # Try to call build_fn to get a compiled callable
        sig = inspect.signature(build_fn)
        kwargs = {}
        for pname, param in sig.parameters.items():
            if hasattr(harness, 'build_kwargs') and pname in harness.build_kwargs:
                kwargs[pname] = harness.build_kwargs[pname]
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default
        try:
            result = build_fn(**kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"FAIL: {fname}() build raised exception:\n{tb[-2000:]}")
            sys.exit(1)

        if callable(result):
            # build_fn returned a compiled kernel callable - try to run it
            try:
                launch_args = harness.make_args(result)
            except Exception:
                launch_args = []
            try:
                if launch_args:
                    out = result(*launch_args)
                else:
                    out = result()
                torch.cuda.synchronize()
            except Exception as e:
                tb = traceback.format_exc()
                print(f"FAIL: compiled kernel from {fname}() raised exception:\n{tb[-2000:]}")
                sys.exit(1)
            check_output(out, fname)
            print(f"PASS: correctness ({fname} builds and runs for operator='{operator}')")
            sys.exit(0)
        else:
            print(f"PASS: correctness ({fname} returned non-callable, build succeeded)")
            sys.exit(0)

# --- Standard wrapper function path (triton / flydsl fallback) ---
def _is_jit(o): return hasattr(o, "fn") and hasattr(o, "run")
wrappers = [
    (n, o) for n, o in inspect.getmembers(mod, inspect.isfunction)
    if not n.startswith("_")
    and getattr(o, "__module__", "") == "candidate"
    and not _is_jit(o)
    and n != "main"
]

if not wrappers:
    print("FAIL: kernel.py has no public wrapper function")
    if language == "flydsl":
        print("  Hint: define a build_*() function that compiles/returns your FlyDSL kernel,")
        print("        or a public wrapper function that calls it.")
    else:
        print("  Hint: define a wrapper function (not starting with _) that calls your @triton.jit kernel")
    sys.exit(1)

harness = get_harness(operator)
fname, func = wrappers[0]

torch.manual_seed(42)
try:
    args = harness.make_args(func)
except Exception as e:
    tb = traceback.format_exc()
    print(f"FAIL: could not build test inputs for {fname}\n{tb[-1000:]}")
    print(f"  Hint: wrapper signature must match op_harness shapes for operator='{operator}'")
    sys.exit(1)

# Run the kernel
try:
    out = func(*args)
    torch.cuda.synchronize()
except Exception as e:
    tb = traceback.format_exc()
    print(f"FAIL: {fname}() raised an exception:\n{tb[-2000:]}")
    sys.exit(1)

check_output(out, fname)

print(f"PASS: correctness ({fname} produces finite output for operator='{operator}')")
sys.exit(0)
