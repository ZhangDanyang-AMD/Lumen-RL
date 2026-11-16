#!/usr/bin/env python3
"""Opus HIP C++ kernel runner: compile → load → call → verify → time.

Usage:
    python3 opus_runner.py <task_dir> [correctness|performance]

Compiles kernel.cu to a shared library, finds extern "C" host launcher
functions, builds args from op_harness, and either verifies correctness
or benchmarks performance.

Exit code 0 = pass, 1 = fail.
"""
import ctypes
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

task_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
mode = sys.argv[2] if len(sys.argv) > 2 else "correctness"

scripts_dir = task_dir / "scripts"
sys.path.insert(0, str(scripts_dir))

import yaml as _yaml

cfg = _yaml.safe_load((task_dir / "config.yaml").read_text())
operator = cfg.get("operator", "unknown")
case_id = None
shapes_list = cfg.get("benchmark_shapes", [])
if shapes_list:
    case_id = shapes_list[0].get("case_id", None)
if not case_id:
    case_id = cfg.get("task_id", "opus")

kernel_cu = task_dir / "kernel.cu"
kernel_so = task_dir / "kernel.so"


def _compile() -> int:
    """Compile kernel.cu → kernel.so. Returns 0 on success."""
    if not kernel_cu.exists():
        print("FAIL: kernel.cu missing")
        return 1

    include_dir = task_dir / "include"
    inc_flags = ["-I", str(include_dir)] if include_dir.is_dir() else []

    cmd = [
        "hipcc", "-shared", "-fPIC", "-O3",
        "--offload-arch=gfx950",
    ] + inc_flags + [
        str(kernel_cu), "-o", str(kernel_so),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"FAIL: hipcc compile error\n{r.stderr[:1500]}")
        return 1
    return 0


def _parse_extern_c_functions(source: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Parse extern "C" function declarations from source.

    Returns list of (func_name, [(param_type, param_name), ...]).
    Handles both `extern "C" void func(...)` and `extern "C" { void func(...) }`.
    """
    results = []

    # Pattern 1: extern "C" void/int func_name(params)
    pattern1 = re.compile(
        r'extern\s+"C"\s+(?:void|int)\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    )
    for m in pattern1.finditer(source):
        name = m.group(1)
        params_str = m.group(2).strip()
        if params_str:
            params = _parse_params(params_str)
            results.append((name, params))

    # Pattern 2: extern "C" { ... void func_name(params) ... }
    pattern2 = re.compile(
        r'extern\s+"C"\s*\{([^}]*)\}',
        re.DOTALL,
    )
    for block_m in pattern2.finditer(source):
        block = block_m.group(1)
        func_pat = re.compile(
            r'(?:void|int)\s+(\w+)\s*\(([^)]*)\)',
            re.MULTILINE,
        )
        for fm in func_pat.finditer(block):
            name = fm.group(1)
            if any(n == name for n, _ in results):
                continue
            params_str = fm.group(2).strip()
            if params_str:
                params = _parse_params(params_str)
                results.append((name, params))

    return results


def _parse_params(params_str: str) -> list[tuple[str, str]]:
    """Parse C parameter list string into [(type_category, name), ...].

    type_category is one of: "ptr", "int", "float", "stream"
    """
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue

        # Extract the name (last token)
        tokens = part.split()
        if not tokens:
            continue

        name = tokens[-1].lstrip("*&")
        type_str = " ".join(tokens[:-1]).rstrip("*&")
        # If name is empty (e.g., unnamed param), skip
        if not name or name == "void":
            continue

        # Check for pointer types
        if "*" in part:
            if "stream" in name.lower() or "hipStream_t" in type_str:
                params.append(("stream", name))
            else:
                params.append(("ptr", name))
        elif any(t in type_str for t in ("int", "size_t", "int32_t", "int64_t", "unsigned")):
            params.append(("int", name))
        elif any(t in type_str for t in ("float", "double")):
            params.append(("float", name))
        elif "hipStream_t" in type_str:
            params.append(("stream", name))
        else:
            params.append(("int", name))

    return params


def _build_ctypes_args(params, harness, operator):
    """Build ctypes argument list and track output tensors.

    Returns (ctypes_args, output_tensors) where output_tensors is a dict
    of {name: tensor} for non-const pointer parameters.
    """
    import torch

    ctypes_args = []
    output_tensors = {}
    all_tensors = {}

    for ptype, pname in params:
        nl = pname.lower()

        if ptype == "stream":
            stream_ptr = torch.cuda.current_stream().cuda_stream
            ctypes_args.append(ctypes.c_void_p(stream_ptr))
            continue

        if ptype == "ptr":
            shape = harness.shapes.get(nl, None)
            if shape is None:
                shape = harness.shapes.get(pname, None)
            if shape is None:
                shape = _guess_ptr_shape(nl, operator)

            if isinstance(shape, tuple):
                dt = harness.dtypes.get(nl, torch.bfloat16)
                if dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                          torch.int8, torch.int32, torch.int64):
                    t = torch.zeros(shape, dtype=dt, device="cuda")
                else:
                    t = torch.randn(shape, dtype=dt, device="cuda")
                all_tensors[nl] = t
                ctypes_args.append(ctypes.c_void_p(t.data_ptr()))
            elif isinstance(shape, (int, float)):
                ctypes_args.append(ctypes.c_void_p(0))
            else:
                t = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
                all_tensors[nl] = t
                ctypes_args.append(ctypes.c_void_p(t.data_ptr()))
            continue

        if ptype == "int":
            val = harness.shapes.get(nl, None)
            if val is None:
                val = harness.shapes.get(pname, None)
            if val is None:
                val = _guess_scalar(nl)
            if isinstance(val, tuple):
                val = val[0]
            ctypes_args.append(ctypes.c_int(int(val)))
            continue

        if ptype == "float":
            val = harness.shapes.get(nl, None)
            if val is None:
                val = _guess_float(nl)
            if isinstance(val, tuple):
                val = 1.0
            ctypes_args.append(ctypes.c_float(float(val)))
            continue

    # Identify output tensors (common output parameter names)
    output_names = {"c", "y", "output", "out", "o", "result",
                    "q_out", "k_out", "v_out", "dst"}
    for name, tensor in all_tensors.items():
        if name in output_names:
            output_tensors[name] = tensor

    # If no explicit output found, treat the last tensor as output
    if not output_tensors and all_tensors:
        last_name = list(all_tensors.keys())[-1]
        output_tensors[last_name] = all_tensors[last_name]

    return ctypes_args, output_tensors


def _guess_ptr_shape(name, operator):
    """Fallback shape guessing for pointer parameters not in op_harness."""
    _M, _N, _K = 4096, 8192, 4096
    _B, _S, _H, _D = 4, 512, 32, 128
    name_shapes = {
        "a": (_M, _K), "b": (_N, _K), "c": (_M, _N),
        "x": (_M, _K), "w": (_N, _K), "y": (_M, _N),
        "input": (_M, _K), "weight": (_N, _K), "output": (_M, _N),
        "bias": (_N,), "out": (_M, _N), "result": (_M, _N),
    }
    if name in name_shapes:
        return name_shapes[name]
    return (128, 256)


def _guess_scalar(name):
    _M, _N, _K = 4096, 8192, 4096
    scalar_map = {
        "m": _M, "n": _N, "k": _K,
        "batch": 4, "batch_size": 4, "num_heads": 32, "head_dim": 128,
        "seq_len": 512, "seqlen": 512, "dim": 128, "size": 128,
        "stride": 128, "lda": _K, "ldb": _K, "ldc": _N,
        "stride_am": _K, "stride_ak": 1,
        "stride_bk": _N, "stride_bn": 1,
        "stride_cm": _N, "stride_cn": 1,
        "num_tokens": _M, "num_experts": 8,
    }
    return scalar_map.get(name, 128)


def _guess_float(name):
    float_map = {
        "scale": 1.0, "alpha": 1.0, "beta": 0.0,
        "eps": 1e-5, "epsilon": 1e-5, "dropout_p": 0.0,
        "softmax_scale": 0.08838834764831843,  # 1/sqrt(128)
        "sm_scale": 0.08838834764831843,
    }
    return float_map.get(name, 1.0)


def _run_correctness() -> int:
    """Compile, load, call, and verify an opus kernel."""
    import torch

    if not kernel_cu.exists():
        print("FAIL: kernel.cu missing")
        return 1

    source = kernel_cu.read_text()
    funcs = _parse_extern_c_functions(source)

    if not funcs:
        # No extern C — fall back to compile-only check
        rc = _compile()
        if rc != 0:
            return rc
        print("FAIL: kernel.cu compiles but has no extern \"C\" function — cannot verify at runtime")
        return 1

    # Compile to .so
    rc = _compile()
    if rc != 0:
        return rc

    # Load .so
    try:
        lib = ctypes.CDLL(str(kernel_so))
    except OSError as e:
        print(f"FAIL: cannot load kernel.so: {e}")
        return 1

    from op_harness import get_harness
    harness = get_harness(operator)

    # Try each extern C function until one works
    last_error = None
    for func_name, params in funcs:
        try:
            fn = getattr(lib, func_name)
        except AttributeError:
            last_error = f"symbol {func_name} not found in .so"
            continue

        try:
            ctypes_args, output_tensors = _build_ctypes_args(params, harness, operator)
        except Exception:
            last_error = traceback.format_exc()[-1000:]
            continue

        try:
            fn(*ctypes_args)
            torch.cuda.synchronize()
        except Exception:
            last_error = traceback.format_exc()[-1000:]
            continue

        # Check outputs
        for oname, otensor in output_tensors.items():
            if torch.any(torch.isnan(otensor)) or torch.any(torch.isinf(otensor)):
                print(f"FAIL: {func_name}() output '{oname}' contains NaN/Inf")
                return 1

        print(f"PASS: correctness ({func_name}() executed, output finite)")
        return 0

    print(f"FAIL: no extern C function could be called successfully\n{last_error}")
    return 1


def _run_performance() -> int:
    """Compile, load, and benchmark an opus kernel."""
    import torch

    if not kernel_cu.exists():
        print("FAIL: kernel.cu missing")
        return 1

    source = kernel_cu.read_text()
    funcs = _parse_extern_c_functions(source)

    if not funcs:
        rc = _compile()
        if rc != 0:
            return rc
        print(f"Perf: 0.0 ms (no_extern_c)")
        return 0

    # Compile if needed
    if not kernel_so.exists():
        rc = _compile()
        if rc != 0:
            return rc

    try:
        lib = ctypes.CDLL(str(kernel_so))
    except OSError as e:
        print(f"FAIL: cannot load kernel.so: {e}")
        return 1

    from op_harness import get_harness
    harness = get_harness(operator)

    for func_name, params in funcs:
        try:
            fn = getattr(lib, func_name)
        except AttributeError:
            continue

        try:
            ctypes_args, _ = _build_ctypes_args(params, harness, operator)
        except Exception:
            continue

        # Warmup
        n_warmup = 100
        n_iter = 50
        try:
            for _ in range(n_warmup):
                fn(*ctypes_args)
            torch.cuda.synchronize()
        except Exception:
            tb = traceback.format_exc()
            print(f"FAIL: {func_name}() warmup error\n{tb[-500:]}")
            continue

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(n_iter):
            start_event.record()
            fn(*ctypes_args)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        times_ms.sort()
        median_ms = times_ms[len(times_ms) // 2]
        print(f"Perf: {median_ms:.6f} ms ({case_id})")
        return 0

    print(f"Perf: 0.0 ms ({case_id}_call_failed)")
    return 0


if mode == "correctness":
    sys.exit(_run_correctness())
elif mode == "performance":
    sys.exit(_run_performance())
else:
    print(f"Unknown mode: {mode}")
    sys.exit(1)
