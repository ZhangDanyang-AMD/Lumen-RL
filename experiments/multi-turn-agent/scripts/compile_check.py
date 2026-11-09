#!/usr/bin/env python3
"""Real compile check: import module, find wrapper, call with op_harness args.

For Triton/FlyDSL, this triggers actual JIT/compilation rather than just
checking syntax or import. For Opus, runs hipcc.

Exit code 0 = compile pass, 1 = compile fail.
Stdout contains PASS/FAIL message with error details for model debugging.
"""
import importlib.util
import inspect
import subprocess
import sys
import traceback
from pathlib import Path

task_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
scripts_dir = task_dir / "scripts"
sys.path.insert(0, str(scripts_dir))

import yaml as _yaml

cfg = _yaml.safe_load((task_dir / "config.yaml").read_text())
language = cfg.get("language", "triton")
operator = cfg.get("operator", "unknown")

kernel_py = task_dir / "kernel.py"
kernel_cu = task_dir / "kernel.cu"


AITER_INCLUDE_DIRS = [
    "/home/jimguo12/aiter/csrc/include",
    "/home/jimguo12/aiter/csrc/opus_gemm/include",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx950",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942/a8w8",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942/a16w16",
    "/home/jimguo12/aiter/csrc/opus_moe/include",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950/a8w4",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950/a16w16",
    "/home/jimguo12/aiter/csrc/kernels/rope",
    "/home/jimguo12/aiter/csrc/cpp_itfs/pa",
    "/home/jimguo12/aiter/csrc/cpp_itfs",
    "/opt/venv/lib/python3.12/site-packages/torch/include",
    "/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
]


def _opus_compile():
    if not kernel_cu.exists():
        print("FAIL: kernel.cu missing")
        return 1
    include_dir = task_dir / "include"
    inc_flags = ["-I", str(include_dir)] if include_dir.is_dir() else []
    for d in AITER_INCLUDE_DIRS:
        if Path(d).is_dir():
            inc_flags.extend(["-I", d])
    r = subprocess.run(
        ["hipcc", "--genco", "-O3", "--offload-arch=gfx950"] + inc_flags +
        [str(kernel_cu), "-o", "/dev/null"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        print(f"FAIL: hipcc compile error\n{r.stderr[:1500]}")
        return 1
    print("PASS: compile (hipcc)")
    return 0


def _python_compile():
    """Triton or FlyDSL: import + find wrapper + call with op_harness args."""
    if not kernel_py.exists():
        print("FAIL: kernel.py missing")
        return 1

    # Step 1: Import the module
    try:
        spec = importlib.util.spec_from_file_location("kernel", str(kernel_py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        tb = traceback.format_exc()
        print(f"FAIL: kernel.py import error\n{tb[-1500:]}")
        return 1

    # Step 2: Find public wrapper functions
    fns = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_") or name == "main":
            continue
        if getattr(obj, "__module__", None) != "kernel":
            continue
        fns.append((name, obj))

    # For FlyDSL, prefer build_*/launch_*/compile_* functions
    build_fns = [
        (n, f) for n, f in fns
        if n.startswith("build_") or n.startswith("launch_") or n.startswith("compile_")
    ]

    target_fns = build_fns if build_fns else fns
    if not target_fns:
        print("FAIL: no public wrapper function found in kernel.py")
        return 1

    # Step 3: Build args from op_harness and call the function
    import torch
    from op_harness import get_harness

    harness = get_harness(operator)
    name, fn = target_fns[0]
    try:
        args = harness.make_args(fn)
        kwargs = {}
    except Exception:
        tb = traceback.format_exc()
        print(f"FAIL: op_harness.make_args failed for {name}()\n{tb[-1500:]}")
        return 1

    try:
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
    except Exception:
        tb = traceback.format_exc()
        print(f"FAIL: {name}() execution error (Triton JIT or runtime)\n{tb[-1500:]}")
        return 1

    print(f"PASS: compile ({name}() executed successfully)")
    return 0


if language == "opus":
    sys.exit(_opus_compile())
else:
    sys.exit(_python_compile())
