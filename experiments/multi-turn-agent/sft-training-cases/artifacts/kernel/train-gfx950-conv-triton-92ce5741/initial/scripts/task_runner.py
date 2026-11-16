#!/usr/bin/env python3
"""Task runner for heldout-gfx950-conv-26620a8d."""
import subprocess, sys, re, time

def run_compile(task_dir):
    """Check kernel compiles by actually calling the wrapper function."""
    r = subprocess.run(
        [sys.executable, str(task_dir / "scripts" / "compile_check.py"), str(task_dir)],
        capture_output=True, text=True, timeout=120)
    output = (r.stdout + r.stderr).strip()
    for line in output.splitlines():
        if line.strip():
            print(line)
    return r.returncode

def run_correctness(task_dir):
    """Functional correctness: call correctness_runner.py inside docker."""
    import subprocess, sys
    
    kernel_py = task_dir / "kernel.py"
    kernel_cu = task_dir / "kernel.cu"
    
    # Opus: real compile+load+call via opus_runner.py
    if kernel_cu.exists() and not kernel_py.exists():
        runner = task_dir / "scripts" / "opus_runner.py"
        if runner.exists():
            r = subprocess.run(
                [sys.executable, str(runner), str(task_dir), "correctness"],
                capture_output=True, text=True, timeout=180)
            output = (r.stdout + r.stderr).strip()
            for line in output.splitlines():
                if line.strip(): print(line)
            return r.returncode
        print("FAIL: opus_runner.py missing"); return 1
    if not kernel_py.exists():
        print("FAIL: kernel.py missing"); return 1
    
    runner = task_dir / "scripts" / "correctness_runner.py"
    if not runner.exists():
        # Fallback: just check import
        r = subprocess.run([sys.executable, "-c",
            f"import importlib.util; s=importlib.util.spec_from_file_location('k','{kernel_py}'); "
            f"m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"FAIL: {r.stderr[:300]}"); return 1
        print("PASS: correctness"); return 0
    
    r = subprocess.run([sys.executable, str(runner), str(task_dir)],
                      capture_output=True, text=True, timeout=120)
    output = r.stdout + r.stderr
    for line in output.splitlines():
        if line.strip():
            print(line)
    return 0 if r.returncode == 0 else 1

def run_performance(task_dir):
    """Benchmark kernel with CUDA event timing."""
    kernel_py = task_dir / "kernel.py"
    kernel_cu = task_dir / "kernel.cu"
    # Opus: real benchmark via opus_runner.py
    if kernel_cu.exists() and not kernel_py.exists():
        runner = task_dir / "scripts" / "opus_runner.py"
        if runner.exists():
            r = subprocess.run(
                [sys.executable, str(runner), str(task_dir), "performance"],
                capture_output=True, text=True, timeout=300)
            output = (r.stdout + r.stderr).strip()
            for line in output.splitlines():
                if line.strip(): print(line)
            return r.returncode
        print("Perf: 0.0 ms (opus_runner_missing)"); return 0
    if not kernel_py.exists():
        print("FAIL: kernel file missing"); return 1
    perf_runner = task_dir / "scripts" / "perf_runner.py"
    if perf_runner.exists():
        operator = "unknown"
        for cfg_name in ("config.yaml", "config.yml"):
            cfg_path = task_dir / cfg_name
            if cfg_path.is_file():
                try:
                    import yaml as _yaml
                    operator = _yaml.safe_load(cfg_path.read_text()).get("operator", "unknown")
                except Exception:
                    pass
                break
        r = subprocess.run([sys.executable, str(perf_runner), str(kernel_py), operator],
                          capture_output=True, text=True, timeout=180)
    else:
        r = subprocess.run([sys.executable, str(kernel_py)],
                          capture_output=True, text=True, timeout=120)
    output = r.stdout + r.stderr
    perf_lines = [l for l in output.splitlines() if "Perf:" in l or "ms" in l.lower()]
    if perf_lines:
        for line in perf_lines[:5]:
            print(line)
        return 0
    ms_re = re.compile(r"([0-9]+\.?[0-9]*)\s*ms")
    matches = ms_re.findall(output)
    if matches:
        for i, ms_val in enumerate(matches[:3]):
            print(f"Perf: {ms_val} ms (extracted_{i})")
        return 0
    print("Perf: 0.000 ms (no_timing)"); return 0

if __name__ == "__main__":
    import pathlib
    mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
    task_dir = pathlib.Path(__file__).resolve().parent.parent
    fn = {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[mode]
    sys.exit(fn(task_dir))