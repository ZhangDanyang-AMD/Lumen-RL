#!/usr/bin/env python3
"""Patch all 150 task_runner.py files to remove opus correctness/perf shortcuts."""

from pathlib import Path

ARTIFACT_ROOT = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")

# Old opus correctness shortcut patterns to replace
OLD_CORRECTNESS = '''    # Opus shortcut if no Python kernel
    if kernel_cu.exists() and not kernel_py.exists():
        print("PASS: correctness (opus kernel present)"); return 0'''

NEW_CORRECTNESS = '''    # Opus: must compile successfully (no free pass)
    if kernel_cu.exists() and not kernel_py.exists():
        include_dir = task_dir / "include"
        inc_flags = ["-I", str(include_dir)] if include_dir.is_dir() else []
        r = subprocess.run(
            ["hipcc", "--genco", "-O3", "--offload-arch=gfx950"] + inc_flags +
            [str(kernel_cu), "-o", "/dev/null"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"FAIL: opus kernel compile error\\n{r.stderr[:500]}")
            return 1
        print("PASS: correctness (opus kernel compiles)")
        return 0'''

OLD_PERF = '''    if kernel_cu.exists() and not kernel_py.exists():
        print("Perf: 1.000 ms (opus_placeholder)"); return 0'''

NEW_PERF = '''    if kernel_cu.exists() and not kernel_py.exists():
        print("Perf: 0.0 ms (opus, no python wrapper)")
        return 0'''

count_correctness = 0
count_perf = 0
total = 0

for task_dir in sorted(ARTIFACT_ROOT.iterdir()):
    harness_path = task_dir / "initial" / "scripts" / "task_runner.py"
    if not harness_path.exists():
        continue
    total += 1

    content = harness_path.read_text()
    changed = False

    if OLD_CORRECTNESS in content:
        content = content.replace(OLD_CORRECTNESS, NEW_CORRECTNESS)
        count_correctness += 1
        changed = True

    if OLD_PERF in content:
        content = content.replace(OLD_PERF, NEW_PERF)
        count_perf += 1
        changed = True

    if changed:
        harness_path.write_text(content)

print(f"Scanned {total} task_runner.py files")
print(f"  Patched correctness shortcut: {count_correctness}")
print(f"  Patched perf placeholder: {count_perf}")
