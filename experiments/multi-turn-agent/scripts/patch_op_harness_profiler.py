#!/usr/bin/env python3
"""Patch all 150 op_harness.py: warmup=200, run=50 independent profiler contexts, median."""

from pathlib import Path

ARTIFACT_ROOT = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")

OLD_DEFAULTS = "    n_warmup: int = 5\n    n_iter: int = 20"
NEW_DEFAULTS = "    n_warmup: int = 360\n    n_iter: int = 50"

OLD_PROFILE = '''def _profile(func: Callable, args: list, n_warmup: int = 5, n_iter: int = 20) -> float:
    """Return total GPU kernel time per call in ms via torch.profiler."""
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
    return total_us / 1000.0 / n_iter'''

NEW_PROFILE = '''def _profile(func: Callable, args: list, n_warmup: int = 360, n_iter: int = 50) -> float:
    """Return median GPU kernel time per call in ms via torch.profiler.

    Each of the n_iter iterations runs in its own profiler context to get
    independent device-time measurements.  The median is reported.
    """
    try:
        for _ in range(n_warmup):
            func(*args)
        torch.cuda.synchronize()
    except Exception:
        return 0.0

    times_ms: list[float] = []
    for _ in range(n_iter):
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                func(*args)
                torch.cuda.synchronize()
        except Exception:
            continue
        events = [e for e in prof.key_averages() if e.device_time_total > 0]
        if events:
            total_us = sum(e.device_time_total for e in events)
            times_ms.append(total_us / 1000.0)

    if not times_ms:
        return 0.0
    times_ms.sort()
    mid = len(times_ms) // 2
    return times_ms[mid]'''

count_defaults = 0
count_profile = 0
total = 0

for task_dir in sorted(ARTIFACT_ROOT.iterdir()):
    harness_path = task_dir / "initial" / "scripts" / "op_harness.py"
    if not harness_path.exists():
        continue
    total += 1

    content = harness_path.read_text()
    changed = False

    if OLD_DEFAULTS in content:
        content = content.replace(OLD_DEFAULTS, NEW_DEFAULTS)
        count_defaults += 1
        changed = True

    if OLD_PROFILE in content:
        content = content.replace(OLD_PROFILE, NEW_PROFILE)
        count_profile += 1
        changed = True

    if changed:
        harness_path.write_text(content)

print(f"Scanned {total} op_harness.py files")
print(f"  Patched defaults (warmup/iter): {count_defaults}")
print(f"  Patched _profile function: {count_profile}")
