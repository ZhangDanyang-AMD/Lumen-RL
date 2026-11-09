#!/usr/bin/env python3
"""Deploy correctness_runner_v2.py to all 150 task scripts/ directories.

Copies the FlyDSL-aware correctness_runner from the project scripts/ dir
into each artifact's scripts/correctness_runner.py, replacing the old version.
Idempotent: skips files that already contain the build_* detection marker.
"""
import shutil
from pathlib import Path

ARTIFACT_ROOT = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")
SCRIPTS_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPTS_DIR / "correctness_runner_v2.py"

if not SOURCE.exists():
    print(f"ERROR: {SOURCE} not found")
    exit(1)

source_content = SOURCE.read_text()
count = 0

for task_dir in sorted(ARTIFACT_ROOT.iterdir()):
    target = task_dir / "initial" / "scripts" / "correctness_runner.py"
    if not target.parent.exists():
        continue

    if target.exists():
        existing = target.read_text()
        if "build_fns" in existing and "FlyDSL" in existing:
            continue

    target.write_text(source_content)
    count += 1

print(f"Deployed correctness_runner_v2 to {count} task directories")
