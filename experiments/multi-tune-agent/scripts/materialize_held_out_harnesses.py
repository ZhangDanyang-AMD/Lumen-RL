"""Regenerate complete GEAK harnesses for held-out tasks from factory templates.

Reads kernel.jsonl, reconstructs Top10Request for each task, calls
render_template() to deterministically produce config.yaml, scripts/task_runner.py,
and metadata.json. Verifies SHA-256 against protected hashes.
"""

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_tune_agent.top10_canonical_templates import Top10Request, render_template
from multi_tune_agent.top10_inventory import AITER_GIT_SHA


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def reconstruct_request(task: dict) -> Top10Request:
    """Reconstruct a Top10Request from a kernel.jsonl task record."""
    family = task["top10_family"]
    lane = task["lane"]
    language = lane.split("_", 1)[0]
    contract = task["contract"]["kernel_contract"]

    seed = {
        "source_sha": AITER_GIT_SHA,
        "split_version": "v5",
        "split_group": "held_out",
        "source_artifacts": [task["factory"]["source_artifact"]],
        "source_lineage_id": task["source_lineage_id"],
        "contract_family_id": task["contract_family_id"],
        "top10_family": family,
    }

    return Top10Request(
        request_id=task["task_id"],
        request_text="Deterministic eval-only held-out contract; generation is forbidden.",
        family=family,
        language=language,
        shape=tuple(int(v) for v in contract["shape"].values()),
        seed_provenance=seed,
        recognized_contract={
            "target_gpu": "gfx942",
            "language": language,
            "contract": dict(contract),
        },
    )


def materialize(held_out_root: str, docker_mode: bool = False):
    held_out_root = Path(held_out_root)
    tasks_path = held_out_root / "tasks" / "kernel.jsonl"

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} held-out tasks")

    ok = 0
    sha_mismatch = 0
    errors = 0

    for task in tasks:
        task_id = task["task_id"]
        task_dir = held_out_root / "artifacts" / "kernel" / task_id / "initial"

        try:
            request = reconstruct_request(task)
            rendered = render_template(request)
        except Exception as e:
            print(f"  ERROR {task_id}: render_template failed: {e}")
            errors += 1
            continue

        runner_content = rendered.get("scripts/task_runner.py", "")
        runner_hash = _sha256(runner_content)
        expected_hash = task.get("protected", {}).get("harness", {}).get("sha256", "")

        if expected_hash and runner_hash != expected_hash:
            print(f"  WARN {task_id}: SHA-256 mismatch (got {runner_hash[:12]}, expected {expected_hash[:12]})")
            sha_mismatch += 1

        # Copy initial_source.py -> kernel.py
        initial_src = task_dir / "initial_source.py"
        kernel_dst = task_dir / "kernel.py"
        if initial_src.exists() and not kernel_dst.exists():
            shutil.copy2(initial_src, kernel_dst)

        # Write config.yaml
        config_content = rendered.get("config.yaml", "")
        if not docker_mode:
            config_content = config_content.replace(
                'docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} -w "$PWD" '
                '${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3',
                'HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} python3',
            )
            config_content = config_content.replace(
                'docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:?required} -w "$PWD" '
                '${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3',
                'HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} python3',
            )
        config_path = task_dir / "config.yaml"
        config_path.write_text(config_content, encoding="utf-8")

        # Write scripts/task_runner.py
        scripts_dir = task_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        runner_path = scripts_dir / "task_runner.py"
        runner_path.write_text(runner_content, encoding="utf-8")

        # Write metadata.json
        metadata = {
            "task_id": task_id,
            "family": task["top10_family"],
            "lane": task["lane"],
            "source_suite": task.get("source_suite", ""),
            "contract": task["contract"],
            "factory": task["factory"],
            "harness_sha256": runner_hash,
        }
        metadata_path = task_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        ok += 1

    print(f"\nDone: {ok} OK, {sha_mismatch} SHA mismatches, {errors} errors")
    print(f"Total: {len(tasks)} tasks")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out-root", default="/home/danyzhan/held-out-benchmark")
    parser.add_argument("--docker-mode", action="store_true")
    args = parser.parse_args()
    materialize(args.held_out_root, docker_mode=args.docker_mode)
