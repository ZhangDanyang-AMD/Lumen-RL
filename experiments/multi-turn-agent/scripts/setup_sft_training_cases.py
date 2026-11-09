#!/usr/bin/env python3
"""Clone held-out benchmark harnesses into training-split cases for GEAKflow SFT collection.

The held-out tasks have complete, verified harnesses (task_runner.py, compile_check.py,
correctness_runner.py, perf_runner.py, config.yaml, kernel.py, metadata.json).
We clone them with split_group="train" so GEAKflow accepts them for SFT.

Usage:
    python3 scripts/setup_sft_training_cases.py [--lang triton|flydsl|all] [--repeats 30]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import yaml
from pathlib import Path

HELD_OUT = Path("/home/danyzhan/agent-phase2-held-out-aiter")
CASES_YAML = HELD_OUT / "receipts" / "phase2_benchmark_cases.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "sft-training-cases"


def clone_harness(
    task_id: str,
    src_dir: Path,
    dst_dir: Path,
    new_task_id: str,
    repeat_idx: int,
) -> dict:
    """Clone a held-out harness into a training-split case."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    config_src = src_dir / "config.yaml"
    cfg = yaml.safe_load(config_src.read_text())

    cfg["task_id"] = new_task_id
    cfg.pop("best_config", None)

    scripts_src = src_dir / "scripts"
    scripts_dst = dst_dir / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)

    for script in ("task_runner.py", "compile_check.py", "correctness_runner.py",
                    "perf_runner.py", "op_harness.py", "opus_runner.py"):
        src = scripts_src / script
        if src.exists():
            shutil.copy2(src, scripts_dst / script)

    kernel_src = src_dir / "kernel.py"
    if kernel_src.exists():
        shutil.copy2(kernel_src, dst_dir / "kernel.py")

    kernel_jit_src = src_dir / "kernel_jit.py"
    if kernel_jit_src.exists():
        shutil.copy2(kernel_jit_src, dst_dir / "kernel_jit.py")

    build_src = src_dir / "build"
    if build_src.is_dir():
        build_dst = dst_dir / "build"
        if build_dst.exists():
            shutil.rmtree(build_dst)
        shutil.copytree(build_src, build_dst)

    (dst_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )

    meta_src = src_dir / "metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta["task_id"] = new_task_id
        meta["cloned_from"] = task_id
        meta["split_group"] = "train"
        meta["repeat_idx"] = repeat_idx
        (dst_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all", choices=["triton", "flydsl", "all"])
    parser.add_argument("--repeats", type=int, default=30,
                        help="Number of repeated clones per task (each gets a unique ID for multiple agent runs)")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases_data = yaml.safe_load(CASES_YAML.read_text())
    all_tasks = cases_data.get("tasks", [])

    filtered = []
    for t in all_tasks:
        task_id = t["id"]
        cfg_path = HELD_OUT / "artifacts" / "kernel" / task_id / "initial" / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        lang = cfg.get("language", "unknown")
        if args.lang == "all" and lang in ("triton", "flydsl"):
            filtered.append((task_id, lang, cfg.get("operator", "unknown")))
        elif lang == args.lang:
            filtered.append((task_id, lang, cfg.get("operator", "unknown")))

    print(f"Found {len(filtered)} {args.lang} benchmark tasks to clone")

    catalog_tasks = []
    for task_id, lang, operator in filtered:
        src_dir = HELD_OUT / "artifacts" / "kernel" / task_id / "initial"
        for r in range(args.repeats):
            suffix = hashlib.md5(f"{task_id}-train-r{r}".encode()).hexdigest()[:8]
            new_id = f"train-gfx950-{operator}-{lang}-{suffix}"
            dst_dir = output_root / "artifacts" / "kernel" / new_id / "initial"

            cfg = clone_harness(task_id, src_dir, dst_dir, new_id, r)

            shapes = cfg.get("benchmark_shapes", [])
            kernel_prompt = ""
            if shapes and isinstance(shapes[0], dict):
                kernel_prompt = shapes[0].get("kernel_prompt", "")

            direction = f"Write a high-performance {lang} kernel for the \"{operator}\" operator on AMD gfx950."
            if kernel_prompt:
                direction += f" {kernel_prompt}"

            catalog_tasks.append({
                "id": new_id,
                "type": "aiter_generated",
                "kernel_path": str(dst_dir),
                "direction": direction,
                "provenance": {
                    "case_seed": {
                        "split_group": "train",
                        "source_lineage_id": f"clone-{task_id}",
                        "split_version": "v5",
                        "cloned_from": task_id,
                        "repeat_idx": r,
                    }
                },
            })

    catalog_path = output_root / "sft_training_cases.yaml"
    catalog_path.write_text(
        yaml.safe_dump({"tasks": catalog_tasks}, sort_keys=False, allow_unicode=True)
    )

    by_lang = {}
    for _, lang, _ in filtered:
        by_lang[lang] = by_lang.get(lang, 0) + 1

    total = len(catalog_tasks)
    print(f"\nCreated {total} training cases ({args.repeats} repeats each):")
    for lang, count in sorted(by_lang.items()):
        print(f"  {lang}: {count} unique × {args.repeats} repeats = {count * args.repeats}")
    print(f"\nCatalog: {catalog_path}")
    print(f"Artifacts: {output_root / 'artifacts'}")

    gen_manifest = {
        "version": 1,
        "purpose": f"gfx950-sft-training-{args.lang}",
        "requests": [],
    }
    for entry in catalog_tasks:
        gen_manifest["requests"].append({
            "id": entry["id"],
            "request": entry["direction"],
            "seed_provenance": entry["provenance"]["case_seed"],
        })

    manifest_path = output_root / "generation_requests.yaml"
    manifest_path.write_text(
        yaml.safe_dump(gen_manifest, sort_keys=False, allow_unicode=True)
    )
    print(f"Generation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
