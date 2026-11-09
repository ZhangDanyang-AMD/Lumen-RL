"""Build improvement-driven Planner SFT samples from end-to-end experience.

Positives are the ``new_best`` Planner turns of every recorded end-to-end
episode; teacher corrections replace other submit turns with a MICake-measured
one-axis move from the session best that beat it. Measurements are cached per
config and reused across shards and generations.

Example:
    python experiments/micake-subagent-rsi/scripts/build_improvements.py --exp-dir output/micake-subagent-rsi/e2e_v2 \
        --gpu 6 --shard 0 --shards 4 --out /tmp/improve_s0.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import DEFAULT_CONFIG, TASK  # noqa: E402
from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from micake_rsi.improve import (  # noqa: E402
    config_key, correctable, exploit_candidates, exploit_sample, improvement_sample, improves, planner_children, walk_steps,
)

logger = logging.getLogger("build_improvements")


def experience(dirs: list[Path]) -> list[dict]:
    records = []
    for directory in dirs:
        for path in sorted(glob.glob(str(directory / "e2e*.jsonl"))):
            records += [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return records


async def main_async(args) -> None:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    micake = build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **{**cfg["micake"]["kwargs"], "gpu_id": args.gpu})
    cache: dict = {}
    for path in args.cache_in:
        if path.exists():
            cache.update(json.loads(path.read_text()))
    cache_out = args.out.with_suffix(".cache.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples, stats = [], {"records": 0, "children": 0, "positives": 0, "corrected": 0, "no_improvement": 0, "repeated_target": 0,
                       "measured": 0}
    targets: dict[tuple[str, str], int] = {}

    async def evaluate(config):
        key = config_key(config)
        if key not in cache:
            result = await micake.measure(config)
            perf = result.get("performance") or {}
            cache[key] = {"stage": result.get("stage"), "validity": perf.get("validity"), "speedup": perf.get("geomean_speedup")}
            stats["measured"] += 1
            cache_out.write_text(json.dumps(cache))
        return cache[key]

    await micake.reset(TASK)
    domain = micake.domain
    try:
        for n, record in enumerate(experience(args.exp_dir)):
            if n % args.shards != args.shard:
                continue
            stats["records"] += 1
            for c, child in enumerate(planner_children(record)):
                stats["children"] += 1
                key = f"{record['key']}@{record.get('arm')}#{c}"
                positive = improvement_sample(child, f"{key}:positive")
                if positive:
                    samples.append(positive)
                    stats["positives"] += 1
                for entry in correctable(list(walk_steps(child)))[: args.max_corrections]:
                    chosen, chosen_speedup = None, 0.0
                    for i, candidate in enumerate(exploit_candidates(entry["best_config"], domain, entry["submitted"])):
                        if i >= args.max_attempts:
                            break
                        measured = await evaluate(candidate)
                        if improves(measured, entry["best_speedup"]) and measured["speedup"] > chosen_speedup:
                            chosen, chosen_speedup = candidate, measured["speedup"]
                    if chosen is None:
                        stats["no_improvement"] += 1
                        continue
                    target = (config_key(entry["best_config"]), config_key(chosen))
                    targets[target] = targets.get(target, 0) + 1
                    if targets[target] > args.max_per_target:
                        stats["repeated_target"] += 1
                        continue
                    samples.append(exploit_sample(child, entry, chosen, f"{key}:exploit{entry['step']['step_index']}"))
                    stats["corrected"] += 1
    finally:
        await micake.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    cache_out.write_text(json.dumps(cache))
    logger.info("improvements: %s -> %s", stats, args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--exp-dir", type=Path, action="append", required=True, help="directory with e2e*.jsonl records")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=64, help="measured one-axis candidates per corrected turn (steepest ascent)")
    parser.add_argument("--max-corrections", type=int, default=3, help="corrected turns per Planner episode")
    parser.add_argument("--max-per-target", type=int, default=3, help="samples per (session best, correction) pair")
    parser.add_argument("--cache-in", type=Path, action="append", default=[], help="measurement caches to seed from")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
