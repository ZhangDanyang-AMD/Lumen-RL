"""Build MICake-verified corrections for resource-rejected Planner turns in rollout pools.

Example:
    python experiments/micake-subagent-rsi/scripts/build_corrections.py --gpu 6 \
        --pool output/micake-subagent-rsi/rft_pool --pool output/micake-subagent-rsi/rft_pool_dsv4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT, TASK  # noqa: E402
from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from micake_rsi.corrections import correction_sample, duplicate_repairs, repairs  # noqa: E402
from sandbox.micake.tool_runtime import HIDDEN_FIXED_FIELDS  # noqa: E402

logger = logging.getLogger("build_corrections")


def _key(config) -> str:
    return json.dumps(config, sort_keys=True)


def submitted_config(arguments, domain):
    """Full config of submit arguments (hidden fixed fields restored), or None when malformed."""
    if not isinstance(arguments, dict) or not isinstance(arguments.get("schedule"), dict) or not isinstance(arguments.get("realization"), dict):
        return None
    config = {"schedule": dict(arguments["schedule"]), "realization": dict(arguments["realization"])}
    for section, names in HIDDEN_FIXED_FIELDS.items():
        for name in names:
            config[section][name] = domain[section][name][0]
    return config


async def main_async(args) -> None:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    micake = build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **{**cfg["micake"]["kwargs"], "gpu_id": args.gpu})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.out.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    await micake.reset(TASK)
    domain = micake.domain
    samples, stats = [], {"corrected": 0, "no_repair": 0}

    async def evaluate(config):
        key = _key(config)
        if key not in cache:
            result = await micake.measure(config)
            perf = result.get("performance") or {}
            cache[key] = {"stage": result.get("stage"), "validity": perf.get("validity"),
                          "speedup": perf.get("geomean_speedup")}
            cache_path.write_text(json.dumps(cache))
        return cache[key]

    try:
        for pool in args.pool:
            for n, line in enumerate((pool / "planner.jsonl").read_text(encoding="utf-8").splitlines()):
                if n % args.shards != args.shard:
                    continue
                record = json.loads(line)
                episode = record["episode"]
                if any(len(m.get("tool_calls") or ()) > 1 for m in episode["messages"]):
                    continue
                submitted = {_key(micake.task_packet()["baseline_config"])}
                try:
                    packet = json.loads(episode["messages"][1]["content"])
                except (ValueError, TypeError):
                    packet = {}
                submitted.update(_key(h["config"]) for h in packet.get("history") or () if h.get("config"))
                for index, step in enumerate(episode["steps"]):
                    label = step["result"].get("execution_result")
                    config = (step["result"].get("metadata") or {}).get("config")
                    if label == "duplicate":
                        config = submitted_config(step["action"].get("arguments"), domain)
                    if label not in ("resource_rejected", "duplicate") or not config:
                        if config:
                            submitted.add(_key(config))
                        continue
                    stats[label] = stats.get(label, 0) + 1
                    candidates = repairs(config, domain) if label == "resource_rejected" else duplicate_repairs(config, domain)
                    chosen = None
                    tried = 0
                    for candidate in candidates:
                        if tried >= args.max_attempts:
                            break
                        if _key(candidate) in submitted:
                            continue
                        tried += 1
                        outcome = await evaluate(candidate)
                        if outcome["stage"] == "measured" and outcome["validity"] == "VALID" and (outcome["speedup"] or 0) >= args.min_speedup:
                            chosen = candidate
                            break
                    submitted.add(_key(config))
                    if chosen is None:
                        stats["no_repair"] += 1
                        continue
                    stats["corrected"] += 1
                    submitted.add(_key(chosen))
                    samples.append(correction_sample(record, index, chosen, config, label))
                    logger.info("%s step %d %s corrected (%s)", record["key"], index, label, cache[_key(chosen)])
    finally:
        await micake.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    logger.info("corrections: %s -> %s", stats, args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--min-speedup", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "corrections" / "planner.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
