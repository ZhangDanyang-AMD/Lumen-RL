"""Collect specialist rollouts on a training pool for step-level rejection-sampling SFT.

The training pool is disjoint from the A/B benchmark inputs: gen0 instructions
plus crafted training instructions, and critic contexts built from this
script's own planner histories. Every finished rollout is appended to
``<out>/<role>.jsonl`` with its step outcomes; ``build_rft_data.py`` turns
them into trainable samples.

Example:
    python experiments/micake-subagent-rsi/scripts/collect_rft.py --phase planner --samples 4 --gpu 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import (  # noqa: E402
    DEFAULT_CONFIG, REPO_ROOT, TASK, TimedMicake, append, arm_spec, done_keys, llm_seconds, replay,
)
from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from lumenrl.agentic.rollout import AgenticRolloutEngine, RolloutConfig  # noqa: E402
from micake_rsi.ab_eval import critic_metrics, planner_metrics  # noqa: E402
from micake_rsi.e2e_contexts import e2e_planner_contexts  # noqa: E402
from micake_rsi.manager_env import ManagerSettings  # noqa: E402
from micake_rsi.roles import build_endpoint_policy  # noqa: E402
from micake_rsi.subagent_envs import CriticEnvironment, PlannerEnvironment  # noqa: E402

logger = logging.getLogger("collect_rft")

GEN0_EPISODES = REPO_ROOT / "output" / "micake-subagent-rsi" / "gen0" / "episodes.jsonl"
TRAIN_INSTRUCTIONS = (
    "Try num_warps=8 with matrix_instr_nonkdim=16 on the baseline tile and compare TWO_DIMENSIONAL against "
    "GROUPED_M ordering. Submit 3 candidates and report the geomean of each.",
    "Explore tile_k=16 and tile_k=64 variants of the 64x64 baseline with num_stages between 2 and 4. Every "
    "candidate must stay inside the shared-memory limit. Submit 3 candidates.",
    "Investigate 128x64 and 64x128 output tiles with tile_k=32 and 8 warps. Submit 3 candidates and report "
    "which workloads gain or lose.",
    "Study occupancy: vary waves_per_eu and num_warps on the baseline tile. Submit 3 candidates.",
    "Explore unroll_factor and vectorize_factor changes combined with matrix_instr_nonkdim=16. Submit 3 candidates.",
    "small-square and deep-k launch few CTAs. Propose smaller tiles (for example 32x32 or 32x64) that raise "
    "parallelism and submit 3 candidates.",
    "Use GROUPED_M with group_m in {4, 8, 16} on a 64x64x32 tile with 8 warps and nonkdim=16. Submit 3 candidates.",
    "Submit three diverse candidates that are each likely to beat the baseline and that respect the "
    "shared-memory limit, then report the best one.",
    "Compare pipeline_depth/num_stages 2 against 4 with tile_k=32 and 8 warps. Submit 3 candidates.",
    "Target large-square: 128x128 tiles with tile_k=16 or 32 and num_stages=2, within shared memory. Submit 3 candidates.",
)
PRESCRIPTIVE_INSTRUCTIONS = (
    "Submit exactly these 3 candidates: (A) tile 128x128x64, num_stages=3, pipeline_depth=3, num_warps=8, "
    "nonkdim=16; (B) tile 128x128x64, num_stages=4, pipeline_depth=4, num_warps=8; (C) tile 128x64x64, "
    "num_stages=4, pipeline_depth=4, num_warps=8, GROUPED_M group_m=8. Report the geomean of each.",
    "Large tiles should win on large-square. Try 128x128 with tile_k=64 and deep pipelines (num_stages=4, "
    "pipeline_depth=4), num_warps 8, matrix_instr_nonkdim=16. Submit 3 variants of this and report.",
    "Push latency hiding to the maximum: tile_k=64, num_stages=4, pipeline_depth=4 on 128x64 and 64x128 "
    "tiles with 8 warps. Submit 3 candidates.",
    "Try the biggest legal-looking shapes: 128x128x32 with num_stages=4 and pipeline_depth=4, unroll_factor=8, "
    "vectorize_factor=8, num_warps=8. Submit 3 candidates around this point.",
    "Submit these three: 64x128x64 stages4/depth4 warps8; 128x64x64 stages3/depth4 warps8; 128x128x32 "
    "stages4/depth4 warps8 nonkdim16. Report per-workload results.",
    "Test whether deeper software pipelining helps deep-k: tile 64x64x64 with num_stages=4, pipeline_depth=4, "
    "and tile 128x64x64 with the same pipelining; num_warps=8. Submit 3 candidates.",
    "For the tall workload try 128x128x64 with GROUPED_M group_m=8, num_stages=3, pipeline_depth=4, 8 warps, "
    "plus two neighbours of it. Submit 3 candidates.",
    "Maximize arithmetic intensity: 128x128 tiles, tile_k=64, unroll_factor=8, vectorize_factor=8, "
    "num_stages=3, pipeline_depth=3, num_warps=8. Submit 3 candidates near it.",
)
HISTORY_INSTRUCTION = (
    "Use the evaluated history: keep what worked, avoid every configuration family that was rejected or slow, "
    "and submit 3 new candidates that try to beat the current best. Report the outcome."
)
CRITIC_QUESTIONS = (
    "Diagnose the evaluated history and recommend ONE configuration for immediate evaluation.",
    "Which bottleneck limits the current best? Name it with evidence from the history and recommend one "
    "configuration that addresses it without being resource-rejected.",
)


def training_instructions() -> list[str]:
    gen0 = []
    for line in GEN0_EPISODES.read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if episode["role"] == "planner":
            gen0.append(json.loads(episode["messages"][1]["content"])["manager_instruction"])
    return gen0 + list(TRAIN_INSTRUCTIONS)


def collected_histories(out: Path) -> list[list[dict]]:
    """Counted submissions of earlier planner rollouts (config + hypothesis), oldest first."""
    histories = []
    for line in (out / "planner.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        entries = [{"config": row["config"], "hypothesis": "earlier candidate"} for row in record["trace"]
                   if row["counted"] and row.get("config")]
        if entries:
            histories.append(entries)
    return histories


def history_contexts(out: Path, count: int, seed: int) -> list[list[dict]]:
    """Concatenations of two earlier planner histories, de-duplicated."""
    histories = collected_histories(out)
    rng = random.Random(seed)
    contexts = []
    for _ in range(count):
        a, b = rng.sample(histories, 2)
        seen, merged = set(), []
        for entry in a + b:
            key = json.dumps(entry["config"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(entry)
        contexts.append(merged[:6])
    return contexts


async def run_planner(cfg, micake, timed, arm, items, samples, out: Path, phase: str, shard: int = 0, shards: int = 1) -> None:
    settings = ManagerSettings.from_dict(cfg["manager"]["settings"])
    spec = arm_spec(cfg, arm, "planner")
    engine = AgenticRolloutEngine(build_endpoint_policy(spec.endpoint, spec.binding),
                                  RolloutConfig(spec.max_turns, int(cfg["rsi"]["specialist_max_no_tool_nudges"])))
    finished = done_keys(out / "planner.jsonl")
    for k in range(samples):
        for i, (history, instruction) in enumerate(items):
            key = f"{phase}:{arm}:{i}:{k}"
            if key in finished or i % shards != shard:
                continue
            await micake.reset(TASK)
            await replay(micake, history)
            timed.start()
            env = PlannerEnvironment(micake, settings.history_limit, settings.planner_report_reward)
            began = time.monotonic()
            episode = await engine.run_episode(env, {"instruction": instruction, "max_submissions": spec.max_submissions},
                                               group_id=f"{phase}-{i}", role="planner", episode_id=uuid.uuid4().hex)
            data = episode.to_dict()
            append(out / "planner.jsonl", {"key": key, "phase": phase, "arm": arm, "item": i, "sample": k,
                                           "history_len": len(history), "wall_s": time.monotonic() - began,
                                           "llm_s": llm_seconds([episode]), "metrics": planner_metrics(data),
                                           "trace": timed.trace, "episode": data})
            m = planner_metrics(data)
            logger.info("%s wasted=%d/%d best=%.4f", key, m["wasted"], m["actions"], m["best_speedup"])


async def run_critic(cfg, micake, timed, arm, contexts, samples, out: Path) -> None:
    settings = ManagerSettings.from_dict(cfg["manager"]["settings"])
    spec = arm_spec(cfg, arm, "critic")
    engine = AgenticRolloutEngine(build_endpoint_policy(spec.endpoint, spec.binding),
                                  RolloutConfig(spec.max_turns, int(cfg["rsi"]["specialist_max_no_tool_nudges"])))
    finished = done_keys(out / "critic.jsonl")
    for k in range(samples):
        for i, history in enumerate(contexts):
            key = f"critic:{arm}:{i}:{k}"
            if key in finished:
                continue
            question = CRITIC_QUESTIONS[(i + k) % len(CRITIC_QUESTIONS)]
            await micake.reset(TASK)
            await replay(micake, history)
            context_best = float(micake.best()["geomean_speedup_vs_baseline"])
            timed.start()
            env = CriticEnvironment(micake, settings.history_limit, settings.critic_rewards)
            episode = await engine.run_episode(env, {"instruction": question}, group_id=f"critic-{i}", role="critic",
                                               episode_id=uuid.uuid4().hex)
            data = episode.to_dict()
            metrics = critic_metrics(data, context_best)
            append(out / "critic.jsonl", {"key": key, "phase": "critic", "arm": arm, "item": i, "sample": k,
                                          "context_best": context_best, "metrics": metrics, "trace": timed.trace,
                                          "episode": data})
            logger.info("%s probe=%s speedup=%.4f best=%.4f", key, metrics["probe_class"], metrics["probe_speedup"], context_best)


async def main_async(args) -> None:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    kwargs = {**cfg["micake"]["kwargs"], "gpu_id": args.gpu}
    micake = build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **kwargs)
    timed = TimedMicake(micake)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "planner":
            items = [([], text) for text in training_instructions()]
            await run_planner(cfg, micake, timed, args.arm, items, args.samples, args.out, "planner")
        elif args.phase == "planner_prescriptive":
            items = [([], text) for text in PRESCRIPTIVE_INSTRUCTIONS]
            await run_planner(cfg, micake, timed, args.arm, items, args.samples, args.out, "planner_prescriptive")
        elif args.phase == "planner_e2e":
            items = [(c["history"], c["instruction"]) for c in e2e_planner_contexts(args.e2e_dir, "train")]
            await run_planner(cfg, micake, timed, args.arm, items, args.samples, args.out, "planner_e2e", args.shard, args.shards)
        elif args.phase == "planner_history":
            items = [(h, HISTORY_INSTRUCTION) for h in history_contexts(args.out, args.contexts, seed=1)]
            await run_planner(cfg, micake, timed, args.arm, items, args.samples, args.out, "planner_history")
        else:
            await run_critic(cfg, micake, timed, args.arm, history_contexts(args.out, args.contexts, seed=2), args.samples, args.out)
    finally:
        await micake.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", required=True, choices=("planner", "planner_prescriptive", "planner_history", "planner_e2e", "critic"))
    parser.add_argument("--arm", default="base")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--e2e-dir", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "e2e_v1")
    parser.add_argument("--shard", type=int, default=0, help="run items with index %% shards == shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "rft_pool")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
