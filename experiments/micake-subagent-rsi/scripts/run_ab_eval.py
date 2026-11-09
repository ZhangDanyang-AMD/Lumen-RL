"""A/B benchmark of specialist generations on MICake C144 (see ``micake_rsi/ab_eval.py``).

Arms:
    dsv4    DSV4.1 serves the Planner/Critic roles (same prompts and budgets).
    base    untuned Qwen3-Coder-30B-A3B specialists (gen0 weights).
    rsi     RSI-trained specialists (gen1 merged checkpoints).
    gen2    step-level rejection-sampling SFT specialists (gen2 merged checkpoints).
    search  rule-based greedy harness, no LLM (e2e only).

Studies: ``planner`` and ``critic`` isolate one specialist on fixed inputs;
``e2e`` runs the frozen DSV4.1 Manager with direct submission disabled and a
cold result cache per episode. Every finished trial is appended to
``<out>/<study>.jsonl``; rerunning skips trials already recorded.

Example:
    export DSV4_API_KEY=$(rg -o '[0-9a-f]{64}' /home/danyzhan/DSV4-API.key)
    python experiments/micake-subagent-rsi/scripts/run_ab_eval.py --study planner --samples 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from lumenrl.agentic.core.protocols import ToolCall  # noqa: E402
from lumenrl.agentic.rollout import AgenticRolloutEngine, RolloutConfig  # noqa: E402
from micake_rsi.ab_eval import critic_metrics, e2e_metrics, planner_metrics  # noqa: E402
from micake_rsi.guards import remeasure  # noqa: E402
from micake_rsi.manager_env import ManagerEnvironment, ManagerSettings  # noqa: E402
from micake_rsi.e2e_contexts import e2e_planner_contexts  # noqa: E402
from micake_rsi.roles import Endpoint, PolicyBinding, RoleSpec, build_endpoint_policy, load_role_specs  # noqa: E402
from micake_rsi.search_harness import GreedySearch, submission_arguments  # noqa: E402
from micake_rsi.subagent_envs import CriticEnvironment, CriticRewards, PlannerEnvironment  # noqa: E402
from sandbox.micake.tool_runtime import SUBMIT_TOOL  # noqa: E402

logger = logging.getLogger("ab_eval")

DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "gen0_planner_critic.yaml"
RSI_ROOT = REPO_ROOT / "output" / "micake-subagent-rsi"
GEN1_EPISODES = RSI_ROOT / "gen1" / "episodes.jsonl"
TRAINED_ARMS = {
    "rsi": ("gen1", {"planner": 8100, "critic": 8101}),
    "gen2": ("gen2", {"planner": 8103}),
    "gen3": ("gen3", {"planner": 8105}),
    "gen4": ("gen4", {"planner": 8106}),
    "gen5": ("gen5", {"planner": 8107}),
    "gen6": ("gen6", {"planner": 8107}),
    "gen7": ("gen7", {"planner": 8108}),
    "gen8": ("gen8", {"planner": 8109}),
}
BASE_URL = "http://127.0.0.1:8102/v1"
BASE_MODEL = "/home/danyzhan/Lumen/experiments/GEAK-agent-coder/models/Qwen3-Coder-30B-A3B-Instruct"
TASK = {"task_id": "c144-gemm-portfolio", "max_submissions": 10}
GENERIC_QUESTION = (
    "Diagnose the evaluated history and recommend ONE new configuration for immediate evaluation "
    "that is likely to beat the current best without being resource-rejected."
)
CRAFTED_INSTRUCTIONS = (
    "Find a configuration that beats the baseline geomean on the five-workload portfolio. Start from the "
    "baseline and change one or two parameters per candidate. Submit 3 candidates and report per-workload results.",
    "Raise arithmetic intensity: try larger tile_k or deeper pipelines while staying within the shared-memory "
    "limit in resource_limits. Submit 3 candidates that each fit the limit and report which workloads improve.",
    "Keep the baseline tile shape and explore realization-only changes (num_warps, matrix_instr_nonkdim, "
    "program_order with group_m, waves_per_eu). Submit 3 distinct candidates and report the best one.",
    "The tall (m4096 n256) and wide (m256 n4096) workloads dominate the geomean. Propose non-square tiles "
    "(tile_m != tile_n) that fit the shared-memory limit. Submit 3 candidates and report per-workload speedups.",
)


PLANNER_ARMS: dict[str, tuple[int, str]] = {}
FALLBACK = {"role_arm": "base"}


def register_planner_arm(spec: str) -> None:
    """``name=port:checkpoint``: Planner served from ``checkpoint`` on ``port``; other roles use ``FALLBACK``."""
    name, rest = spec.split("=", 1)
    port, checkpoint = rest.split(":", 1)
    PLANNER_ARMS[name] = (int(port), checkpoint)


def arm_endpoints(cfg: Mapping[str, Any], arm: str) -> dict[str, Endpoint]:
    """Endpoint per specialist role for ``arm``."""
    specs = load_role_specs(cfg["specialists"])
    manager = dict(cfg["manager"]["endpoint"])
    out: dict[str, Endpoint] = {}
    for role, spec in specs.items():
        base = {**{k: getattr(spec.endpoint, k) for k in ("api_key_env", "verify_tls", "timeout_s", "max_retries")},
                "sampling": dict(spec.endpoint.sampling), "extra_body": dict(spec.endpoint.extra_body)}
        if arm in PLANNER_ARMS:
            if role == "planner":
                port, checkpoint = PLANNER_ARMS[arm]
                out[role] = Endpoint.from_dict({**base, "base_url": f"http://127.0.0.1:{port}/v1", "model": checkpoint})
            elif FALLBACK["role_arm"] == "dsv4":
                out[role] = Endpoint.from_dict(manager)
            else:
                out[role] = Endpoint.from_dict({**base, "base_url": BASE_URL, "model": BASE_MODEL})
        elif arm == "base" or (arm in TRAINED_ARMS and role not in TRAINED_ARMS[arm][1]):
            out[role] = Endpoint.from_dict({**base, "base_url": BASE_URL, "model": BASE_MODEL})
        elif arm in TRAINED_ARMS:
            generation, ports = TRAINED_ARMS[arm]
            checkpoint = RSI_ROOT / generation / "checkpoints" / role
            out[role] = Endpoint.from_dict({**base, "base_url": f"http://127.0.0.1:{ports[role]}/v1", "model": str(checkpoint)})
        elif arm == "dsv4":
            out[role] = Endpoint.from_dict(manager)
        else:
            raise ValueError(f"arm {arm!r} has no specialist endpoints")
    return out


def arm_spec(cfg: Mapping[str, Any], arm: str, role: str) -> RoleSpec:
    spec = load_role_specs(cfg["specialists"])[role]
    binding = PolicyBinding(role, spec.binding.base_model_id, f"{role}-{arm}", arm, f"{role}-{arm}", False)
    return RoleSpec(binding, arm_endpoints(cfg, arm)[role], spec.max_turns, spec.max_submissions)


def held_out_planner_instructions() -> list[str]:
    """Standalone gen1 planner instructions (gen1 data is never trained on) plus crafted ones."""
    picked = []
    for line in GEN1_EPISODES.read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if episode["role"] != "planner":
            continue
        packet = json.loads(episode["messages"][1]["content"])
        if not packet.get("history"):
            picked.append(packet["manager_instruction"])
    return picked + list(CRAFTED_INSTRUCTIONS)


def critic_contexts() -> list[dict[str, Any]]:
    """Fixed histories from gen1 critic calls, plus a 3-entry prefix with a generic question."""
    contexts = []
    for line in GEN1_EPISODES.read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if episode["role"] != "critic":
            continue
        packet = json.loads(episode["messages"][1]["content"])
        history = [{"config": h["config"], "hypothesis": h["hypothesis"]} for h in packet["history"]]
        contexts.append({"id": f"gen1-{episode['episode_id'][:8]}", "history": history, "question": packet["manager_question"]})
    longest = max(contexts, key=lambda c: len(c["history"]))
    contexts.append({"id": f"{longest['id']}-prefix3", "history": longest["history"][:3], "question": GENERIC_QUESTION})
    return contexts


class TimedMicake:
    """Records every MICake ``step`` (stage, speedup, budget used, timestamps)."""

    def __init__(self, micake: Any) -> None:
        self.micake = micake
        self.inner_step = micake.step
        self.trace: list[dict[str, Any]] = []
        self.t0 = time.monotonic()
        self.eval_s = 0.0
        micake.step = self.step

    def start(self) -> None:
        self.trace, self.t0, self.eval_s = [], time.monotonic(), 0.0

    async def step(self, action: ToolCall):
        began = time.monotonic()
        result = await self.inner_step(action)
        self.eval_s += time.monotonic() - began
        if action.name == SUBMIT_TOOL:
            obs = result.observation or {}
            perf = obs.get("performance") or {}
            self.trace.append({
                "t": time.monotonic() - self.t0,
                "index": self.micake._state["submissions"],
                "counted": obs.get("submission_counted", True) and result.execution_result not in ("schema_invalid", "duplicate"),
                "stage": result.execution_result,
                "speedup": perf.get("geomean_speedup_vs_baseline") if perf.get("validity") == "VALID" else None,
                "config": (result.metadata or {}).get("config"),
            })
        return result


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def interleaved(items: list[Any], arms: list[str], samples: int, start: int = 0):
    """(sample, item, arm) with the arm order rotated per trial to cancel drift."""
    for k in range(start, start + samples):
        for i, item in enumerate(items):
            shift = (k + i) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                yield k, i, item, arm


def llm_seconds(episodes: list[Any]) -> float:
    return sum(turn.latency_s for episode in episodes for turn in episode.turns)


def e2e_test_items(directory: Path, limit: int) -> list[tuple[list, str]]:
    """Held-out real Manager delegations (history, instruction), evenly thinned to ``limit``."""
    contexts = e2e_planner_contexts(directory, "test")
    stride = max(1, len(contexts) // limit)
    return [(c["history"], c["instruction"]) for c in contexts[::stride][:limit]]


async def run_planner(cfg, micake, timed, arms, samples, out: Path, sample_start: int = 0,
                      items: list[tuple[list, str]] | None = None, study: str = "planner",
                      shard: int = 0, shards: int = 1) -> None:
    settings = ManagerSettings.from_dict(cfg["manager"]["settings"])
    nudges = int(cfg["rsi"]["specialist_max_no_tool_nudges"])
    specs = {arm: arm_spec(cfg, arm, "planner") for arm in arms}
    engines = {arm: AgenticRolloutEngine(build_endpoint_policy(s.endpoint, s.binding), RolloutConfig(s.max_turns, nudges)) for arm, s in specs.items()}
    finished = done_keys(out)
    items = items if items is not None else [([], text) for text in held_out_planner_instructions()]
    for k, i, (history, instruction), arm in interleaved(items, arms, samples, sample_start):
        key = f"{study}:{arm}:{i}:{k}"
        if key in finished or i % shards != shard:
            continue
        await micake.reset(TASK)
        await replay(micake, history)
        timed.start()
        env = PlannerEnvironment(micake, settings.history_limit, settings.planner_report_reward)
        began = time.monotonic()
        episode = await engines[arm].run_episode(env, {"instruction": instruction, "max_submissions": specs[arm].max_submissions},
                                                 group_id=f"planner-{i}", role="planner", episode_id=uuid.uuid4().hex)
        wall = time.monotonic() - began
        data = episode.to_dict()
        record = {"key": key, "study": study, "arm": arm, "item": i, "sample": k, "history_len": len(history), "wall_s": wall,
                  "llm_s": llm_seconds([episode]), "eval_s": timed.eval_s, "metrics": planner_metrics(data),
                  "trace": timed.trace, "episode": data}
        append(out, record)
        logger.info("%s wasted=%d/%d best=%.4f wall=%.0fs", key, record["metrics"]["wasted"], record["metrics"]["actions"],
                    record["metrics"]["best_speedup"], wall)


def submit_call(arguments: Mapping[str, Any]) -> ToolCall:
    return ToolCall(uuid.uuid4().hex, SUBMIT_TOOL, dict(arguments), json.dumps(arguments))


async def replay(micake, history) -> None:
    for entry in history:
        await micake.step(submit_call(submission_arguments(entry["config"], entry["hypothesis"] or "replay")))


async def run_critic(cfg, micake, timed, arms, samples, out: Path) -> None:
    settings = ManagerSettings.from_dict(cfg["manager"]["settings"])
    nudges = int(cfg["rsi"]["specialist_max_no_tool_nudges"])
    specs = {arm: arm_spec(cfg, arm, "critic") for arm in arms}
    engines = {arm: AgenticRolloutEngine(build_endpoint_policy(s.endpoint, s.binding), RolloutConfig(s.max_turns, nudges)) for arm, s in specs.items()}
    finished = done_keys(out)
    for k, i, context, arm in interleaved(critic_contexts(), arms, samples):
        key = f"critic:{arm}:{i}:{k}"
        if key in finished:
            continue
        await micake.reset(TASK)
        await replay(micake, context["history"])
        context_best = float(micake.best()["geomean_speedup_vs_baseline"])
        timed.start()
        env = CriticEnvironment(micake, settings.history_limit, settings.critic_rewards)
        began = time.monotonic()
        episode = await engines[arm].run_episode(env, {"instruction": context["question"]},
                                                 group_id=context["id"], role="critic", episode_id=uuid.uuid4().hex)
        wall = time.monotonic() - began
        data = episode.to_dict()
        record = {"key": key, "study": "critic", "arm": arm, "item": i, "context": context["id"], "sample": k,
                  "context_best": context_best, "wall_s": wall, "llm_s": llm_seconds([episode]), "eval_s": timed.eval_s,
                  "metrics": critic_metrics(data, context_best), "trace": timed.trace, "episode": data}
        append(out, record)
        logger.info("%s probe=%s speedup=%.4f (context best %.4f)", key, record["metrics"]["probe_class"],
                    record["metrics"]["probe_speedup"], context_best)


async def run_search_episode(micake, budget: int) -> dict[str, Any]:
    search = GreedySearch(micake.task_packet()["baseline_config"], micake.domain)
    while micake.remaining_submissions() > 0:
        candidate = search.propose()
        if candidate is None:
            break
        result = await micake.step(submit_call(submission_arguments(candidate, "greedy coordinate move")))
        perf = (result.observation or {}).get("performance") or {}
        search.observe(candidate, perf.get("geomean_speedup_vs_baseline") if perf.get("validity") == "VALID" else None)
    return {"terminal_state": "BUDGET_EXHAUSTED" if micake.remaining_submissions() <= 0 else "SEARCH_EXHAUSTED"}


async def run_e2e(cfg, micake, timed, arms, samples, out: Path, verify_repeats: int, sample_start: int = 0) -> None:
    manager_cfg = cfg["manager"]
    settings = ManagerSettings.from_dict({**manager_cfg["settings"], "allow_direct_submit": False})
    nudges = int(cfg["rsi"]["specialist_max_no_tool_nudges"])
    manager_endpoint = Endpoint.from_dict(manager_cfg["endpoint"])
    manager_binding = PolicyBinding.from_dict("manager", manager_cfg["binding"])
    teams: dict[str, tuple[ManagerEnvironment, AgenticRolloutEngine]] = {}
    for arm in arms:
        if arm == "search":
            continue
        specialists = {}
        for role in ("planner", "critic"):
            spec = arm_spec(cfg, arm, role)
            specialists[role] = (spec, AgenticRolloutEngine(build_endpoint_policy(spec.endpoint, spec.binding), RolloutConfig(spec.max_turns, nudges)))
        env = ManagerEnvironment(micake, specialists, settings)
        engine = AgenticRolloutEngine(build_endpoint_policy(manager_endpoint, manager_binding), RolloutConfig.from_dict(manager_cfg["rollout"]))
        teams[arm] = (env, engine)
    finished = done_keys(out)
    budget = int(TASK["max_submissions"])
    for k, _, _, arm in interleaved([TASK], arms, samples, sample_start):
        key = f"e2e:{arm}:{k}"
        if key in finished:
            continue
        micake.clear_result_cache()
        timed.start()
        began = time.monotonic()
        if arm == "search":
            await micake.reset(TASK)
            summary = await run_search_episode(micake, budget)
            manager_s, specialist_s, children, episode_data = 0.0, 0.0, [], None
        else:
            env, engine = teams[arm]
            episode_id = uuid.uuid4().hex
            env.bind_episode(episode_id)
            episode = await engine.run_episode(env, TASK, group_id=TASK["task_id"], role="manager", episode_id=episode_id)
            children = list(env.child_episodes)
            manager_s, specialist_s = llm_seconds([episode]), llm_seconds(children)
            summary = {"terminal_state": episode.terminal_state, "delegations": len(children),
                       "roles": [c.role for c in children]}
            episode_data = episode.to_dict()
        wall = time.monotonic() - began
        metrics = e2e_metrics(timed.trace, budget)
        best = micake.best()
        verified = None
        if best["source"] == "agent" and verify_repeats > 0:
            verified = await remeasure(micake, best["config"], verify_repeats)
        record = {"key": key, "study": "e2e", "arm": arm, "sample": k, "wall_s": wall, "eval_s": timed.eval_s,
                  "manager_llm_s": manager_s, "specialist_llm_s": specialist_s, "metrics": metrics, "summary": summary,
                  "best": best, "verified": verified, "trace": timed.trace, "episode": episode_data,
                  "children": [c.to_dict() for c in children]}
        append(out, record)
        logger.info("%s best=%.4f wasted=%d/%d wall=%.0fs eval=%.0fs", key, metrics["final_best"], metrics["wasted"],
                    metrics["actions"], wall, timed.eval_s)


async def main_async(args) -> None:
    for spec in args.planner_arm:
        register_planner_arm(spec)
    FALLBACK["role_arm"] = args.fallback
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    kwargs = dict(cfg["micake"]["kwargs"])
    if args.gpu is not None:
        kwargs["gpu_id"] = args.gpu
    micake = build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **kwargs)
    timed = TimedMicake(micake)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.out_name or args.study}.jsonl"
    arms = args.arms.split(",")
    try:
        if args.study == "planner":
            await run_planner(cfg, micake, timed, arms, args.samples, out, args.sample_start)
        elif args.study == "planner_e2e":
            await run_planner(cfg, micake, timed, arms, args.samples, out, args.sample_start,
                              items=e2e_test_items(args.e2e_dir, args.max_items), study="planner_e2e",
                              shard=args.shard, shards=args.shards)
        elif args.study == "critic":
            await run_critic(cfg, micake, timed, arms, args.samples, out)
        else:
            await run_e2e(cfg, micake, timed, arms, args.samples, out, args.verify_repeats, args.sample_start)
    finally:
        await micake.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study", required=True, choices=("planner", "planner_e2e", "critic", "e2e"))
    parser.add_argument("--arms", default="base,rsi,dsv4")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--verify-repeats", type=int, default=2)
    parser.add_argument("--sample-start", type=int, default=0, help="first e2e sample index (for sharding)")
    parser.add_argument("--gpu", type=int, help="MICake GPU override")
    parser.add_argument("--e2e-dir", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "e2e_v1")
    parser.add_argument("--max-items", type=int, default=24, help="planner_e2e held-out contexts")
    parser.add_argument("--shard", type=int, default=0, help="planner_e2e: run items with index %% shards == shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--planner-arm", action="append", default=[],
                        help="name=port:checkpoint Planner arm (repeatable); its other roles use --fallback")
    parser.add_argument("--fallback", choices=("base", "dsv4"), default="base", help="non-Planner roles of --planner-arm arms")
    parser.add_argument("--out-name", help="results file stem (default: the study name)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "ab_eval")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
