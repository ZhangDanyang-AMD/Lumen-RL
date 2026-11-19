#!/usr/bin/env python3
"""Planner-only agentic RL on the MMKCA GEMM sandbox, until the Planner beats DSV4.1 on held-out.

Every episode runs through the ``mmkca_gemm`` backend: one regime (prefill or
decode), one dtype (bf16 or fp8 blockwise), the same Planner prompt, tools and
submit budget for every arm. One generation:

1. Serve the champion and score it and DSV4.1 on the held-out Qwen3-30B-A3B GEMMs
   (DSV4.1 is frozen, so its held-out episodes are scored once and reused).
2. Roll out the champion ``samples_per_task`` times, plus DSV4.1 once, on fresh
   training bundles. An episode is kept when its best beats the median of its
   group; its improving submits and its report become SFT targets.
3. SFT a challenger from the base model on the search episodes plus every kept
   rollout so far, merge, serve, and score it on held-out.
4. Stop when the challenger (or champion) beats DSV4.1: a significantly faster
   kernel overall and per dtype, no worse per regime, and a median time to 1.04
   that is not slower. Otherwise promote it if it beats the champion.

Example:
    python experiments/micake-subagent-rsi/scripts/rsi_mmkca.py --champion-checkpoint output/mmkca-planner/g1/checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR, SCRIPTS_DIR):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from lumenrl.agentic.core.protocols import ResetResult, StepResult, ToolCall  # noqa: E402
from lumenrl.agentic.rollout import AgenticRolloutEngine, RolloutConfig  # noqa: E402
from micake_rsi.ab_eval import e2e_metrics, planner_metrics  # noqa: E402
from micake_rsi.cycle import MERGE_MANIFEST, training_config  # noqa: E402
from micake_rsi.planner_gate import PlannerGate, beats_reference, gate_decision, versus  # noqa: E402
from micake_rsi.roles import Endpoint, PolicyBinding, build_endpoint_policy  # noqa: E402
from micake_rsi.sft_data import write_role_sft  # noqa: E402
from micake_rsi.subagent_envs import PlannerEnvironment  # noqa: E402
from run_cycle import _merge, _serve, _train  # noqa: E402
from train_mmkca_planner import load_episodes  # noqa: E402
from train_rft import DATA_PARALLEL  # noqa: E402

logger = logging.getLogger("rsi_mmkca")
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "mmkca_planner.yaml"
KEY_FILE = Path("/home/danyzhan/DSV4-API.key")
SERVE_ROLE = "mmkca-planner"
GIB = 1024 ** 3
MERGE_HEADROOM = 75 * GIB
TRAIN_BUNDLE_OFFSET = 1000
UNCOUNTED = ("duplicate", "schema_invalid", "report")


def instruction(portfolio: str, dtype: str) -> str:
    return f"Find a {portfolio} {dtype} schedule that beats the baseline geomean. Start from opening."


def append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TimedPlanner(PlannerEnvironment):
    """Planner child that also records when each counted submission returned."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace: list[dict[str, Any]] = []
        self.started = time.monotonic()

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        self.started = time.monotonic()
        return await super().reset(task)

    async def step(self, action: ToolCall) -> StepResult:
        result = await super().step(action)
        counted = result.execution_result not in UNCOUNTED and result.observation.get("submission_counted", True)
        performance = result.observation.get("performance") or {}
        self.trace.append({"index": len([r for r in self.trace if r["counted"]]) + int(bool(counted)),
                           "t": time.monotonic() - self.started, "stage": result.execution_result,
                           "speedup": performance.get("geomean_speedup_vs_baseline"), "counted": bool(counted)})
        return result


class Arms:
    """Policies by arm name; the planner arm points at whatever checkpoint is being served."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        dsv4 = cfg["manager"]["endpoint"]
        self.policies = {"dsv4": build_endpoint_policy(
            Endpoint.from_dict(dsv4),
            PolicyBinding("planner", dsv4["model"], "none", "frozen", "dsv41-frozen", False))}

    def serve(self, name: str, checkpoint: Path) -> None:
        planner = self.cfg["planner"]
        port = int(self.cfg["rsi"]["serve"]["port"])
        _serve(self.cfg["train"]["serve"]["image"], {"role": SERVE_ROLE, "checkpoint": str(checkpoint), "port": port,
                                                     "gpus": str(self.cfg["rsi"]["serve"]["gpus"]), "tp": 1})
        endpoint = Endpoint.from_dict({**planner["endpoint"], "base_url": f"http://127.0.0.1:{port}/v1", "model": str(checkpoint)})
        binding = PolicyBinding.from_dict("planner", {**planner["binding"], "policy_version": name, "adapter_version": name})
        self.policies = {"dsv4": self.policies["dsv4"], name: build_endpoint_policy(endpoint, binding)}

    @staticmethod
    def stop_serving() -> None:
        subprocess.run(["docker", "rm", "-f", f"atom-micake-{SERVE_ROLE}"], check=False, capture_output=True)


async def run_job(env, policy, cfg: Mapping[str, Any], job: Mapping[str, Any], verify: bool) -> dict[str, Any]:
    rsi, settings = cfg["rsi"], cfg["manager"]["settings"]
    task = {"portfolio": job["portfolio"], "dtype": job["dtype"], "split": job["split"], "bundle": job.get("bundle"),
            "max_submissions": int(rsi["max_submissions"])}
    await env.reset(task)
    planner = TimedPlanner(env, int(settings["history_limit"]), float(settings["planner_report_reward"]))
    engine = AgenticRolloutEngine(policy, RolloutConfig(int(rsi["max_turns"]), int(rsi["max_no_tool_nudges"])))
    began = time.monotonic()
    episode = await engine.run_episode(planner, {"instruction": instruction(job["portfolio"], job["dtype"]),
                                                 "max_submissions": int(rsi["max_submissions"])},
                                       group_id=job["group"], role="planner", episode_id=uuid.uuid4().hex)
    wall = time.monotonic() - began
    data = episode.to_dict()
    best = env.best()
    record = {**job, "wall_s": wall, "trace": planner.trace, "best": {"source": best["source"],
              "speedup": best["geomean_speedup_vs_baseline"], "config": best["config"]},
              "metrics": {**planner_metrics(data), **e2e_metrics(planner.trace, int(rsi["max_submissions"]))},
              "episode": data}
    if verify and best["source"] == "agent":
        again = await env.measure(best["config"])
        speedup = (again.get("performance") or {}).get("geomean_speedup")
        record["verified"] = {"stage": again.get("stage"), "speedup": min(float(speedup), float(best["geomean_speedup_vs_baseline"]))
                              if again.get("stage") == "measured" and speedup else 1.0}
    return record


async def run_jobs(cfg: Mapping[str, Any], arms: Arms, jobs: Sequence[dict[str, Any]], out: Path, verify: bool) -> list[dict[str, Any]]:
    """Run ``jobs`` over one MMKCA environment per worker GPU; finished keys are skipped."""
    done = {r["key"] for r in read_records(out)}
    pending: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        if job["key"] not in done:
            pending.put_nowait(job)
    if pending.empty():
        return read_records(out)
    module, backend = cfg["micake"]["module"], cfg["micake"]["backend"]
    gpus = [int(g) for g in cfg["rsi"]["workers"]]
    envs = [build_environment(backend, module, **{**cfg["micake"]["kwargs"], "gpu_id": gpu}) for gpu in gpus]

    async def worker(env) -> None:
        while not pending.empty():
            job = pending.get_nowait()
            try:
                record = await run_job(env, arms.policies[job["arm"]], cfg, job, verify)
            except Exception:  # noqa: BLE001 - one bad episode must not stop the generation
                logger.exception("%s failed", job["key"])
                continue
            append(out, record)
            m = record["metrics"]
            found = (m.get("best_at") or {}).get("seconds")
            logger.info("%s best=%.4f t_best=%s verified=%s wasted=%d/%d wall=%.0fs", job["key"], m["final_best"],
                        "baseline" if found is None else f"{found:.0f}s", (record.get("verified") or {}).get("speedup"),
                        m["wasted"], m["actions"], record["wall_s"])

    try:
        await asyncio.gather(*(worker(env) for env in envs))
    finally:
        await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
    return read_records(out)


def heldout_jobs(cfg: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    jobs = []
    for repeat in range(int(cfg["rsi"]["heldout_repeats"])):
        for portfolio, dtype in cfg["rsi"]["cells"]:
            jobs.append({"key": f"heldout:{arm}:{portfolio}:{dtype}:{repeat}", "arm": arm, "portfolio": portfolio,
                         "dtype": dtype, "split": "heldout", "group": f"heldout:{portfolio}:{dtype}", "repeat": repeat})
    return jobs


def rollout_jobs(cfg: Mapping[str, Any], generation: int, champion: str) -> list[dict[str, Any]]:
    rsi = cfg["rsi"]
    jobs = []
    per_cell = int(rsi["bundles_per_cell"])
    for portfolio, dtype in rsi["cells"]:
        for offset in range(per_cell):
            bundle = TRAIN_BUNDLE_OFFSET + generation * per_cell + offset
            group = f"train:{portfolio}:{dtype}:{bundle}"
            arms = [(champion, k) for k in range(int(rsi["samples_per_task"]))] + [("dsv4", 0)]
            for arm, sample in arms:
                jobs.append({"key": f"{group}:{arm}:{sample}", "arm": arm, "portfolio": portfolio, "dtype": dtype,
                             "split": "train", "bundle": bundle, "group": group, "sample": sample})
    return jobs


def kept_samples(records: Sequence[Mapping[str, Any]], min_gain: float) -> list[dict[str, Any]]:
    """Group-relative rejection sampling: episodes above their group median, improving turns only."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    samples = []
    for group in groups.values():
        bests = [float(r["metrics"]["final_best"]) for r in group]
        median, top = statistics.median(bests), max(bests)
        for record, best in zip(group, bests):
            if best <= 1.0 or (best < median * (1.0 + min_gain) and best < top):
                continue
            sample = episode_sample(record)
            if sample is not None:
                samples.append(sample)
    return samples


def episode_sample(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """``write_role_sft`` sample: +1 on submits that set a new best and on the closing report."""
    episode = record["episode"]
    messages = episode["messages"]
    if any(len(m.get("tool_calls") or ()) > 1 for m in messages):
        return None
    assistant = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    steps = []
    for step in episode["steps"]:
        observation = step["result"].get("observation") or {}
        name = step["action"]["name"]
        improving = step["result"].get("execution_result") == "measured" and bool(observation.get("new_best"))
        steps.append({"step_id": step["step_index"], "turn_index": step["turn_index"],
                      "assistant_message_index": assistant[step["turn_index"]], "tool": name,
                      "execution_result": step["result"].get("execution_result"),
                      "advantage": 1.0 if improving or name == "report" else -1.0})
    if not any(s["advantage"] > 0 and s["tool"] != "report" for s in steps):
        return None
    return {"role": "planner", "episode_id": episode["episode_id"], "source_key": record["key"], "messages": messages,
            "tools": (episode.get("environment") or {}).get("tools"), "steps": steps}


def legality_samples(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One training target per illegal submit: the nearest verifier-legal schedule in that context.

    The illegal tool call is not part of the prefix, so the loss is on producing the legal one
    instead of it. The tool schema only offers MFMA instructions legal for the episode dtype.
    """
    scripts = Path("/home/danyzhan/MICake-MMKCA/scripts")
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from gemm_portfolios import legal_mfma, repair_schedule, schedule_violation

    episode = record["episode"]
    messages = episode["messages"]
    tools = (episode.get("environment") or {}).get("tools")
    if any(len(m.get("tool_calls") or ()) > 1 for m in messages) or not tools or len(messages) < 2:
        return []
    try:
        rows = json.loads(messages[1]["content"]).get("portfolio") or []
    except (json.JSONDecodeError, TypeError):
        return []
    if not rows:
        return []
    dtype = record["dtype"]
    assistant = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    visible = json.loads(json.dumps(tools))
    for tool in visible:
        props = (((tool.get("function") or {}).get("parameters") or {}).get("properties") or {})
        mfma = ((props.get("realization") or {}).get("properties") or {}).get("mfma")
        if mfma is not None:
            mfma["enum"] = legal_mfma(dtype)
    samples = []
    for step in episode["steps"]:
        if step["result"].get("execution_result") != "verify_fail" or step["action"]["name"] != "submit_schedule":
            continue
        arguments = step["action"].get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        config = {"schedule": arguments.get("schedule") or {}, "realization": arguments.get("realization") or {}}
        if schedule_violation(config, dtype, rows) is None:
            continue
        repaired = repair_schedule(config, dtype, rows)
        if repaired is None:
            continue
        index = assistant[step["turn_index"]]
        reason = schedule_violation(config, dtype, rows)
        call = {"id": f"legal-{step['step_index']}", "type": "function", "function": {
            "name": "submit_schedule",
            "arguments": json.dumps({"schedule": repaired["schedule"], "realization": repaired["realization"],
                                     "hypothesis": f"{reason}. Submitting the nearest legal schedule."}),
        }}
        samples.append({
            "role": "planner", "episode_id": f"{episode['episode_id']}-legal-{step['step_index']}",
            "source_key": f"{record['key']}:legal:{step['step_index']}",
            "messages": list(messages[:index]) + [{"role": "assistant", "content": reason, "tool_calls": [call]}],
            "tools": visible,
            "steps": [{"step_id": 0, "turn_index": 0, "assistant_message_index": index, "tool": "submit_schedule",
                       "execution_result": "legal_repair", "advantage": 1.0}],
        })
    return samples


def train_challenger(cfg: Mapping[str, Any], gen_dir: Path, samples: list[dict[str, Any]]) -> Path:
    train = cfg["train"]
    checkpoint = gen_dir / "checkpoint"
    if (checkpoint / MERGE_MANIFEST).exists():
        return checkpoint
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(train["base_model"], trust_remote_code=True)
    data_dir = gen_dir / "train" / "planner"
    manifest = write_role_sft(samples, tokenizer, data_dir, min_advantage=0.0,
                              max_sequence_length=int(train["max_sequence_length"]),
                              tokenizer_sha256=train["tokenizer_sha256"], tokenizer_path=train["base_model"])
    rows = manifest["artifacts"]["train"]["rows"]
    steps = max(1, math.ceil(float(cfg["rsi"]["epochs"]) * rows / DATA_PARALLEL))
    config = training_config({**train, "max_steps": steps}, role="planner", base_model=train["base_model"],
                             data_dir=data_dir, manifest=manifest)
    config_path = data_dir / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger.info("training %s: %d samples, %d rows, %d steps", gen_dir.name, len(samples), rows, steps)
    Arms.stop_serving()
    _train(config_path, train["image"], DATA_PARALLEL)
    free = shutil.disk_usage(gen_dir).free
    if free < MERGE_HEADROOM:
        raise SystemExit(f"disk free {free / GIB:.0f} GiB is below {MERGE_HEADROOM / GIB:.0f} GiB; not merging")
    _merge(train["image"], Path(train["base_model"]), Path(config["output"]["adapter_dir"]), checkpoint)
    return checkpoint


def per_regime_not_worse(records: Sequence[Mapping[str, Any]], name: str, reference: str) -> bool:
    for portfolio in ("decode", "prefill"):
        group = [r for r in records if r["portfolio"] == portfolio]
        compared = versus(group, name, reference, 1.04).get("best") or {}
        if not compared or compared.get("rsi", 0.0) < compared.get("base", 0.0):
            return False
    return True


def log_comparison(title: str, records: Sequence[Mapping[str, Any]], name: str, reference: str) -> dict[str, Any]:
    table = {}
    for label, group in [("all", records)] + [(f"{p}/{d}", [r for r in records if r["portfolio"] == p and r["dtype"] == d])
                                             for p in ("decode", "prefill") for d in ("bf16", "fp8")]:
        compared = versus(group, name, reference, 1.04)
        best = compared.get("best") or {}
        if not best:
            continue
        table[label] = {"arm": best.get("rsi"), "reference": best.get("base"), "p": best.get("p"),
                        "t_best_arm": compared["arm"]["median_time_to_best"], "t_best_ref": compared["reference"]["median_time_to_best"],
                        "t104_arm": compared["arm"]["median_time_to_tau"], "t104_ref": compared["reference"]["median_time_to_tau"]}
        logger.info("%s %-13s %s %.4f vs %s %.4f (p=%.3f) | median t_best %.0fs vs %.0fs | median t1.04 %.0fs vs %.0fs",
                    title, label, name, best["rsi"], reference, best["base"], best.get("p", float("nan")),
                    table[label]["t_best_arm"], table[label]["t_best_ref"], table[label]["t104_arm"], table[label]["t104_ref"])
    return table


def load_state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(root: Path, state: Mapping[str, Any]) -> None:
    (root / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


async def heldout(cfg, arms: Arms, root: Path, name: str) -> list[dict[str, Any]]:
    """Held-out records of ``name`` and DSV4.1 (DSV4.1 is scored once)."""
    dsv4 = await run_jobs(cfg, arms, heldout_jobs(cfg, "dsv4"), root / "heldout" / "dsv4.jsonl", verify=True)
    mine = await run_jobs(cfg, arms, heldout_jobs(cfg, name), root / "heldout" / f"{name}.jsonl", verify=True)
    expected = len(heldout_jobs(cfg, name))
    for arm, rows in (("dsv4", dsv4), (name, mine)):
        if len(rows) < expected:
            raise SystemExit(f"{arm}: {len(rows)}/{expected} held-out episodes finished; see the log before rerunning")
    return dsv4 + mine


async def main_async(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = (REPO_ROOT / cfg["rsi"]["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gate = PlannerGate()
    state = load_state(root)
    if not state:
        state = {"champion": {"name": args.champion, "checkpoint": str(args.champion_checkpoint.resolve())}, "generations": []}
        save_state(root, state)
    arms = Arms(cfg)
    min_gain = float(cfg["rsi"]["min_group_gain"])
    search = load_episodes(Path(cfg["rsi"]["search_data"]))
    generations = itertools.count() if args.generations is None else range(args.generations)
    for _ in generations:
        generation = len(state["generations"])
        champion = state["champion"]
        arms.serve(champion["name"], Path(champion["checkpoint"]))
        records = await heldout(cfg, arms, root, champion["name"])
        log_comparison(f"held-out champion", records, champion["name"], "dsv4")
        if beats_reference(records, champion["name"], gate) and per_regime_not_worse(records, champion["name"], "dsv4"):
            logger.info("champion %s beats DSV4.1 on held-out; stopping", champion["name"])
            state["beats_dsv4"] = champion["name"]
            save_state(root, state)
            return

        gen_dir = root / f"g{generation}"
        jobs = rollout_jobs(cfg, generation, champion["name"])
        rollouts = await run_jobs(cfg, arms, jobs, gen_dir / "rollouts.jsonl", verify=False)
        if len(rollouts) < 0.9 * len(jobs):
            raise SystemExit(f"g{generation}: {len(rollouts)}/{len(jobs)} rollouts finished; see the log before rerunning")
        kept = kept_samples(rollouts, min_gain)
        (gen_dir / "kept.jsonl").write_text("".join(json.dumps(s) + "\n" for s in kept), encoding="utf-8")
        pool = [s for g in range(generation + 1) for s in read_records(root / f"g{g}" / "kept.jsonl")]
        legal = [sample for g in range(generation + 1) for record in read_records(root / f"g{g}" / "rollouts.jsonl")
                 for sample in legality_samples(record)]
        by_arm = {arm: sum(1 for s in kept if s["source_key"].split(":")[-2] == arm) for arm in (champion["name"], "dsv4")}
        logger.info("g%d: %d rollouts, %d kept (%s), pool %d, search %d, legal repairs %d", generation, len(rollouts),
                    len(kept), by_arm, len(pool), len(search), len(legal))
        name = f"rl{generation}"
        checkpoint = train_challenger(cfg, gen_dir, search + pool * int(args.rl_repeat) + legal)
        arms.serve(name, checkpoint)
        records = await heldout(cfg, arms, root, name)
        table = log_comparison("held-out challenger", records, name, "dsv4")
        champion_records = read_records(root / "heldout" / f"{champion['name']}.jsonl")
        decision = gate_decision(champion_records + [r for r in records if r["arm"] == name], champion["name"], name, gate)
        beats = beats_reference(records, name, gate) and per_regime_not_worse(records, name, "dsv4")
        state["generations"].append({"name": name, "checkpoint": str(checkpoint), "parent": champion["name"],
                                     "rollouts": len(rollouts), "kept": len(kept), "pool": len(pool), "vs_dsv4": table,
                                     "decision": decision["decision"], "reasons": decision["reasons"], "beats_dsv4": beats})
        logger.info("g%d %s vs champion %s: %s %s", generation, name, champion["name"], decision["decision"], decision["reasons"])
        if beats or decision["decision"] == "PROMOTE":
            old = Path(champion["checkpoint"])
            state["champion"] = {"name": name, "checkpoint": str(checkpoint)}
            if old.resolve() != checkpoint.resolve() and old.is_relative_to(REPO_ROOT / "output"):
                shutil.rmtree(old, ignore_errors=True)
        else:
            shutil.rmtree(checkpoint, ignore_errors=True)
        save_state(root, state)
        if beats:
            state["beats_dsv4"] = name
            save_state(root, state)
            logger.info("%s beats DSV4.1 on held-out; stopping", name)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--champion", default="sft1")
    parser.add_argument("--champion-checkpoint", type=Path, default=REPO_ROOT / "output" / "mmkca-planner" / "g1" / "checkpoint")
    parser.add_argument("--generations", type=int, default=None,
                        help="cap on generations; omitted means run until a challenger beats DSV4.1")
    parser.add_argument("--rl-repeat", type=int, default=2, help="copies of each kept rollout in the SFT mix")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not os.environ.get("DSV4_API_KEY"):
        match = re.search(r"[0-9a-f]{64}", KEY_FILE.read_text(encoding="utf-8"))
        if match is None:
            raise SystemExit("DSV4 key file has no key")
        os.environ["DSV4_API_KEY"] = match.group(0)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
