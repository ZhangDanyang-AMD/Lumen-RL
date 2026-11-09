"""One RSI generation: frozen Manager + specialist bindings -> trajectories -> data -> promotion.

    rollout (DSV4.1 Manager delegates to Planner/Critic, MICake evaluates)
      -> hierarchical credit (local + lambda * global)
      -> per-role RL / SFT datasets
      -> metrics vs. champion -> BOOTSTRAP / PROMOTE / HOLD
      -> manifest (lineage) + champion pointer

Role-adapter training consumes ``rl/<role>.jsonl``; the trained adapter is
served (ATOM merged checkpoint) and enters the next generation as challenger
through ``specialists.<role>.binding/endpoint`` in that generation's config.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Mapping

from lumenrl.agentic.core.backends import build_environment
from lumenrl.agentic.core.trajectory import Episode, TrajectoryStore
from lumenrl.agentic.rollout import AgenticRolloutEngine, AgentPolicy, RolloutConfig
from micake_rsi.debug_env import DebugSettings
from micake_rsi.export import export_specialist_data
from micake_rsi.generation import (
    MANIFEST_SCHEMA,
    Lineage,
    PromotionGate,
    config_digest,
    generation_metrics,
    git_commit,
    promotion_decision,
)
from micake_rsi.guards import (
    BASELINE,
    GuardSettings,
    harness_fingerprint,
    reverify_champion,
    verification_summary,
    verify_generation,
)
from micake_rsi.manager_env import ManagerEnvironment, ManagerSettings
from micake_rsi.rewards import HierarchicalRewardConfig, assign_specialist_credit
from micake_rsi.roles import BoundPolicy, Endpoint, PolicyBinding, build_endpoint_policy, load_role_specs
from sandbox.micake.environment import MicakeGemmEnvironment

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _endpoint_view(endpoint: Endpoint) -> dict[str, Any]:
    return {"base_url": endpoint.base_url, "model": endpoint.model, "sampling": dict(endpoint.sampling)}


def _champion_records(champion: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Best configs of the champion: from its guard records, else from its Manager episodes."""
    guards = champion.get("guards") or {}
    if guards.get("records"):
        return list(guards["records"])
    manifest = json.loads(Path(champion["manifest"]).read_text(encoding="utf-8"))
    records = []
    for row in TrajectoryStore(Path(manifest["episodes"]["path"])).load():
        if row.get("role") != "manager":
            continue
        summary = (row.get("environment") or {}).get("final_summary") or {}
        agent = summary.get("best_source") == "agent" and summary.get("best_config")
        records.append({
            "episode_id": row.get("episode_id"),
            "status": "claimed" if agent else BASELINE,
            "config": summary.get("best_config") if agent else None,
            "speedup": float(summary.get("best_speedup", 1.0)),
        })
    return records


async def _run_guards(
    micake: MicakeGemmEnvironment,
    managers: list[Episode],
    champion: Mapping[str, Any] | None,
    settings: GuardSettings,
    harness_start: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify this generation's bests; re-measure the champion when the harness differs."""
    harness_end = harness_fingerprint(settings.harness_roots)
    records = await verify_generation(micake, managers, settings)
    summary = verification_summary(records, settings)
    summary["harness_changed_during_rollout"] = harness_start["sha256"] != harness_end["sha256"]
    champion_harness = ((champion or {}).get("guards") or {}).get("harness") or {}
    if champion is not None and champion_harness.get("sha256") != harness_end["sha256"]:
        reverified = await reverify_champion(micake, _champion_records(champion), settings)
        summary["champion_mean_best_speedup"] = reverified["mean_best_speedup"]
        summary["champion_reverified"] = reverified
        logger.info("champion %s re-measured on current harness: %.4f",
                    champion["generation_id"], reverified["mean_best_speedup"] or 1.0)
    return {"harness": harness_end, "harness_at_start": harness_start, "records": records, "summary": summary}


async def run_generation(
    cfg: Mapping[str, Any],
    micake: MicakeGemmEnvironment | None = None,
    policies: Mapping[str, AgentPolicy] | None = None,
) -> dict[str, Any]:
    """Run one generation and record it in the lineage.

    Args:
        cfg: Parsed experiment config (see ``configs/gen0_planner_critic.yaml``).
        micake: Injected MICake environment (tests); built from ``cfg.micake`` when omitted.
        policies: Injected policies keyed by ``manager`` / role (tests); built from endpoints when omitted.

    Returns:
        The generation manifest.
    """
    rsi = cfg["rsi"]
    lineage = Lineage(Path(rsi["root"]))
    generation_id = rsi["generation_id"]
    gen_dir = lineage.root / generation_id
    if (gen_dir / "manifest.json").exists():
        raise FileExistsError(f"generation {generation_id} already recorded at {gen_dir}")

    micake = micake or build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **cfg["micake"]["kwargs"])
    specs = load_role_specs(cfg["specialists"])
    manager_cfg = cfg["manager"]
    manager_binding = PolicyBinding.from_dict("manager", manager_cfg["binding"])
    manager_endpoint = Endpoint.from_dict(manager_cfg["endpoint"])
    injected = dict(policies or {})
    targets = {"manager": (manager_endpoint, manager_binding), **{r: (s.endpoint, s.binding) for r, s in specs.items()}}
    policies = {
        name: BoundPolicy(injected[name], binding) if name in injected else build_endpoint_policy(endpoint, binding)
        for name, (endpoint, binding) in targets.items()
    }

    specialists = {
        role: (spec, AgenticRolloutEngine(policies[role], RolloutConfig(spec.max_turns, int(rsi["specialist_max_no_tool_nudges"]))))
        for role, spec in specs.items()
    }
    debug_cfg = cfg.get("debug")
    debug_settings = DebugSettings.from_dict(debug_cfg) if debug_cfg and debug_cfg.get("enabled") else None
    if debug_settings is not None:
        debug_binding = PolicyBinding.from_dict("debugger", {**dict(manager_cfg["binding"]), "trainable": False})
        if "debugger" in injected:
            policies["debugger"] = BoundPolicy(injected["debugger"], debug_binding)
        else:
            policies["debugger"] = build_endpoint_policy(manager_endpoint, debug_binding)
    manager_env = ManagerEnvironment(micake, specialists, ManagerSettings.from_dict(manager_cfg["settings"]))
    if debug_settings is not None:
        manager_env.attach_debug(
            AgenticRolloutEngine(policies["debugger"], RolloutConfig(debug_settings.max_turns, int(rsi["specialist_max_no_tool_nudges"]))),
            debug_settings,
        )
        manager_env.audit_dir = gen_dir / "debug"
    manager_engine = AgenticRolloutEngine(policies["manager"], RolloutConfig.from_dict(manager_cfg["rollout"]))

    store = TrajectoryStore(gen_dir / "episodes.jsonl")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.open("a", encoding="utf-8").close()
    managers: list[Episode] = []
    children: dict[str, list[Episode]] = {}
    guard_cfg = cfg.get("guards")
    guard_settings = GuardSettings.from_dict(guard_cfg) if guard_cfg and guard_cfg.get("enabled") else None
    harness_start = harness_fingerprint(guard_settings.harness_roots) if guard_settings else None
    started = dt.datetime.now(dt.timezone.utc)
    guards: dict[str, Any] | None = None
    try:
        for task in rsi["tasks"]:
            for index in range(int(rsi["episodes_per_task"])):
                episode_id = uuid.uuid4().hex
                manager_env.bind_episode(episode_id)
                logger.info("generation %s task %s episode %d/%d (%s)",
                            generation_id, task["task_id"], index + 1, rsi["episodes_per_task"], episode_id[:8])
                manager = await manager_engine.run_episode(
                    manager_env, task, group_id=task["task_id"], role="manager", episode_id=episode_id,
                )
                managers.append(manager)
                children[episode_id] = list(manager_env.child_episodes)
                store.append(manager)
                for child in children[episode_id]:
                    store.append(child)
        champion = lineage.champion()
        if guard_settings is not None:
            guards = await _run_guards(micake, managers, champion, guard_settings, harness_start)
            (gen_dir / "verification.json").write_text(json.dumps(guards, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        await manager_env.close()
        for policy in policies.values():
            await policy.close()

    credits = assign_specialist_credit(managers, children, HierarchicalRewardConfig.from_dict(rsi["rewards"]))
    datasets = export_specialist_data(gen_dir, generation_id, managers, children, credits, float(rsi["sft_min_total_reward"]))
    metrics = generation_metrics(managers, children, credits)
    promotion = promotion_decision(
        metrics, champion["metrics"] if champion else None, PromotionGate.from_dict(rsi["promotion"]),
        guards["summary"] if guards else None,
    )
    promotion["champion_generation_id"] = champion["generation_id"] if champion else None

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generation_id": generation_id,
        "parent_generation_id": rsi.get("parent_generation_id"),
        "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_digest": config_digest(cfg),
        "config": copy.deepcopy(dict(cfg)),
        "code": {"lumenrl_commit": git_commit(REPO_ROOT), "micake_commit": git_commit(cfg["micake"]["kwargs"]["micake_root"])},
        "manager": {"binding": manager_binding.__dict__, "endpoint": _endpoint_view(manager_endpoint)},
        "specialists": {
            role: {"binding": spec.binding.__dict__, "endpoint": _endpoint_view(spec.endpoint)} for role, spec in specs.items()
        },
        "evaluator": {k: managers[0].environment.get(k) for k in ("backend", "candidate_profile", "portfolio_profile", "fidelity", "measurement")} if managers else {},
        "episodes": {"path": str(store.path), "manager": len(managers), "specialist": sum(len(c) for c in children.values())},
        "datasets": datasets,
        "metrics": metrics,
        "guards": guards,
        "repairs": list(manager_env.repairs),
        "promotion": promotion,
        "next": {
            "status": "AWAITING_ROLE_ADAPTER_TRAINING",
            "train_inputs": {role: d["rl"] for role, d in datasets.items() if specs[role].binding.trainable},
            "note": "train role LoRA on rl/<role>.jsonl (recompute old log-probs locally), serve merged checkpoint, "
                    "run next generation with the new binding as challenger",
        },
    }
    path = lineage.record(manifest)
    logger.info("generation %s recorded: %s (%s)", generation_id, path, promotion["decision"])
    return manifest
