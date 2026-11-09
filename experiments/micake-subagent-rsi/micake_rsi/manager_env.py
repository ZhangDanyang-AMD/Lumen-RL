"""Manager environment: frozen DSV4.1 drives MICake directly or delegates to specialists.

The Manager is not trained. Its environment is the shared MICake session
plus a ``delegate`` tool; each delegation runs a full specialist child
episode (its own policy, prompt and tools) against the same session and
returns the child's report as the tool observation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lumenrl.agentic.core.protocols import (
    AgentEnvironment,
    EnvironmentCapabilities,
    ResetResult,
    StepResult,
    ToolCall,
)
from lumenrl.agentic.core.trajectory import Episode
from lumenrl.agentic.rollout import AgenticRolloutEngine
from micake_rsi.debug_env import DebugEnvironment, DebugSettings
from micake_rsi.roles import RoleSpec
from micake_rsi.subagent_envs import CriticEnvironment, CriticRewards, PlannerEnvironment
from sandbox.micake.environment import UNKNOWN_STAGES, MicakeGemmEnvironment
from sandbox.micake.tool_runtime import SUBMIT_TOOL

logger = logging.getLogger(__name__)

DELEGATE_TOOL = "delegate"

MANAGER_PROMPT = """You are the Manager of an AMD GPU kernel optimization team working inside the MICake sandbox.

Goal: find a GEMM schedule (FP16 x FP16 -> FP32, Triton, MI355X / gfx950, wave64) that is correct on every
workload and as fast as possible vs. the baseline. All candidates share one submission budget.

Your team (call `delegate`):
- planner: give it a focused search direction (e.g. "explore larger N tiles with GROUPED_M for the wide
  workload"). It submits a few candidates from the shared budget and reports what it found.
- critic: ask it to diagnose the evaluated history. It names the dominant bottleneck with evidence and
  recommends one new configuration, which MICake evaluates immediately (one budget unit).
{submit_rule} Call exactly one tool per turn.
Typical loop: let the planner explore, ask the critic to diagnose, direct the planner with the diagnosis.
Call `finish` with a summary when the budget is spent or further work is unlikely to help.
If a submission returns an unknown harness failure, you are paused into a debug session: read and patch
MICake under the editable roots, then that same submission is retried once. Candidate failures are not patched.
"""


@dataclass(frozen=True)
class ManagerSettings:
    """Manager tool policy (values come from YAML)."""

    allow_direct_submit: bool
    max_delegations: int
    history_limit: int
    planner_report_reward: float
    critic_rewards: CriticRewards

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManagerSettings":
        return cls(
            allow_direct_submit=bool(value["allow_direct_submit"]),
            max_delegations=int(value["max_delegations"]),
            history_limit=int(value["history_limit"]),
            planner_report_reward=float(value["planner_report_reward"]),
            critic_rewards=CriticRewards.from_dict(value["critic_rewards"]),
        )


class ManagerEnvironment(AgentEnvironment):
    """Shared MICake session + delegation to specialist child episodes."""

    def __init__(
        self,
        micake: MicakeGemmEnvironment,
        specialists: Mapping[str, tuple[RoleSpec, AgenticRolloutEngine]],
        settings: ManagerSettings,
    ) -> None:
        self.micake = micake
        self.specialists = dict(specialists)
        self.settings = settings
        self.episode_id = ""
        self.child_episodes: list[Episode] = []
        self._repairs = 0
        self._debug_engine: AgenticRolloutEngine | None = None
        self._debug_settings: DebugSettings | None = None
        self.repairs: list[dict[str, Any]] = []
        self.audit_dir: Path | None = None

    def bind_episode(self, episode_id: str) -> None:
        """Set the Manager episode id used as ``parent_episode_id`` of children."""
        self.episode_id = episode_id
        self.child_episodes = []
        self._repairs = 0

    def attach_debug(self, engine: AgenticRolloutEngine, settings: DebugSettings) -> None:
        """Run ``engine`` (frozen DSV4.1) when MICake returns an unknown outcome."""
        self._debug_engine = engine
        self._debug_settings = settings
        self.micake.on_unknown = self._on_unknown

    async def _on_unknown(self, failure: Mapping[str, Any]) -> dict[str, Any]:
        settings = self._debug_settings
        engine = self._debug_engine
        if settings is None or engine is None or self._repairs >= settings.max_repairs_per_episode:
            logger.info("debug repair skipped; budget exhausted or debug is not attached")
            return {"retry": False, "summary": "repair budget exhausted"}
        self._repairs += 1
        env = DebugEnvironment(settings, failure)
        episode = await engine.run_episode(
            env, {"failure_stage": failure.get("stage")}, group_id=self.episode_id,
            parent_episode_id=self.episode_id, role="debugger",
        )
        self.child_episodes.append(episode)
        logger.info("debug %s retry=%s files=%s", episode.episode_id[:8], env.retry, env.files)
        record: dict[str, Any] = {
            "manager_episode_id": self.episode_id,
            "debug_episode_id": episode.episode_id,
            "failure_stage": failure.get("stage"),
            "files": list(env.files),
            "retry": env.retry,
            "summary": env.summary,
        }
        diff = env.diff()
        if diff and self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            patch = self.audit_dir / f"{episode.episode_id}.diff"
            patch.write_text(diff, encoding="utf-8")
            record["patch"] = str(patch)
        self.repairs.append(record)
        if not env.retry:
            record["outcome"] = "rolled_back" if env.rollback() else "no_change"
            return {"retry": False, "summary": env.summary, "files": list(env.files)}

        async def settle(stage: str) -> None:
            if stage in UNKNOWN_STAGES:
                record["outcome"] = "rolled_back"
                record["rolled_back"] = env.rollback()
            else:
                record["outcome"] = "kept"
            record["stage_after"] = stage

        return {"retry": True, "summary": env.summary, "files": list(env.files), "settle": settle}

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(backend="micake_manager", snapshot=False, async_step=True)

    def _delegate_schema(self) -> dict[str, Any]:
        budgets = {role: spec.max_submissions for role, (spec, _) in self.specialists.items()}
        return {
            "type": "function",
            "function": {
                "name": DELEGATE_TOOL,
                "description": (
                    "Run a specialist on the shared MICake session and get its report. "
                    f"Per-delegation submission budgets: {budgets}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": sorted(self.specialists)},
                        "instruction": {"type": "string", "description": "Concrete sub-task or question for the specialist."},
                    },
                    "required": ["role", "instruction"],
                    "additionalProperties": False,
                },
            },
        }

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        reset = await self.micake.reset(task)
        tools = [t for t in reset.tools if self.settings.allow_direct_submit or t["function"]["name"] != SUBMIT_TOOL]
        tools.append(self._delegate_schema())
        observation = {**reset.observation, "team": sorted(self.specialists), "max_delegations": self.settings.max_delegations}
        metadata = {**reset.metadata, "manager": {"allow_direct_submit": self.settings.allow_direct_submit,
                                                  "max_delegations": self.settings.max_delegations}}
        submit_rule = (
            "You may also call `submit_schedule` yourself."
            if self.settings.allow_direct_submit
            else "Every candidate must come from a specialist: delegate all submissions."
        )
        return ResetResult(MANAGER_PROMPT.format(submit_rule=submit_rule), observation, tools, metadata)

    async def step(self, action: ToolCall) -> StepResult:
        if action.name == DELEGATE_TOOL:
            return await self._delegate(action.arguments)
        if action.name == SUBMIT_TOOL and not self.settings.allow_direct_submit:
            return StepResult({"error": "UNKNOWN_TOOL", "detail": "direct submission disabled; delegate instead"},
                              0.0, False, "RUNNING", "schema_invalid")
        return await self.micake.step(action)

    async def _delegate(self, args: Mapping[str, Any] | None) -> StepResult:
        role = (args or {}).get("role")
        instruction = (args or {}).get("instruction")
        if role not in self.specialists or not isinstance(instruction, str) or not instruction.strip():
            return StepResult({"error": "DELEGATE_ARGUMENTS_INVALID", "detail": f"role in {sorted(self.specialists)}, non-empty instruction"},
                              0.0, False, "RUNNING", "schema_invalid")
        if len(self.child_episodes) >= self.settings.max_delegations:
            return StepResult({"error": "DELEGATION_LIMIT", "detail": "no delegations left; submit or finish"},
                              0.0, False, "RUNNING", "schema_invalid")
        if self.micake.remaining_submissions() <= 0:
            return StepResult({"error": "BUDGET_EXHAUSTED"}, 0.0, True, "BUDGET_EXHAUSTED", "budget_exhausted")

        spec, engine = self.specialists[role]
        if role == "planner":
            child = PlannerEnvironment(self.micake, self.settings.history_limit, self.settings.planner_report_reward)
            task = {"instruction": instruction.strip(), "max_submissions": spec.max_submissions}
        else:
            child = CriticEnvironment(self.micake, self.settings.history_limit, self.settings.critic_rewards)
            task = {"instruction": instruction.strip()}
        episode = await engine.run_episode(
            child, task, group_id=self.episode_id, parent_episode_id=self.episode_id, role=role,
        )
        self.child_episodes.append(episode)
        outcome = child.outcome(episode.terminal_state)
        observation = {**outcome.for_manager(), "remaining_submissions": self.micake.remaining_submissions(),
                       "best": self.micake.best(), "child_episode_id": episode.episode_id}
        logger.info("delegation %s -> %s: %s, %d submissions, best %.4f",
                    role, episode.episode_id[:8], episode.terminal_state, len(outcome.submissions), outcome.best_after)
        if outcome.shared_terminal:
            return StepResult(observation, 0.0, True, outcome.shared_terminal, f"delegate_{role}")
        return StepResult(observation, 0.0, False, "RUNNING", f"delegate_{role}")

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        reward, summary = await self.micake.finalize(terminal_state)
        summary["child_episode_ids"] = [ep.episode_id for ep in self.child_episodes]
        summary["delegations"] = [ep.role for ep in self.child_episodes]
        return reward, summary

    async def close(self) -> None:
        await self.micake.close()
