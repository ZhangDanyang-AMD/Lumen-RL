"""Specialist child environments operating on the Manager's shared MICake session.

A child never owns MICake state: it borrows the Manager's
:class:`MicakeGemmEnvironment`, so every candidate it evaluates consumes the
shared submission budget and shows up in the shared history. The child's
step rewards are its *local* reward; the episode-level *global* reward is
assigned afterwards from the Manager episode outcome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.protocols import (
    AgentEnvironment,
    EnvironmentCapabilities,
    ResetResult,
    StepResult,
    ToolCall,
)
from sandbox.micake.environment import MicakeGemmEnvironment
from sandbox.micake.tool_runtime import (
    SUBMIT_TOOL,
    ToolArgumentError,
    parse_submission,
    tool_schemas,
)

REPORT_TOOL = "report"
DIAGNOSIS_TOOL = "report_diagnosis"
SUMMARY_LIMIT = 1500
BOTTLENECKS = (
    "occupancy_limited",
    "register_pressure",
    "lds_capacity",
    "underfilled_grid",
    "memory_bandwidth",
    "mfma_underutilized",
    "pipeline_latency",
    "tail_waste",
    "correctness_risk",
    "unknown",
)

PLANNER_PROMPT = """You are the Planner specialist in an AMD GPU kernel optimization team (MICake sandbox).

The Manager delegated a concrete search sub-task to you. You plan and submit typed GEMM schedule candidates
(FP16 x FP16 -> FP32, Triton, MI355X / gfx950, wave64) from a closed domain; MICake verifies, compiles,
checks correctness and measures speedup vs. the baseline (>1.0 is faster).

Rules:
- Act only through tools, exactly one tool call per turn.
- Follow the Manager's instruction. Use the shared history: never resubmit a configuration already tried.
- Each `submit_schedule` consumes one unit of your sub-budget. Correctness first, then speed.
- When the sub-budget is used up or you have a clear conclusion, call `report` with what you tried,
  what you learned, and the best configuration you found.
"""

CRITIC_PROMPT = """You are the Performance Critic specialist in an AMD GPU kernel optimization team (MICake sandbox).

You get the evaluated history of GEMM schedule candidates (FP16 x FP16 -> FP32, Triton, MI355X / gfx950,
wave64): compiler facts (VGPR/AGPR/SGPR, spills, LDS bytes, occupancy waves), correctness, and per-workload
speedup vs. baseline (>1.0 is faster). Diagnose the dominant bottleneck from this evidence.

Call `report_diagnosis` exactly once with:
- `bottleneck`: one category,
- `evidence`: the concrete facts that support it (cite numbers),
- `recommended_schedule` + `recommended_realization`: ONE new configuration (not yet tried) that addresses it,
- `expected_effect`: what should change and on which workloads.
MICake evaluates your recommended configuration, so it must be correct and fast, not just plausible.
"""


def compact_history(history: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Last ``limit`` submissions reduced to the facts specialists reason over."""
    rows = []
    for entry in history[-limit:]:
        result = entry["result"]
        row: dict[str, Any] = {
            "submission": entry["submission"],
            "config": entry["config"],
            "hypothesis": entry["hypothesis"],
            "stage": result.get("stage"),
        }
        for key in ("diagnostic", "compiled_facts", "correctness", "performance"):
            if key in result:
                row[key] = result[key]
        if "harness_repair" in entry:
            row["harness_repair"] = entry["harness_repair"]
        rows.append(row)
    return rows


def _shared_context(micake: MicakeGemmEnvironment, history_limit: int) -> dict[str, Any]:
    packet = micake.task_packet()
    return {
        "objective": packet["objective"],
        "target": packet["target"],
        "portfolio": packet["portfolio"],
        "baseline_config": packet["baseline_config"],
        "baseline_compiled_facts": packet["baseline_compiled_facts"],
        "domain": packet["domain"],
        "resource_limits": packet.get("resource_limits"),
        "history": compact_history(micake.history(), history_limit),
        "best": micake.best(),
        "shared_remaining_submissions": micake.remaining_submissions(),
    }


@dataclass(frozen=True)
class ChildOutcome:
    """What a child episode hands back to the Manager."""

    role: str
    terminal_state: str
    submissions: list[dict[str, Any]]
    report: dict[str, Any] | None
    best_before: float
    best_after: float
    shared_terminal: str | None

    def for_manager(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "terminal_state": self.terminal_state,
            "report": self.report,
            "submissions": self.submissions,
            "best_speedup_before": self.best_before,
            "best_speedup_after": self.best_after,
        }


class _ChildBase(AgentEnvironment):
    role = ""

    def __init__(self, micake: MicakeGemmEnvironment, history_limit: int) -> None:
        self.micake = micake
        self.history_limit = history_limit
        self.submissions: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.shared_terminal: str | None = None
        self.best_before = 1.0

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(backend=f"micake_{self.role}", snapshot=False, async_step=True)

    def _begin(self) -> None:
        self.best_before = float(self.micake.best()["geomean_speedup_vs_baseline"])

    async def _submit(self, action: ToolCall) -> StepResult:
        """Forward a submission to the shared MICake session and record it."""
        result = await self.micake.step(action)
        if result.observation.get("submission_counted", True) and result.execution_result != "schema_invalid":
            self.submissions.append({
                "submission": result.observation.get("submission"),
                "config": result.metadata.get("config"),
                "stage": result.observation.get("stage"),
                "geomean_speedup": (result.observation.get("performance") or {}).get("geomean_speedup_vs_baseline"),
                "new_best": result.observation.get("new_best"),
                "reward": result.reward,
            })
        if result.terminal_state in ("UNKNOWN_OUTCOME", "BUDGET_EXHAUSTED"):
            self.shared_terminal = result.terminal_state
        return result

    def outcome(self, terminal_state: str) -> ChildOutcome:
        return ChildOutcome(
            role=self.role,
            terminal_state=terminal_state,
            submissions=copy.deepcopy(self.submissions),
            report=copy.deepcopy(self.report),
            best_before=self.best_before,
            best_after=float(self.micake.best()["geomean_speedup_vs_baseline"]),
            shared_terminal=self.shared_terminal,
        )

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        return 0.0, self.outcome(terminal_state).for_manager()


class PlannerEnvironment(_ChildBase):
    """Planner child: submits candidates under a sub-budget, then reports."""

    role = "planner"

    def __init__(self, micake: MicakeGemmEnvironment, history_limit: int, report_reward: float) -> None:
        super().__init__(micake, history_limit)
        self.report_reward = report_reward
        self.sub_budget = 0

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        self._begin()
        self.sub_budget = min(int(task["max_submissions"]), self.micake.remaining_submissions())
        submit = next(t for t in tool_schemas(self.micake.domain) if t["function"]["name"] == SUBMIT_TOOL)
        report = {
            "type": "function",
            "function": {
                "name": REPORT_TOOL,
                "description": "End your sub-task and report findings to the Manager.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "description": "What you tried, learned and recommend."}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }
        observation = {
            "manager_instruction": task["instruction"],
            "sub_budget": self.sub_budget,
            **_shared_context(self.micake, self.history_limit),
        }
        tools = [submit, report]
        return ResetResult(PLANNER_PROMPT, observation, tools, {"role": self.role, "sub_budget": self.sub_budget, "tools": tools})

    async def step(self, action: ToolCall) -> StepResult:
        if action.name == REPORT_TOOL:
            self.report = {"summary": str((action.arguments or {}).get("summary", ""))[:SUMMARY_LIMIT]}
            return StepResult({"status": "reported"}, self.report_reward, True, "AGENT_FINISHED", "report")
        if action.name != SUBMIT_TOOL:
            return StepResult(
                {"error": "UNKNOWN_TOOL", "detail": f"available tools: {SUBMIT_TOOL}, {REPORT_TOOL}"},
                self.micake.weights.schema_invalid, False, "RUNNING", "schema_invalid",
            )
        if len(self.submissions) >= self.sub_budget:
            return StepResult(
                {"error": "SUB_BUDGET_EXHAUSTED", "detail": f"call `{REPORT_TOOL}` to hand back to the Manager"},
                self.micake.weights.schema_invalid, False, "RUNNING", "schema_invalid",
            )
        result = await self._submit(action)
        if self.shared_terminal:
            return StepResult(result.observation, result.reward, True, self.shared_terminal,
                              result.execution_result, result.receipt_refs, result.metadata)
        observation = {**result.observation, "sub_budget_remaining": self.sub_budget - len(self.submissions)}
        if len(self.submissions) >= self.sub_budget:
            observation["detail"] = "Sub-budget used up; call `report` now."
        return StepResult(observation, result.reward, False, "RUNNING",
                          result.execution_result, result.receipt_refs, result.metadata)


@dataclass(frozen=True)
class CriticRewards:
    """Local reward terms for the Critic (values come from YAML)."""

    valid_diagnosis: float
    invalid_diagnosis: float
    probe: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CriticRewards":
        return cls(float(value["valid_diagnosis"]), float(value["invalid_diagnosis"]), bool(value["probe"]))


class CriticEnvironment(_ChildBase):
    """Critic child: one structured diagnosis; its recommendation is probed in MICake."""

    role = "critic"

    def __init__(self, micake: MicakeGemmEnvironment, history_limit: int, rewards: CriticRewards) -> None:
        super().__init__(micake, history_limit)
        self.rewards = rewards

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        self._begin()
        params = next(t for t in tool_schemas(self.micake.domain) if t["function"]["name"] == SUBMIT_TOOL)
        sections = params["function"]["parameters"]["properties"]
        tool = {
            "type": "function",
            "function": {
                "name": DIAGNOSIS_TOOL,
                "description": "Report the dominant bottleneck and one recommended configuration (MICake evaluates it).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bottleneck": {"type": "string", "enum": list(BOTTLENECKS)},
                        "evidence": {"type": "string", "description": "Concrete facts from the history, with numbers."},
                        "recommended_schedule": sections["schedule"],
                        "recommended_realization": sections["realization"],
                        "expected_effect": {"type": "string", "description": "Expected change and affected workloads."},
                    },
                    "required": ["bottleneck", "evidence", "recommended_schedule", "recommended_realization", "expected_effect"],
                    "additionalProperties": False,
                },
            },
        }
        observation = {"manager_question": task["instruction"], **_shared_context(self.micake, self.history_limit)}
        return ResetResult(CRITIC_PROMPT, observation, [tool], {"role": self.role, "probe": self.rewards.probe, "tools": [tool]})

    async def step(self, action: ToolCall) -> StepResult:
        args = action.arguments
        if action.name != DIAGNOSIS_TOOL:
            return StepResult({"error": "UNKNOWN_TOOL", "detail": f"available tool: {DIAGNOSIS_TOOL}"},
                              self.rewards.invalid_diagnosis, False, "RUNNING", "schema_invalid")
        try:
            diagnosis = self._parse(args)
        except ToolArgumentError as exc:
            return StepResult({"error": exc.code, "detail": exc.detail},
                              self.rewards.invalid_diagnosis, False, "RUNNING", "schema_invalid")
        self.report = diagnosis
        reward = self.rewards.valid_diagnosis
        observation: dict[str, Any] = {"status": "diagnosis_recorded"}
        if not (self.rewards.probe and self.micake.remaining_submissions() > 0):
            return StepResult(observation, reward, True, "AGENT_FINISHED", "diagnosis")
        probe_call = ToolCall(
            call_id=f"{action.call_id}-probe",
            name=SUBMIT_TOOL,
            arguments={
                "schedule": args["recommended_schedule"],
                "realization": args["recommended_realization"],
                "hypothesis": f"[critic:{diagnosis['bottleneck']}] {diagnosis['expected_effect']}",
            },
            raw_arguments="",
        )
        probe = await self._submit(probe_call)
        observation["probe"] = probe.observation
        self.report["probe_stage"] = probe.observation.get("stage") or ("duplicate" if probe.observation.get("duplicate") else None)
        terminal = self.shared_terminal or "AGENT_FINISHED"
        return StepResult(observation, reward + probe.reward, True, terminal, "diagnosis_probed",
                          probe.receipt_refs, probe.metadata)

    def _parse(self, args: Mapping[str, Any] | None) -> dict[str, Any]:
        if args is None:
            raise ToolArgumentError("ARGUMENTS_NOT_JSON_OBJECT", "tool arguments must be a JSON object")
        expected = {"bottleneck", "evidence", "recommended_schedule", "recommended_realization", "expected_effect"}
        if set(args) != expected:
            raise ToolArgumentError("SECTION_FIELDS_INVALID", f"missing={sorted(expected - set(args))} extra={sorted(set(args) - expected)}")
        if args["bottleneck"] not in BOTTLENECKS:
            raise ToolArgumentError("VALUE_OUT_OF_DOMAIN", f"bottleneck={args['bottleneck']!r}")
        for key in ("evidence", "expected_effect"):
            if not isinstance(args[key], str) or not args[key].strip():
                raise ToolArgumentError("HYPOTHESIS_REQUIRED", f"{key} must be a non-empty string")
        submission = parse_submission(
            {"schedule": args["recommended_schedule"], "realization": args["recommended_realization"],
             "hypothesis": args["expected_effect"]},
            self.micake.domain,
        )
        return {
            "bottleneck": args["bottleneck"],
            "evidence": args["evidence"].strip()[:SUMMARY_LIMIT],
            "recommended_config": submission.config,
            "expected_effect": submission.hypothesis,
        }
