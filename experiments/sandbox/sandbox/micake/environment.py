"""MICake GEMM schedule-optimization environment for LumenRL AgenticRL.

One episode = the agent iteratively submits typed C144 GEMM candidates;
MICake verifies, compiles for gfx950, checks GPU correctness and optionally
measures against the baseline. Rewards are correctness-gated and cost-aware.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import uuid
from dataclasses import dataclass, fields
from typing import Any, Awaitable, Callable, Mapping

from lumenrl.agentic.core.protocols import (
    AgentEnvironment,
    EnvironmentCapabilities,
    ResetResult,
    StepResult,
    ToolCall,
)
from sandbox.micake.evaluator import UNKNOWN_STAGES, MicakeWorkerClient
from sandbox.micake.receipts import receipts_for
from sandbox.micake.tool_runtime import (
    FINISH_TOOL,
    SUBMIT_TOOL,
    ToolArgumentError,
    parse_submission,
    tool_schemas,
)

logger = logging.getLogger(__name__)

BACKEND_NAME = "micake_gemm"
FIDELITY_LEVELS = ("verify", "compile", "correctness", "performance")

SYSTEM_PROMPT = """You are an AMD GPU kernel optimization agent working inside the MICake sandbox.

Goal: find a GEMM schedule (FP16 x FP16 -> FP32, Triton, AMD MI355X / gfx950, wave64) that is correct on
every workload in the portfolio and as fast as possible relative to the baseline schedule.

How to work:
- You can only act through tools. Call exactly one tool per turn.
- `submit_schedule` takes a typed schedule + realization from a closed domain and a short hypothesis.
  MICake (not you) generates the kernel source, compiles it, checks correctness and measures it.
- Read each result: compiler facts (VGPR/AGPR/SGPR, spills, shared memory bytes, occupancy waves),
  correctness per workload, and speedup vs. baseline (>1.0 is faster). Use them to form the next hypothesis.
- Correctness comes before performance. Avoid resubmitting a configuration you already tried.
- Large shared-memory use or spills usually limit occupancy; small tiles underfill large problems.
- `resource_limits.max_group_segment_bytes` is the gfx950 LDS cap. A kernel above it is rejected before launch and still consumes one submission.
- Call `finish` with a short summary when further submissions are unlikely to help.
"""


@dataclass(frozen=True)
class RewardWeights:
    """Correctness-gated, cost-aware turn rewards (values come from YAML)."""

    schema_invalid: float
    verify_fail: float
    compile_fail: float
    compile_pass: float
    correctness_fail: float
    correctness_pass: float
    performance_log_clip: float
    new_best_bonus: float
    repeat_penalty: float
    unknown_outcome: float
    terminal_best_speedup: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardWeights":
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError(f"reward keys mismatch: missing={expected - set(value)} extra={set(value) - expected}")
        return cls(**{key: float(value[key]) for key in expected})


def _clipped_log(speedup: float, clip: float) -> float:
    return max(-clip, min(clip, math.log(speedup)))


class MicakeGemmEnvironment(AgentEnvironment):
    """AgentEnvironment backed by MICake's C144 typed GEMM workflow."""

    def __init__(
        self,
        micake_root: str,
        micake_python: str,
        gpu_id: int,
        rocm_path: str,
        request_timeout_s: float,
        fidelity: str,
        max_submissions: int,
        measurement: Mapping[str, int],
        reward: Mapping[str, float],
        evaluator: MicakeWorkerClient | None = None,
        new_best_min_gain: float = 0.0,
    ) -> None:
        """Args:
            micake_root: MICake checkout (owner of verifier/compiler/GPU evaluator).
            micake_python: Python interpreter with MICake's pinned Triton.
            gpu_id: Physical GPU exposed to the worker via ``HIP_VISIBLE_DEVICES``.
            rocm_path: ROCm installation root.
            request_timeout_s: Per-evaluation timeout; expiry is an unknown outcome.
            fidelity: Default highest stage: verify | compile | correctness | performance.
            max_submissions: Default submit_schedule budget per episode.
            measurement: Paired AB/BA protocol for the performance stage.
            reward: :class:`RewardWeights` values.
            evaluator: Injected worker client (tests); built from the above when omitted.
            new_best_min_gain: Relative margin over the current best a speedup must clear to
                become the new best (run-to-run noise band).
        """
        if fidelity not in FIDELITY_LEVELS:
            raise ValueError(f"fidelity must be one of {FIDELITY_LEVELS}")
        self._default_fidelity = fidelity
        self._default_max_submissions = max_submissions
        self._measurement = dict(measurement)
        self._weights = RewardWeights.from_dict(reward)
        self._new_best_min_gain = float(new_best_min_gain)
        self._evaluator = evaluator or MicakeWorkerClient(
            micake_root, micake_python, gpu_id, request_timeout_s, rocm_path,
        )
        self._description: Mapping[str, Any] | None = None
        self._result_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._state: dict[str, Any] = {}
        self.on_unknown: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] | None = None

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            backend=BACKEND_NAME, snapshot=True, async_step=True, fidelity_levels=FIDELITY_LEVELS,
        )

    async def _evaluate(self, config: Mapping[str, Any], fidelity: str) -> tuple[dict[str, Any], bool]:
        key = (json.dumps(config, sort_keys=True), fidelity)
        if key in self._result_cache:
            return copy.deepcopy(self._result_cache[key]), True
        result = await self._evaluator.evaluate(config, fidelity, self._measurement)
        if result.get("stage") not in UNKNOWN_STAGES:
            self._result_cache[key] = copy.deepcopy(result)
        return result, False

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        if self._description is None:
            self._description = await self._evaluator.describe()
        description = self._description
        fidelity = task.get("fidelity", self._default_fidelity)
        if fidelity not in FIDELITY_LEVELS:
            raise ValueError(f"task fidelity {fidelity!r} not in {FIDELITY_LEVELS}")
        max_submissions = int(task.get("max_submissions", self._default_max_submissions))
        baseline, _ = await self._evaluate(description["baseline_config"], "compile")
        self._state = {
            "task_id": task.get("task_id", "c144-gemm-portfolio"),
            "fidelity": fidelity,
            "max_submissions": max_submissions,
            "submissions": 0,
            "seen": {},
            "best": {"speedup": 1.0, "config": description["baseline_config"], "source": "baseline"},
            "correct_count": 0,
            "finished_summary": None,
            "history": [],
        }
        self._state["task_packet"] = observation = {
            "task_id": self._state["task_id"],
            "objective": "Maximize geomean speedup vs. baseline across all workloads while staying correct.",
            "target": description["target"],
            "portfolio": description["portfolio"],
            "baseline_config": description["baseline_config"],
            "baseline_compiled_facts": baseline.get("compiled_facts"),
            "domain": description["domain"],
            "resource_limits": description.get("resource_limits"),
            "evaluation_fidelity": fidelity,
            "budget": {"max_submissions": max_submissions},
        }
        return ResetResult(
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            tools=tool_schemas(description["domain"]),
            metadata={
                "backend": BACKEND_NAME,
                "task_id": self._state["task_id"],
                "candidate_profile": description["candidate_profile"],
                "portfolio_profile": description["portfolio_profile"],
                "fidelity": fidelity,
                "max_submissions": max_submissions,
                "measurement": self._measurement,
                "reward_weights": self._weights.__dict__,
            },
        )

    async def step(self, action: ToolCall) -> StepResult:
        w = self._weights
        if action.name == FINISH_TOOL:
            summary = (action.arguments or {}).get("summary", "")
            self._state["finished_summary"] = str(summary)[:1000]
            return StepResult({"status": "finished", "best": self._best_view()}, 0.0, True, "AGENT_FINISHED", "finish")
        if action.name != SUBMIT_TOOL:
            return StepResult(
                {"error": "UNKNOWN_TOOL", "detail": f"available tools: {SUBMIT_TOOL}, {FINISH_TOOL}"},
                w.schema_invalid, False, "RUNNING", "schema_invalid",
            )
        try:
            submission = parse_submission(action.arguments, self._description["domain"])
        except ToolArgumentError as exc:
            return StepResult(
                {"error": exc.code, "detail": exc.detail, "submission_counted": False},
                w.schema_invalid, False, "RUNNING", "schema_invalid",
            )

        config_key = json.dumps(submission.config, sort_keys=True)
        if config_key in self._state["seen"]:
            observation = {
                "duplicate": True,
                "detail": "This exact configuration was already evaluated; previous result repeated.",
                "previous": self._state["seen"][config_key],
                "submission_counted": False,
            }
            return StepResult(observation, w.repeat_penalty, False, "RUNNING", "duplicate")

        self._state["submissions"] += 1
        result, cached = await self._evaluate(submission.config, self._state["fidelity"])
        stage = result.get("stage", "evaluation_unknown")
        repair: Mapping[str, Any] | None = None
        if stage in UNKNOWN_STAGES and self.on_unknown is not None:
            repair = await self.on_unknown({
                "stage": stage,
                "diagnostic": result.get("diagnostic"),
                "config": submission.config,
                "hypothesis": submission.hypothesis,
                "submission": self._state["submissions"],
            })
            if repair.get("retry"):
                await self.restart_evaluator()
                try:
                    self._description = await self._evaluator.describe()
                except Exception:
                    logger.exception("MICake describe failed after harness repair")
                    repair = {**repair, "retry": False, "describe_error": "describe failed after repair"}
                else:
                    result, cached = await self._evaluate(submission.config, self._state["fidelity"])
                    stage = result.get("stage", "evaluation_unknown")
            settle = repair.get("settle")
            if settle is not None:
                await settle(stage)
        reward, new_best = self._reward(stage, result, submission.config)
        remaining = self._state["max_submissions"] - self._state["submissions"]
        view = self._result_view(result)
        self._state["seen"][config_key] = view
        harness_repair = None
        if repair is not None:
            harness_repair = {"retry": bool(repair.get("retry")), "summary": repair.get("summary"), "stage_after": stage}
        history_entry = {
            "submission": self._state["submissions"], "config": submission.config,
            "hypothesis": submission.hypothesis, "result": view, "reward": reward,
        }
        if harness_repair is not None:
            history_entry["harness_repair"] = harness_repair
        self._state["history"].append(history_entry)
        observation = {
            "submission": self._state["submissions"],
            "remaining_submissions": remaining,
            **view,
            "new_best": new_best,
            "best": self._best_view(),
        }
        if harness_repair is not None:
            observation["harness_repair"] = harness_repair
        if stage in UNKNOWN_STAGES:
            terminal, done = "UNKNOWN_OUTCOME", True
        elif remaining <= 0:
            terminal, done = "BUDGET_EXHAUSTED", True
        else:
            terminal, done = "RUNNING", False
        return StepResult(
            observation, reward, done, terminal, stage,
            receipt_refs=receipts_for(result),
            metadata={"hypothesis": submission.hypothesis, "config": submission.config, "cached": cached, "raw": result},
        )

    def _reward(self, stage: str, result: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, bool]:
        w = self._weights
        if stage in UNKNOWN_STAGES or stage in ("gpu_refused", "request_invalid"):
            return w.unknown_outcome, False
        if stage == "verify_fail":
            return w.verify_fail, False
        if stage == "verified":
            return 0.0, False
        if stage in ("compile_fail", "resource_rejected"):
            return w.compile_fail, False
        if stage == "compiled":
            return w.compile_pass, False
        if stage == "correctness_fail":
            return w.correctness_fail, False
        self._state["correct_count"] += 1
        if stage == "correct":
            return w.correctness_pass, False
        performance = result.get("performance") or {}
        speedup = performance.get("geomean_speedup")
        if performance.get("validity") != "VALID" or not speedup:
            return w.correctness_pass, False
        reward = w.correctness_pass + _clipped_log(speedup, w.performance_log_clip)
        if speedup > self._state["best"]["speedup"] * (1.0 + self._new_best_min_gain):
            self._state["best"] = {"speedup": speedup, "config": copy.deepcopy(config), "source": "agent"}
            return reward + w.new_best_bonus, True
        return reward, False

    def _result_view(self, result: Mapping[str, Any]) -> dict[str, Any]:
        view: dict[str, Any] = {"stage": result.get("stage")}
        for key in ("diagnostic", "compiled_facts", "correctness", "performance"):
            if key in result:
                view[key] = result[key]
        if "correctness" in view:
            view["correctness"] = {
                "status": view["correctness"]["status"],
                "failed_workloads": [c["workload_id"] for c in view["correctness"]["cases"] if c["status"] != "PASS"],
            }
        if "performance" in view:
            perf = view["performance"]
            view["performance"] = {
                "validity": perf.get("validity"),
                "geomean_speedup_vs_baseline": _round(perf.get("geomean_speedup")),
                "per_workload_speedup": {k: _round(v) for k, v in (perf.get("per_workload_speedup") or {}).items()},
            }
        return view

    @property
    def weights(self) -> RewardWeights:
        """Turn reward weights."""
        return self._weights

    @property
    def domain(self) -> Mapping[str, Any]:
        """Closed candidate domain reported by the MICake worker."""
        return self._description["domain"]

    def task_packet(self) -> dict[str, Any]:
        """Initial task observation of the current episode."""
        return copy.deepcopy(self._state["task_packet"])

    def history(self) -> list[dict[str, Any]]:
        """Counted submissions of the current episode, in order."""
        return copy.deepcopy(self._state["history"])

    async def measure(self, config: Mapping[str, Any], fidelity: str = "performance") -> dict[str, Any]:
        """Evaluate ``config`` outside the episode: no budget, no result cache, no state change."""
        return dict(await self._evaluator.evaluate(dict(config), fidelity, self._measurement))

    async def restart_evaluator(self) -> None:
        """Drop the worker so the next evaluation starts from a fresh process."""
        await self._evaluator.close()
        self._result_cache.clear()

    def clear_result_cache(self) -> None:
        """Forget cached evaluations so the next episode pays full evaluation cost."""
        self._result_cache.clear()

    def remaining_submissions(self) -> int:
        """Submission budget left in the current episode."""
        return self._state["max_submissions"] - self._state["submissions"]

    def best(self) -> dict[str, Any]:
        """Best candidate so far (``source`` is ``baseline`` until the agent beats it)."""
        return self._best_view()

    def _best_view(self) -> dict[str, Any]:
        best = self._state["best"]
        return {"geomean_speedup_vs_baseline": _round(best["speedup"]), "source": best["source"], "config": best["config"]}

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        best = self._state.get("best", {"speedup": 1.0, "source": "baseline"})
        reward = 0.0
        if best["source"] == "agent":
            reward = self._weights.terminal_best_speedup * _clipped_log(best["speedup"], self._weights.performance_log_clip)
        return reward, {
            "terminal_state": terminal_state,
            "submissions": self._state.get("submissions", 0),
            "correct_count": self._state.get("correct_count", 0),
            "best_speedup": best["speedup"],
            "best_source": best["source"],
            "best_config": best.get("config"),
            "agent_summary": self._state.get("finished_summary"),
        }

    async def snapshot(self) -> str:
        ref = uuid.uuid4().hex
        self._snapshots[ref] = copy.deepcopy(self._state)
        return ref

    async def restore(self, state_ref: str) -> None:
        self._state = copy.deepcopy(self._snapshots[state_ref])

    async def close(self) -> None:
        await self._evaluator.close()


def _round(value: float | None) -> float | None:
    return round(value, 4) if isinstance(value, float) else value
