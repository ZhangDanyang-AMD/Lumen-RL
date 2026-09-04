"""Stateful GEAK tool and forkable kernel-optimization environment."""

from __future__ import annotations

import copy
import json
import math
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from geak_utils import KernelSandbox, TaskSpec, load_tasks

from .config import MultiTuneConfig
from .models import RewardBreakdown, SessionState
from .trajectory import TrajectoryWriter


_SUPPORTED_CASE_TYPES = frozenset(
    {
        "gemm",
        "fused_attention",
        "grouped_gemm",
        "scaled_quant_gemm",
        "quant_fp4_mxfp",
    }
)


class GEAKToolEnvironment:
    """Expose GEAK's constrained sandbox as one stateful code-agent tool.

    A session owns exactly one isolated workspace. Forks materialize from a
    parent's current source while retaining the original task baseline.
    """

    def __init__(
        self,
        config: MultiTuneConfig,
        trajectory: Optional[TrajectoryWriter] = None,
    ) -> None:
        self.config = config
        self.trajectory = trajectory
        self._load_tasks()
        self._sessions: dict[str, SessionState] = {}
        self._baseline_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.config.trajectory_root.mkdir(parents=True, exist_ok=True)

    def _load_tasks(self) -> None:
        tasks = load_tasks(self.config.cases_path)
        unsupported = sorted(
            {
                task.case_type
                for task in tasks
                if task.case_type not in _SUPPORTED_CASE_TYPES
            }
        )
        if unsupported:
            raise ValueError(
                "unsupported MultiTune case type(s): %s; supported types: %s"
                % (
                    ", ".join(unsupported),
                    ", ".join(sorted(_SUPPORTED_CASE_TYPES)),
                )
            )
        self.cases = {task.case_id: task for task in tasks}

    def create(
        self,
        case_id: str,
        *,
        role: str,
        parent_session_id: Optional[str] = None,
        establish_baseline: bool = False,
        source_path: Optional[Path] = None,
        baseline_override: Optional[Mapping[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        case = self._case(case_id)
        parent = self.get(parent_session_id) if parent_session_id else None
        session_id = "%s-%s" % (role.replace("_", "-"), uuid.uuid4().hex[:12])
        episode_dir = self.config.trajectory_root / "sessions" / session_id
        source = (
            Path(source_path).expanduser().resolve()
            if source_path is not None
            else (parent.workspace if parent else case.kernel_path)
        )
        fork_case = TaskSpec(
            case_id=case.case_id,
            case_type=case.case_type,
            kernel_path=source,
            direction=case.direction,
            max_turns=case.max_turns,
        )
        sandbox = KernelSandbox(
            repository_root=self.config.geak_root,
            run_root=self.config.trajectory_root,
            gpu_ids=self.config.gpu_ids,
            command_timeout=self.config.command_timeout,
        )
        started = time.monotonic()
        workspace = sandbox.prepare(fork_case, episode_dir)

        with self._lock:
            cached = self._baseline_cache.get(case_id)
        if baseline_override is not None:
            baseline = copy.deepcopy(dict(baseline_override))
            per_case = baseline.get("per_case_ms")
            if not isinstance(per_case, Mapping) or not per_case:
                raise ValueError("baseline_override requires non-empty per_case_ms")
            sandbox.baseline_ms = {
                str(name): float(value) for name, value in per_case.items()
            }
            with self._lock:
                self._baseline_cache[case_id] = baseline
        elif establish_baseline or cached is None:
            baseline = sandbox.establish_baseline(self.config.baseline_repeats)
            with self._lock:
                self._baseline_cache.setdefault(case_id, baseline)
                baseline = self._baseline_cache[case_id]
        else:
            baseline = cached
            sandbox.baseline_ms = dict(baseline["per_case_ms"])

        state = SessionState(
            session_id=session_id,
            case_id=case_id,
            role=role,
            episode_dir=episode_dir,
            workspace=workspace,
            sandbox=sandbox,
            parent_session_id=parent_session_id,
            best_speedup=parent.best_speedup if parent else 1.0,
        )
        with self._lock:
            self._sessions[session_id] = state
        observation = self.observe(session_id)
        observation["create_seconds"] = time.monotonic() - started
        self._event(
            "environment_create",
            observation,
            role=role,
            phase="create",
        )
        return session_id, observation

    def fork(self, session_id: str, *, role: str) -> tuple[str, dict[str, Any]]:
        parent = self.get(session_id)
        return self.create(
            parent.case_id,
            role=role,
            parent_session_id=session_id,
        )

    def get(self, session_id: Optional[str]) -> SessionState:
        if not session_id:
            raise KeyError("session_id is required")
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise KeyError("unknown GEAK session: %s" % session_id)
        if state.closed:
            raise RuntimeError("GEAK session is closed: %s" % session_id)
        return state

    def observe(self, session_id: str) -> dict[str, Any]:
        state = self.get(session_id)
        baseline = self._baseline_cache.get(state.case_id, {})
        return {
            "session_id": state.session_id,
            "case_id": state.case_id,
            "role": state.role,
            "workspace": str(state.workspace),
            "parent_session_id": state.parent_session_id,
            "allowed_write_paths": list(state.sandbox.allowed_write_paths),
            "baseline": baseline,
            "best_speedup": state.best_speedup,
            "last_evaluation": state.last_evaluation,
        }

    def execute(
        self,
        session_id: str,
        parameters: Mapping[str, Any],
    ) -> tuple[dict[str, Any], RewardBreakdown, dict[str, Any]]:
        state = self.get(session_id)
        action = str(parameters.get("action") or "").strip()
        started = time.monotonic()
        reward = RewardBreakdown(0.0, 0.0, 0.0, 0.0, state.best_speedup)
        try:
            if action == "list_files":
                result = {
                    "ok": True,
                    "files": state.sandbox.list_files(
                        str(parameters.get("path") or ".")
                    ),
                }
            elif action == "read_file":
                result = {
                    "ok": True,
                    "content": state.sandbox.read_file(
                        str(parameters["path"]),
                        int(parameters.get("offset") or 1),
                        int(parameters.get("limit") or 1000),
                    ),
                }
            elif action == "read_candidate":
                source = self.get(str(parameters["source_session_id"]))
                result = {
                    "ok": True,
                    "source_session_id": source.session_id,
                    "content": source.sandbox.read_file(
                        str(parameters["path"]),
                        int(parameters.get("offset") or 1),
                        int(parameters.get("limit") or 1000),
                    ),
                }
            elif action == "write_file":
                state.sandbox.write_file(
                    str(parameters["path"]), str(parameters["content"])
                )
                result = {"ok": True, "path": str(parameters["path"])}
            elif action == "evaluate":
                mode = str(parameters.get("mode") or "full")
                if mode == "full":
                    evaluation = state.sandbox.evaluate().to_dict()
                    reward = self.reward(evaluation, state.best_speedup)
                    state.last_evaluation = evaluation
                    state.last_reward = reward.total
                    if evaluation["correct"]:
                        state.best_speedup = max(
                            state.best_speedup,
                            float(evaluation["speedup_geomean"]),
                        )
                    result = {
                        "ok": bool(evaluation["compiled"] and evaluation["correct"]),
                        "evaluation": evaluation,
                        "reward": reward.to_dict(),
                    }
                else:
                    command = state.sandbox.run_mode(mode).to_dict()
                    result = {"ok": bool(command["ok"]), "command": command}
            elif action == "state":
                result = {"ok": True, "state": self.observe(session_id)}
            else:
                raise ValueError("unsupported GEAK action: %s" % action)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result = {"ok": False, "error": str(exc)}

        metrics = {
            "action": action,
            "elapsed_seconds": time.monotonic() - started,
            "session_id": session_id,
        }
        self._event(
            "tool_result",
            {
                "parameters": dict(parameters),
                "result": result,
                "reward": reward.to_dict(),
                **metrics,
            },
            role=state.role,
            phase="tool",
        )
        return result, reward, metrics

    def verify(
        self, session_id: str
    ) -> tuple[dict[str, Any], RewardBreakdown, dict[str, Any]]:
        return self.execute(session_id, {"action": "evaluate", "mode": "full"})

    def release(self, session_id: str) -> None:
        state = self.get(session_id)
        state.closed = True
        if not self.config.keep_sessions:
            shutil.rmtree(state.episode_dir, ignore_errors=True)
        self._event(
            "environment_release",
            {"session_id": session_id, "workspace": str(state.workspace)},
            role=state.role,
            phase="release",
        )

    @staticmethod
    def reward(evaluation: Mapping[str, Any], previous_best: float) -> RewardBreakdown:
        correct = bool(evaluation.get("compiled") and evaluation.get("correct"))
        speedup = float(evaluation.get("speedup_geomean") or 0.0)
        if not correct:
            return RewardBreakdown(-1.0, -1.0, 0.0, 0.0, speedup)
        performance = max(-1.0, min(math.log(3.0), math.log(max(speedup, 1e-9))))
        improvement = max(0.0, speedup - previous_best)
        total = 1.0 + performance + 0.5 * improvement
        return RewardBreakdown(total, 1.0, performance, improvement, speedup)

    def tool_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "geak",
                    "description": (
                        "Inspect, edit, and evaluate one isolated GEAK kernel workspace. "
                        "Only task-declared kernel source paths are writable."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "list_files",
                                    "read_file",
                                    "read_candidate",
                                    "write_file",
                                    "evaluate",
                                    "state",
                                ],
                            },
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "source_session_id": {"type": "string"},
                            "offset": {"type": "integer", "minimum": 1},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 4000,
                            },
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "compile",
                                    "correctness",
                                    "performance",
                                    "full",
                                ],
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def case_observation(self, case_id: str) -> dict[str, Any]:
        case = self._case(case_id)
        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "kernel_path": str(case.kernel_path),
            "direction": case.direction,
        }

    def _case(self, case_id: str) -> Any:
        try:
            return self.cases[case_id]
        except KeyError as exc:
            raise KeyError(
                "unknown case %r; choices: %s"
                % (case_id, ", ".join(sorted(self.cases)))
            ) from exc

    def _event(
        self,
        event: str,
        payload: Mapping[str, Any],
        *,
        role: str,
        phase: str,
    ) -> None:
        if self.trajectory:
            self.trajectory.append(event, payload, role=role, phase=phase)

    @staticmethod
    def response_text(result: Mapping[str, Any], limit: int = 12000) -> str:
        text = json.dumps(result, sort_keys=True, default=str)
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return text[:half] + "...(truncated)..." + text[-half:]


class GEAKStatefulTool:
    """Runtime tool lifecycle backed by :class:`GEAKToolEnvironment`."""

    name = "geak"

    def __init__(self, environment: GEAKToolEnvironment) -> None:
        self.environment = environment

    def schemas(self) -> list[dict[str, Any]]:
        return self.environment.tool_schema()

    def create(
        self, create_kwargs: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any]]:
        return self.environment.create(
            str(create_kwargs["case_id"]),
            role=str(create_kwargs.get("role") or "engineer"),
            parent_session_id=(
                str(create_kwargs["parent_session_id"])
                if create_kwargs.get("parent_session_id")
                else None
            ),
            establish_baseline=bool(create_kwargs.get("establish_baseline", False)),
        )

    def execute(
        self, instance_id: str, parameters: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, Mapping[str, Any]]:
        result, reward, metrics = self.environment.execute(instance_id, parameters)
        return result, reward.total, metrics

    def calc_reward(self, instance_id: str) -> float:
        return float(self.environment.get(instance_id).last_reward)

    def release(self, instance_id: str) -> None:
        self.environment.release(instance_id)

