"""Abstract sandbox interfaces for GPU kernel evaluation."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class CommandResult:
    """Result of running a single evaluation stage (compile/correctness/performance)."""

    ok: bool
    mode: str = ""
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    per_case_ms: dict[str, float] = field(default_factory=dict)
    timed_out: bool = False

    def to_dict(self, output_limit: int = 12000) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "stdout": self.stdout[-output_limit:],
            "stderr": self.stderr[-output_limit:],
            "elapsed_s": self.elapsed_s,
            "per_case_ms": self.per_case_ms,
            "timed_out": self.timed_out,
        }


@dataclass
class EvalResult:
    """Composite result of compile -> correctness -> performance pipeline."""

    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    perf_ms: float = 0.0
    reward: float = -1.0
    stage: str = ""
    error: str = ""
    compile_result: CommandResult | None = None
    correctness_result: CommandResult | None = None
    performance_result: CommandResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "correct": self.correct,
            "speedup": self.speedup,
            "perf_ms": self.perf_ms,
            "reward": self.reward,
            "stage": self.stage,
            "error": self.error,
        }


@dataclass
class RewardBreakdown:
    """Structured reward decomposition for agent training."""

    total: float
    correctness: float
    performance: float
    improvement: float
    speedup: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "correctness": self.correctness,
            "performance": self.performance,
            "improvement": self.improvement,
            "speedup": self.speedup,
        }


class SandboxBackend(abc.ABC):
    """Abstract backend for evaluating GPU kernels in isolated workspaces.

    Implementations handle workspace setup, subprocess execution through GPU
    isolation, and cleanup. The three-stage evaluation pipeline
    (compile -> correctness -> performance) is the universal interface;
    backends may implement additional stages.
    """

    @abc.abstractmethod
    def prepare_workspace(
        self,
        task_id: str,
        kernel_source: str,
        task_dir: Path,
        gpu_id: int,
    ) -> Path:
        """Create an isolated workspace with the candidate kernel.

        Args:
            task_id: Unique identifier for this evaluation.
            kernel_source: The kernel source code to evaluate.
            task_dir: Directory containing the task harness (task_runner.py, etc).
            gpu_id: GPU device ID for evaluation.

        Returns:
            Path to the prepared workspace directory.
        """

    @abc.abstractmethod
    def run_stage(
        self,
        workspace: Path,
        stage: str,
        gpu_id: int,
        timeout: int = 120,
    ) -> CommandResult:
        """Run one evaluation stage in the workspace.

        Args:
            workspace: Path returned by prepare_workspace.
            stage: One of "compile", "correctness", "performance".
            gpu_id: GPU device ID.
            timeout: Max seconds before killing the subprocess.

        Returns:
            CommandResult with ok, stdout, stderr, timing.
        """

    @abc.abstractmethod
    def cleanup(self, workspace: Path) -> None:
        """Remove workspace and any temporary state."""

    def evaluate(
        self,
        task_id: str,
        kernel_source: str,
        task_dir: Path,
        gpu_id: int,
        baseline_ms: float = 1.0,
        timeout: int = 120,
    ) -> EvalResult:
        """Full compile -> correctness -> performance pipeline.

        Default implementation calls prepare_workspace + run_stage for each
        stage. Subclasses may override for optimized evaluation.
        """
        workspace = self.prepare_workspace(task_id, kernel_source, task_dir, gpu_id)
        try:
            compile_r = self.run_stage(workspace, "compile", gpu_id, timeout)
            if not compile_r.ok:
                return EvalResult(
                    stage="compile", error=compile_r.stderr[-200:],
                    compile_result=compile_r,
                )

            correct_r = self.run_stage(workspace, "correctness", gpu_id, timeout)
            if not correct_r.ok:
                return EvalResult(
                    compiled=True, stage="correctness",
                    error=correct_r.stderr[-200:],
                    compile_result=compile_r, correctness_result=correct_r,
                )

            perf_r = self.run_stage(workspace, "performance", gpu_id, timeout)
            perf_ms = 0.0
            if perf_r.per_case_ms:
                perf_ms = next(iter(perf_r.per_case_ms.values()))
            speedup = baseline_ms / perf_ms if perf_ms > 0 and baseline_ms > 0 else 0.0

            return EvalResult(
                compiled=True, correct=True, speedup=speedup, perf_ms=perf_ms,
                stage="success",
                compile_result=compile_r, correctness_result=correct_r,
                performance_result=perf_r,
            )
        finally:
            self.cleanup(workspace)


class StatefulTool(abc.ABC):
    """Abstract tool with session lifecycle for multi-turn agent interaction.

    Each tool manages sessions: create -> execute (N times) -> release.
    Compatible with ToolAgentLoop for driving model-tool interaction.
    """

    name: str

    @abc.abstractmethod
    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for this tool."""

    @abc.abstractmethod
    def create(
        self, create_kwargs: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any]]:
        """Create a new tool session. Returns (session_id, initial_observation)."""

    @abc.abstractmethod
    def execute(
        self, instance_id: str, parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], float, Mapping[str, Any]]:
        """Execute a tool call. Returns (result, reward, metrics)."""

    @abc.abstractmethod
    def calc_reward(self, instance_id: str) -> float:
        """Compute the final reward for a session."""

    @abc.abstractmethod
    def release(self, instance_id: str) -> None:
        """Release session resources."""


class Environment(abc.ABC):
    """Abstract environment for agent evaluation loops."""

    @abc.abstractmethod
    def case_observation(self, case_id: str) -> dict[str, Any]:
        """Get initial observation for a task case."""

    @abc.abstractmethod
    def verify(
        self, session_id: str,
    ) -> tuple[dict[str, Any], RewardBreakdown, dict[str, Any]]:
        """Verify current state and compute reward."""
