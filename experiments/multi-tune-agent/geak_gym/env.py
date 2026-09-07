"""Gymnasium-compatible environment wrapping the GEAK kernel sandbox."""

from __future__ import annotations

import copy
import logging
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from geak_utils import KernelSandbox, TaskSpec

from .reward_shaping import RewardShapingConfig, shape_reward, shape_step_reward
from .spaces import GEAKAction, GEAKObservation, parse_action

logger = logging.getLogger(__name__)


class GEAKGymEnv:
    """Gymnasium-style environment for GEAK kernel optimisation.

    Each episode:
      1. ``reset()`` — pick a task, prepare sandbox, establish baseline
      2. ``step(action)`` — agent edits / evaluates kernel
      3. Episode ends when ``done=True`` (full evaluation or max turns)

    Not a strict ``gymnasium.Env`` subclass to avoid the gymnasium dependency
    at import time, but follows the same ``reset / step / close`` contract.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        upstream_root: str | Path,
        run_root: str | Path | None = None,
        gpu_ids: str = "0",
        command_timeout: int = 300,
        max_turns: int = 20,
        baseline_repeats: int = 3,
        reward_config: RewardShapingConfig | None = None,
    ) -> None:
        self.tasks = list(tasks)
        self.upstream_root = Path(upstream_root)
        self.run_root = Path(run_root) if run_root else self.upstream_root / "gym_runs"
        self.gpu_ids = str(gpu_ids)
        self.command_timeout = int(command_timeout)
        self.max_turns = int(max_turns)
        self.baseline_repeats = int(baseline_repeats)
        self.reward_config = reward_config or RewardShapingConfig()

        self._task_index = 0
        self._sandbox: KernelSandbox | None = None
        self._turn: int = 0
        self._best_speedup: float = 1.0
        self._done: bool = True
        self._episode_dir: Path | None = None
        self._baseline: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[GEAKObservation, dict[str, Any]]:
        """Start a new episode.  Optionally pass ``task_index`` in *options*."""
        self.close()
        opts = dict(options or {})
        idx = int(opts.get("task_index", self._task_index))
        task = self.tasks[idx % len(self.tasks)]
        self._task_index = (idx + 1) % len(self.tasks)

        episode_id = uuid.uuid4().hex[:12]
        self._episode_dir = self.run_root / f"ep_{task.task_id}_{episode_id}"
        self._sandbox = KernelSandbox(
            upstream_root=self.upstream_root,
            run_root=self.run_root,
            gpu_ids=self.gpu_ids,
            command_timeout=self.command_timeout,
        )
        self._sandbox.prepare(task, self._episode_dir)
        self._baseline = self._sandbox.establish_baseline(self.baseline_repeats)
        self._turn = 0
        self._best_speedup = 1.0
        self._done = False

        obs = GEAKObservation(
            tool_result={"ok": True, "event": "reset", "task_id": task.task_id},
            allowed_write_paths=list(self._sandbox.allowed_write_paths),
            baseline_ms=dict(self._sandbox.baseline_ms),
            current_speedup=1.0,
            turn_index=0,
            done=False,
        )
        info = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "baseline": copy.deepcopy(self._baseline),
        }
        return obs, info

    def step(
        self, action: str | dict[str, Any] | GEAKAction
    ) -> tuple[GEAKObservation, float, bool, bool, dict[str, Any]]:
        """Execute one agent action. Returns (obs, reward, terminated, truncated, info)."""
        if self._done or self._sandbox is None:
            raise RuntimeError("Episode is done or not started. Call reset().")

        if not isinstance(action, GEAKAction):
            action = parse_action(action)

        self._turn += 1
        params = action.to_sandbox_params()
        result = self._sandbox.execute_tool(action.tool_name, action.arguments)

        is_full_eval = (
            action.tool_name == "evaluate"
            and action.arguments.get("mode", "full") == "full"
        )

        if is_full_eval and result.get("ok"):
            eval_data = result.get("evaluation") or result
            speedup = float(eval_data.get("speedup_geomean") or 0.0)
            if eval_data.get("correct"):
                self._best_speedup = max(self._best_speedup, speedup)
            reward = shape_reward(
                eval_data, self._best_speedup, self._turn, self.reward_config
            )
        else:
            reward = shape_step_reward(result, self._turn, self.reward_config)

        truncated = self._turn >= self.max_turns
        terminated = is_full_eval and result.get("ok", False)
        self._done = terminated or truncated

        obs = GEAKObservation(
            tool_result=result,
            allowed_write_paths=list(self._sandbox.allowed_write_paths),
            baseline_ms=dict(self._sandbox.baseline_ms),
            current_speedup=self._best_speedup,
            turn_index=self._turn,
            done=self._done,
        )
        info = {"turn": self._turn, "is_full_eval": is_full_eval}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Release sandbox resources."""
        self._sandbox = None
        self._done = True

    @property
    def turn(self) -> int:
        return self._turn


def make_geak_env(
    cases_path: str | Path,
    upstream_root: str | Path,
    **kwargs: Any,
) -> GEAKGymEnv:
    """Convenience factory: load tasks from a YAML file and create the env."""
    from geak_utils import load_tasks

    tasks = load_tasks(Path(cases_path))
    return GEAKGymEnv(tasks=tasks, upstream_root=upstream_root, **kwargs)
