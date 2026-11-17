"""SWE-bench tasks as an RL rollout environment: a rollout per task sandbox, rewarded by its tests."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from lumenrl.agent.loop import GenerateFn, ToolAgentLoop
from lumenrl.agent.swe.swebench import SweBenchEnvironment, SweBenchTask
from lumenrl.agent.trajectory import Trajectory

logger = logging.getLogger(__name__)


class SweRolloutEnvironment:
    """Runs one rollout of the agent in a fresh, prepared sandbox of the task in `ground_truth`
    (its v5 SWE-bench row as JSON) and sets the trajectory's `env_reward` from grading the
    submitted patch. A sandbox that cannot be opened fails the rollout; a grading failure is
    logged and scored unresolved, since the rollout itself is still a valid sample."""

    def __init__(self, env: SweBenchEnvironment, *, resolved_reward: float = 1.0,
                 unresolved_reward: float = 0.0) -> None:
        self._env = env
        self._resolved, self._unresolved = float(resolved_reward), float(unresolved_reward)

    async def rollout(self, agent: ToolAgentLoop, generate: GenerateFn, messages: list[dict[str, Any]],
                      sampling_params: dict[str, Any], max_response_tokens: int,
                      max_sequence_tokens: int | None, ground_truth: str) -> Trajectory:
        task = SweBenchTask(json.loads(ground_truth))
        t0 = time.monotonic()
        async with self._env.agent_session(task) as bash:
            trajectory = await agent.run(generate, messages, sampling_params, max_response_tokens,
                                         max_sequence_tokens, sessions={"bash": bash})
            submission = bash.submission
        try:
            grade = await self._env.grade(task, submission)
            resolved = grade.resolved
        except Exception as e:  # noqa: BLE001 - the rollout stands; only its grade is lost
            logger.warning("grading %s failed: %r", task.instance_id, e)
            resolved = False
        trajectory.env_reward = self._resolved if resolved else self._unresolved
        trajectory.env_s = time.monotonic() - t0 - trajectory.generate_s - trajectory.tool_s
        return trajectory
