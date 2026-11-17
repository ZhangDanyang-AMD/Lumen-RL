"""Run an agent on SWE-bench tasks and grade every attempt.

For a step-0 evaluation of a policy, and for picking RL tasks: a task the policy sometimes
solves and sometimes does not (`summarize(...)["mixed"]`) is one a group of rollouts learns from.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from lumenrl.agent.loop import GenerateFn, ToolAgentLoop
from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.bash import MiniSweBashTool
from lumenrl.agent.swe.swebench import SweBenchEnvironment, SweBenchTask

logger = logging.getLogger(__name__)


def mini_swe_agent(chat: Any, parser: Any, *, max_turns: int = mini_swe.STEP_LIMIT,
                   max_parallel_calls: int = 8) -> ToolAgentLoop:
    """mini-swe-agent's loop: the bash tool alone, its step limit, and no truncation on top of
    the observation's own."""
    return ToolAgentLoop(chat, [MiniSweBashTool()], parser=parser, max_assistant_turns=max_turns,
                         max_parallel_calls=max_parallel_calls, max_tool_response_length=0)


async def attempt(task: SweBenchTask, agent: ToolAgentLoop, env: SweBenchEnvironment, generate: GenerateFn,
                  sampling_params: dict[str, Any], *, max_response_tokens: int,
                  max_sequence_tokens: int | None = None) -> dict[str, Any]:
    """One rollout of `task` and its grade. A sandbox or grading failure is recorded, not raised."""
    t0 = time.monotonic()
    result: dict[str, Any] = {"instance_id": task.instance_id, "resolved": False}
    try:
        async with env.agent_session(task) as bash:
            trajectory = await agent.run(generate, task.messages(), sampling_params, max_response_tokens,
                                         max_sequence_tokens, sessions={"bash": bash})
            submission = bash.submission
        result.update(
            stop_reason=trajectory.stop_reason, turns=trajectory.assistant_turns, tool_calls=trajectory.tool_calls,
            tool_errors=trajectory.tool_errors, malformed_tool_calls=trajectory.malformed_tool_calls,
            prompt_tokens=len(trajectory.prompt_ids), response_tokens=len(trajectory.response_ids),
            model_tokens=sum(trajectory.response_mask), generate_s=round(trajectory.generate_s, 2),
            tool_s=round(trajectory.tool_s, 2), submission=submission,
        )
        t_grade = time.monotonic()
        grade = await env.grade(task, submission)
        result.update(resolved=grade.resolved, applied=grade.applied, reason=grade.reason,
                      grade_s=round(time.monotonic() - t_grade, 2))
    except Exception as e:  # noqa: BLE001 - one broken sandbox must not end an evaluation
        logger.warning("%s failed: %r", task.instance_id, e)
        result.update(error=f"{type(e).__name__}: {e}")
    result["elapsed_s"] = round(time.monotonic() - t0, 2)
    return result


async def evaluate(tasks: Sequence[SweBenchTask], agent: ToolAgentLoop, env: SweBenchEnvironment,
                   generate_for: Callable[[SweBenchTask], GenerateFn], sampling_params: dict[str, Any], *,
                   n: int = 1, concurrency: int = 8, max_response_tokens: int,
                   max_sequence_tokens: int | None = None,
                   on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """`n` attempts at every task, `concurrency` at a time (each holds a sandbox while it runs
    and opens a second to be graded). `generate_for(task)` is the policy for that task's
    attempts; an engine ignores the task. `on_result` sees each result as it lands."""
    slots = asyncio.Semaphore(max(1, int(concurrency)))

    async def one(task: SweBenchTask, k: int) -> dict[str, Any]:
        async with slots:
            r = await attempt(task, agent, env, generate_for(task), sampling_params,
                              max_response_tokens=max_response_tokens, max_sequence_tokens=max_sequence_tokens)
        r["sample"] = k
        if on_result is not None:
            on_result(r)
        return r

    return list(await asyncio.gather(*(one(task, k) for task in tasks for k in range(n))))


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_task[r["instance_id"]].append(bool(r["resolved"]))
    mixed = sorted(t for t, v in by_task.items() if any(v) and not all(v))
    return {
        "attempts": len(results),
        "tasks": len(by_task),
        "resolved_rate": sum(r["resolved"] for r in results) / max(1, len(results)),
        "tasks_ever_resolved": sum(any(v) for v in by_task.values()),
        "mixed": mixed,
        "errors": sum("error" in r for r in results),
        "stop_reasons": dict(Counter(r.get("stop_reason", "error") for r in results)),
        "grade_reasons": dict(Counter(r.get("reason", "error") for r in results)),
    }
