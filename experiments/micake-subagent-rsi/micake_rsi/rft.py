"""Step-level rejection sampling: which specialist turns become SFT targets.

A Planner submission turn is kept when MICake measured it VALID at or above
``min_speedup``; its report turn is kept when the episode kept a submission.
A Critic diagnosis is kept when its probed recommendation measured VALID at or
above ``min_context_ratio`` of the context best. Every other turn (resource
rejection, compile/correctness failure, duplicate, schema-invalid) gets a
negative advantage, so it is context only and never a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from micake_rsi.ab_eval import PRODUCTIVE, step_class
from micake_rsi.corrections import annotate_lds as annotate


@dataclass(frozen=True)
class RftFilter:
    min_speedup: float = 0.95
    min_context_ratio: float = 0.97


def _speedup(step: Mapping[str, Any]) -> float:
    observation = step["result"].get("observation") or {}
    performance = observation.get("performance") or (observation.get("probe") or {}).get("performance") or {}
    return float(performance.get("geomean_speedup_vs_baseline") or 0.0)


def _assistant_indices(messages: list[Mapping[str, Any]]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def step_advantages(record: Mapping[str, Any], role: str, rft: RftFilter) -> list[float]:
    """+1 for kept turns, -1 for the rest, aligned with ``episode.steps``.

    Episodes with a parallel tool call are dropped entirely: the specialist
    contract is one tool per turn, and the Qwen template merges consecutive
    tool messages, so their prefixes do not render stably.
    """
    steps = record["episode"]["steps"]
    if any(len(m.get("tool_calls") or ()) > 1 for m in record["episode"]["messages"]):
        return [-1.0] * len(steps)
    if role == "planner":
        kept = [
            step_class(s) == PRODUCTIVE and _speedup(s) >= rft.min_speedup
            if s["action"]["name"] != "report" else None
            for s in steps
        ]
        any_kept = any(k for k in kept if k is not None)
        return [(1.0 if (any_kept if k is None else k) else -1.0) for k in kept]
    threshold = rft.min_context_ratio * float(record["context_best"])
    return [
        1.0 if s["result"].get("execution_result") == "diagnosis_probed" and step_class(s) == PRODUCTIVE
        and _speedup(s) >= threshold else -1.0
        for s in steps
    ]


def rft_sample(record: Mapping[str, Any], role: str, rft: RftFilter, annotate_lds: bool = False) -> dict[str, Any]:
    """Trainable sample in the ``write_role_sft`` shape.

    With ``annotate_lds`` every kept Planner submission starts with the exact
    LDS arithmetic of the config it submits.
    """
    episode = record["episode"]
    assistant = _assistant_indices(episode["messages"])
    advantages = step_advantages(record, role, rft)
    messages = list(episode["messages"])
    if annotate_lds and role == "planner":
        for step, adv in zip(episode["steps"], advantages):
            arguments = step["action"].get("arguments") or {}
            if adv > 0 and step["action"]["name"] == "submit_schedule" and isinstance(arguments.get("schedule"), Mapping):
                index = assistant[step["turn_index"]]
                messages[index] = annotate(messages[index], arguments)
    return {
        "role": role,
        "episode_id": episode["episode_id"],
        "source_key": record["key"],
        "messages": messages,
        "tools": (episode.get("environment") or {}).get("tools"),
        "steps": [
            {"step_id": s["step_index"], "turn_index": s["turn_index"],
             "assistant_message_index": assistant[s["turn_index"]], "tool": s["action"]["name"],
             "execution_result": s["result"].get("execution_result"), "advantage": adv}
            for s, adv in zip(episode["steps"], advantages)
        ],
    }
