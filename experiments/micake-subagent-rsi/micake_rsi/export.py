"""Per-role specialist training data export (only specialist turns are trainable).

``rl/<role>.jsonl``: every specialist episode with messages, tool schemas,
the trainable assistant-message indices, per-step returns/advantages and the
weight binding that produced it. Each turn keeps the sampled completion token ids and logprobs.
``requires_logprob_recompute`` is false only when every turn is aligned.
Otherwise the trainer recomputes old log-probs under the recorded adapter.

``sft/<role>.jsonl``: the subset whose total reward clears the configured
threshold and that ended cleanly, for rejection-sampling SFT warm-up.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from lumenrl.agentic.core.trajectory import Episode
from lumenrl.agentic.rollout.token_alignment import ALIGNED
from micake_rsi.guards import UNTRUSTED
from micake_rsi.rewards import SpecialistCredit

SAMPLE_SCHEMA = "micake_specialist_sample_v1"
CLEAN_TERMINALS = ("AGENT_FINISHED", "BUDGET_EXHAUSTED")


def _assistant_indices(messages: list[Mapping[str, Any]]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def specialist_sample(generation_id: str, child: Episode, manager: Episode, credit: SpecialistCredit) -> dict[str, Any]:
    """One trainable record for a specialist episode."""
    assistant = _assistant_indices(child.messages)
    steps = []
    for step, ret, adv in zip(child.steps, credit.step_returns, credit.advantages):
        steps.append({
            "step_id": step.step_index,
            "turn_index": step.turn_index,
            "assistant_message_index": assistant[step.turn_index],
            "tool": step.action.name,
            "execution_result": step.result.execution_result,
            "reward_local": step.result.reward,
            "return": ret,
            "advantage": adv,
        })
    return {
        "schema": SAMPLE_SCHEMA,
        "generation_id": generation_id,
        "role": child.role,
        "binding": child.policy.get("binding"),
        "policy": {k: v for k, v in child.policy.items() if k != "binding"},
        "episode_id": child.episode_id,
        "parent_episode_id": child.parent_episode_id,
        "task_id": manager.task.get("task_id"),
        "terminal_state": child.terminal_state,
        "messages": child.messages,
        "tools": child.environment.get("tools"),
        "trainable_message_indices": assistant,
        "steps": steps,
        "reward": credit.to_dict(),
        "manager_verification": ((manager.environment.get("final_summary") or {}).get("verification") or {}).get("status"),
        "rollout_tokens": [
            {
                "turn_index": turn.turn_index,
                "prompt_token_ids": turn.prompt_token_ids,
                "response_token_ids": turn.response_token_ids,
                "response_logprobs": turn.response_logprobs,
                "logprob_status": turn.logprob_status,
                "token_ids_cover": turn.token_ids_cover,
            }
            for turn in child.turns
        ],
        "requires_logprob_recompute": not child.turns or any(turn.logprob_status != ALIGNED for turn in child.turns),
    }


def export_specialist_data(
    out_dir: Path,
    generation_id: str,
    managers: Iterable[Episode],
    children: Mapping[str, list[Episode]],
    credits: Mapping[str, SpecialistCredit],
    sft_min_total_reward: float,
) -> dict[str, dict[str, Any]]:
    """Write RL and SFT JSONL per role.

    Returns:
        ``role -> {"rl": path, "rl_count": n, "sft": path, "sft_count": m}``.
    """
    rl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manager in managers:
        for child in children.get(manager.episode_id, ()):
            if child.role == "debugger":
                continue
            rl[child.role or ""].append(specialist_sample(generation_id, child, manager, credits[child.episode_id]))

    written: dict[str, dict[str, Any]] = {}
    for role, samples in sorted(rl.items()):
        sft = [
            s for s in samples
            if s["terminal_state"] in CLEAN_TERMINALS
            and s["reward"]["total"] >= sft_min_total_reward
            and s["manager_verification"] not in UNTRUSTED
        ]
        paths = {}
        for kind, rows in (("rl", samples), ("sft", sft)):
            path = out_dir / kind / f"{role}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            paths[kind] = str(path)
            paths[f"{kind}_count"] = len(rows)
        written[role] = paths
    return written
