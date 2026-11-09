"""Improvement-driven Planner targets from end-to-end Manager episodes.

The RSI objective is a faster path to a faster kernel, so a Planner turn is a
positive target only when MICake marked it ``new_best`` (it beat the session
best by the noise band). Every other submission turn can be replaced by a
teacher correction: the fastest MICake-measured untested one-axis move from
the session best at that moment, when it beats that best. Both come from real episodes of
any arm (the current champion, challengers, DSV4.1), so experience compounds
across generations.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterator, Mapping, Sequence

from micake_rsi.corrections import annotate_lds, describe_changes, describe_config
from micake_rsi.search_harness import fits_lds, lds_check, neighbours, submission_arguments
from sandbox.micake.tool_runtime import HIDDEN_FIXED_FIELDS, SUBMIT_TOOL

NEW_BEST_GAIN = 0.005

EXPLOIT_CONTENT = (
    "The session best is {best:.4f} ({best_config}). The fastest way to beat it is an untested one-axis move from "
    "that config: {changes}.",
    "Building on the current best ({best_config}, {best:.4f}x) instead of a new family: {changes}. "
    "This candidate has not been evaluated yet.",
    "History says the best region is {best_config} at {best:.4f}. Next I refine it with {changes}.",
)
EXPLOIT_HYPOTHESIS = "One-axis refinement of the session best ({changes}); expected to beat {best:.4f}."


def config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


def full_config(arguments: Any) -> dict[str, Any] | None:
    """Submitted config with hidden fixed fields restored, or None when malformed."""
    if not isinstance(arguments, Mapping):
        return None
    if not isinstance(arguments.get("schedule"), Mapping) or not isinstance(arguments.get("realization"), Mapping):
        return None
    config = {"schedule": dict(arguments["schedule"]), "realization": dict(arguments["realization"])}
    for section, names in HIDDEN_FIXED_FIELDS.items():
        for name in names:
            config[section].setdefault(name, 1)
    return config


def _packet(child: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(child["messages"][1]["content"])
    except (ValueError, TypeError, KeyError, IndexError):
        return {}


def _has_parallel_calls(child: Mapping[str, Any]) -> bool:
    return any(len(m.get("tool_calls") or ()) > 1 for m in child["messages"])


def planner_children(record: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for child in record.get("children") or ():
        if child.get("role") == "planner" and not _has_parallel_calls(child):
            yield child


def walk_steps(child: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Per submit step: the session best before it, every config already submitted, and its outcome."""
    packet = _packet(child)
    best = packet.get("best") or {}
    best_speedup = float(best.get("geomean_speedup_vs_baseline") or 1.0)
    best_config = best.get("config") or packet.get("baseline_config")
    submitted = {config_key(h["config"]) for h in packet.get("history") or () if h.get("config")}
    if packet.get("baseline_config"):
        submitted.add(config_key(packet["baseline_config"]))
    for step in child["steps"]:
        if step["action"]["name"] != SUBMIT_TOOL:
            continue
        observation = step["result"].get("observation") or {}
        config = (step["result"].get("metadata") or {}).get("config") or full_config(step["action"].get("arguments"))
        yield {"step": step, "best_speedup": best_speedup, "best_config": best_config, "submitted": set(submitted),
               "config": config, "new_best": bool(observation.get("new_best")),
               "stage": step["result"].get("execution_result")}
        if config:
            submitted.add(config_key(config))
        after = observation.get("best") or {}
        if after.get("geomean_speedup_vs_baseline") is not None:
            best_speedup = float(after["geomean_speedup_vs_baseline"])
            best_config = after.get("config") or best_config


def _assistant_indices(child: Mapping[str, Any]) -> list[int]:
    return [i for i, m in enumerate(child["messages"]) if m.get("role") == "assistant"]


def _tools(child: Mapping[str, Any]) -> Any:
    return (child.get("environment") or {}).get("tools")


def improvement_sample(child: Mapping[str, Any], source_key: str) -> dict[str, Any] | None:
    """The child episode with its ``new_best`` submissions (LDS-annotated) and, if any, its report as targets."""
    kept = {entry["step"]["step_index"] for entry in walk_steps(child) if entry["new_best"] and entry["stage"] == "measured"}
    if not kept:
        return None
    assistant = _assistant_indices(child)
    messages = list(child["messages"])
    steps = []
    for step in child["steps"]:
        is_report = step["action"]["name"] == "report"
        advantage = 1.0 if (step["step_index"] in kept or is_report) else -1.0
        index = assistant[step["turn_index"]]
        if step["step_index"] in kept:
            messages[index] = annotate_lds(messages[index], step["action"]["arguments"])
        steps.append({"step_id": step["step_index"], "turn_index": step["turn_index"], "assistant_message_index": index,
                      "tool": step["action"]["name"], "execution_result": step["result"].get("execution_result"),
                      "advantage": advantage})
    return {"role": "planner", "episode_id": child["episode_id"], "source_key": source_key,
            "messages": messages, "tools": _tools(child), "steps": steps}


def exploit_candidates(best_config: Mapping[str, Any], domain: Mapping[str, Any], submitted: set[str]) -> Iterator[dict[str, Any]]:
    """Untested, LDS-fitting one-axis moves from the session best."""
    seen = set(submitted)
    for candidate in neighbours(best_config, domain):
        key = config_key(candidate)
        if key in seen or not fits_lds(candidate):
            continue
        seen.add(key)
        yield candidate


def exploit_message(original: Mapping[str, Any], best_config: Mapping[str, Any], best_speedup: float,
                    candidate: Mapping[str, Any], variant: int) -> dict[str, Any]:
    fields = {"best": best_speedup, "best_config": describe_config(best_config), "changes": describe_changes(best_config, candidate)}
    content = EXPLOIT_CONTENT[variant % len(EXPLOIT_CONTENT)].format(**fields)
    message = copy.deepcopy(dict(original))
    message["content"] = f"{content}\nLDS check: {lds_check(candidate)}."
    call = copy.deepcopy(message["tool_calls"][0])
    call["function"]["arguments"] = json.dumps(submission_arguments(candidate, EXPLOIT_HYPOTHESIS.format(**fields)))
    message["tool_calls"] = [call]
    return message


def exploit_sample(child: Mapping[str, Any], entry: Mapping[str, Any], candidate: Mapping[str, Any], source_key: str) -> dict[str, Any]:
    """SFT sample whose only target replaces ``entry``'s turn with the measured improving ``candidate``."""
    step = entry["step"]
    target = _assistant_indices(child)[step["turn_index"]]
    variant = sum(map(ord, source_key))
    corrected = exploit_message(child["messages"][target], entry["best_config"], entry["best_speedup"], candidate, variant)
    return {"role": "planner", "episode_id": child["episode_id"], "source_key": source_key,
            "messages": list(child["messages"][:target]) + [corrected], "tools": _tools(child),
            "steps": [{"step_id": step["step_index"], "turn_index": step["turn_index"], "assistant_message_index": target,
                       "tool": SUBMIT_TOOL, "execution_result": "exploit_correction", "advantage": 1.0}]}


def improves(measured: Mapping[str, Any] | None, best_speedup: float, gain: float = NEW_BEST_GAIN) -> bool:
    return bool(measured) and measured.get("stage") == "measured" and measured.get("validity") == "VALID" \
        and float(measured.get("speedup") or 0.0) >= best_speedup + gain


def correctable(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Submit steps that did not set a new best and had a usable session best, wasted turns first."""
    kept = [e for e in entries if not e["new_best"] and e["best_config"]]
    return sorted(kept, key=lambda e: e["stage"] == "measured")
