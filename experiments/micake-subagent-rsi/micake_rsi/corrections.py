"""MICake-verified corrections of resource-rejected specialist turns.

For a submission rejected for exceeding the LDS limit, candidate repairs are
the configs closest to the request (fewest changed fields first) whose exact
shared memory (``search_harness.shared_bytes``) fits: lower pipeline_depth or
unroll_factor, smaller tile_k, a smaller tile dimension. The first repair
MICake measures VALID at or above ``min_speedup`` replaces the rejected
assistant turn as an SFT target, with the LDS arithmetic written out; the
context before that turn is kept unchanged.
"""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any, Iterator, Mapping, Sequence

from micake_rsi.search_harness import fits_lds, lds_check, neighbours, submission_arguments

LDS_LIMIT_BYTES = 163840


def _down(values: Sequence[int], current: int, floor: int | None = None) -> list[int]:
    """``current`` followed by smaller domain values (descending), not below ``floor``."""
    smaller = sorted((v for v in values if v < current and (floor is None or v >= floor)), reverse=True)
    return [current] + smaller


def repairs(config: Mapping[str, Any], domain: Mapping[str, Mapping[str, Sequence[Any]]]) -> Iterator[dict[str, Any]]:
    """LDS-fitting variants of ``config``, fewest changed fields first, then the largest remaining tile."""
    s = config["schedule"]
    axes = ("pipeline_depth", "unroll_factor", "tile_k", "tile_m", "tile_n")
    floors = {"pipeline_depth": 2, "unroll_factor": 2, "tile_k": 16, "tile_m": 32, "tile_n": 32}
    choices = [_down(domain["schedule"][axis], s[axis], floor=floors[axis]) for axis in axes]
    options = []
    for values in itertools.product(*choices):
        changed = sum(v != s[a] for a, v in zip(axes, values))
        if not changed:
            continue
        candidate = copy.deepcopy(dict(config))
        candidate["schedule"].update(dict(zip(axes, values)))
        if not fits_lds(candidate):
            continue
        tile = candidate["schedule"]["tile_m"] * candidate["schedule"]["tile_n"]
        options.append((changed, -tile, -candidate["schedule"]["tile_k"], -candidate["schedule"]["pipeline_depth"], candidate))
    for *_, candidate in sorted(options, key=lambda o: o[:4]):
        yield candidate


def duplicate_repairs(config: Mapping[str, Any], domain: Mapping[str, Mapping[str, Sequence[Any]]]) -> Iterator[dict[str, Any]]:
    """Untested one-axis neighbours of an already-evaluated ``config`` that fit the LDS estimate."""
    for candidate in neighbours(config, domain):
        if fits_lds(candidate):
            yield candidate


def describe_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    parts = []
    for section in ("schedule", "realization"):
        for name, value in after[section].items():
            if before[section].get(name) != value:
                parts.append(f"{name} {before[section][name]} -> {value}")
    return ", ".join(parts)


def describe_config(config: Mapping[str, Any]) -> str:
    s, r = config["schedule"], config["realization"]
    return (f"tile {s['tile_m']}x{s['tile_n']}x{s['tile_k']}, num_stages={r['num_stages']}, "
            f"pipeline_depth={s['pipeline_depth']}, unroll={s['unroll_factor']}, vectorize={s['vectorize_factor']}")


CONTENT_TEMPLATES = (
    "{request} would not fit in the gfx950 shared-memory limit (resource_limits: {limit} bytes), so it would be "
    "rejected without a measurement. Submitting the closest variant that fits instead: {changes}.",
    "Before submitting: {request} needs more LDS than the {limit}-byte group segment allows. I will keep the "
    "direction of the request and reduce shared memory ({changes}).",
    "{request} is over the shared-memory budget for gfx950, which would waste a submission. Adjusted to fit: {changes}.",
)
DUPLICATE_CONTENT_TEMPLATES = (
    "{request} was already evaluated in this session, so resubmitting it would only repeat the earlier result. "
    "Trying an untested neighbour instead: {changes}.",
    "That exact configuration is already in the history. Moving one axis to test something new: {changes}.",
)
HYPOTHESIS_TEMPLATES = (
    "Closest LDS-legal variant of the requested schedule ({changes}); keeps its direction within the {limit}-byte limit.",
    "Requested shape reduced to fit shared memory ({changes}); tests the same idea without a resource rejection.",
)


DUPLICATE_HYPOTHESIS = "Untested neighbour of an already-evaluated configuration ({changes})."


def corrected_message(original: Mapping[str, Any], rejected: Mapping[str, Any], repaired: Mapping[str, Any],
                      variant: int = 0, kind: str = "resource_rejected") -> dict[str, Any]:
    """Assistant message submitting ``repaired`` in place of the rejected or duplicate request."""
    fields = {"request": f"The requested configuration ({describe_config(rejected)})",
              "changes": describe_changes(rejected, repaired), "limit": LDS_LIMIT_BYTES}
    if kind == "duplicate":
        content = DUPLICATE_CONTENT_TEMPLATES[variant % len(DUPLICATE_CONTENT_TEMPLATES)].format(**fields)
        content = f"{content}\nLDS check: {lds_check(repaired)}."
        hypothesis = DUPLICATE_HYPOTHESIS.format(**fields)
    else:
        content = CONTENT_TEMPLATES[variant % len(CONTENT_TEMPLATES)].format(**fields)
        content = f"LDS check of the request: {lds_check(rejected)}.\n{content}\nLDS check of the submission: {lds_check(repaired)}."
        hypothesis = HYPOTHESIS_TEMPLATES[variant % len(HYPOTHESIS_TEMPLATES)].format(**fields)
    message = copy.deepcopy(dict(original))
    message["content"] = content
    call = message["tool_calls"][0]
    call["function"]["arguments"] = json.dumps(submission_arguments(repaired, hypothesis))
    message["tool_calls"] = [call]
    return message


def annotate_lds(message: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """``message`` with its submitted config's exact LDS arithmetic as the only LDS line (model-written ones dropped)."""
    annotated = copy.deepcopy(dict(message))
    lines = (annotated.get("content") or "").strip().splitlines()
    rest = "\n".join(line for line in lines if not line.lstrip().startswith("LDS check")).strip()
    line = f"LDS check: {lds_check(config)}."
    annotated["content"] = f"{line}\n{rest}" if rest else line
    return annotated


def correction_sample(record: Mapping[str, Any], step_index: int, repaired: Mapping[str, Any], rejected: Mapping[str, Any],
                      kind: str = "resource_rejected") -> dict[str, Any]:
    """SFT sample whose only target is the corrected turn."""
    episode = record["episode"]
    step = episode["steps"][step_index]
    assistant = [i for i, m in enumerate(episode["messages"]) if m.get("role") == "assistant"]
    target = assistant[step["turn_index"]]
    variant = sum(map(ord, f"{record['key']}#{step_index}"))
    messages = list(episode["messages"][:target]) + [corrected_message(episode["messages"][target], rejected, repaired, variant, kind)]
    return {
        "role": "planner",
        "episode_id": episode["episode_id"],
        "source_key": f"{record['key']}#correction{step_index}",
        "messages": messages,
        "tools": (episode.get("environment") or {}).get("tools"),
        "steps": [{"step_id": step["step_index"], "turn_index": step["turn_index"], "assistant_message_index": target,
                   "tool": "submit_schedule", "execution_result": "corrected", "advantage": 1.0}],
    }
