"""Rule-based search harness: the no-LLM reference arm of the A/B benchmark.

Greedy coordinate descent over MICake's closed domain: from the incumbent,
propose one-axis moves to adjacent domain values in a fixed axis order, skip
configs whose shared memory exceeds the LDS cap, and accept a move
when it beats the incumbent by the noise band.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterator, Mapping, Sequence

from sandbox.micake.tool_runtime import HIDDEN_FIXED_FIELDS

LDS_CAP_BYTES = 163840
AXIS_ORDER = (
    ("realization", "num_warps"),
    ("realization", "matrix_instr_nonkdim"),
    ("realization", "program_order"),
    ("realization", "group_m"),
    ("schedule", "tile_n"),
    ("schedule", "tile_m"),
    ("schedule", "tile_k"),
    ("realization", "num_stages"),
    ("realization", "waves_per_eu"),
    ("schedule", "unroll_factor"),
    ("schedule", "vectorize_factor"),
)
LINKED = {("realization", "num_stages"): ("schedule", "pipeline_depth")}


def lds_multiplier(schedule: Mapping[str, Any]) -> int:
    """Buffered FP16 A+B tiles held in LDS: (pipeline_depth - 1) * unroll_factor, except depth 4 / unroll 2 -> 2.

    Fitted exactly on every compiled MICake C144 config (653 configs, zero
    mismatches); num_stages, num_warps and vectorize_factor do not change it.
    """
    depth, unroll = schedule["pipeline_depth"], schedule["unroll_factor"]
    return 2 if (depth, unroll) == (4, 2) else (depth - 1) * unroll


def shared_bytes(config: Mapping[str, Any]) -> int:
    s = config["schedule"]
    return (s["tile_m"] + s["tile_n"]) * s["tile_k"] * 2 * lds_multiplier(s)


def lds_check(config: Mapping[str, Any]) -> str:
    """Step-by-step arithmetic of ``shared_bytes`` against the cap; each step only uses numbers already written."""
    s = config["schedule"]
    depth, unroll = s["pipeline_depth"], s["unroll_factor"]
    buffers = lds_multiplier(s)
    per_buffer = (s["tile_m"] + s["tile_n"]) * s["tile_k"] * 2
    value = per_buffer * buffers
    if (depth, unroll) == (4, 2):
        buffers_text = f"buffers = {buffers} (pipeline_depth 4 with unroll 2)"
    else:
        buffers_text = f"buffers = (pipeline_depth {depth} - 1) * unroll {unroll} = {buffers}"
    verdict = f"<= {LDS_CAP_BYTES} B, fits" if value <= LDS_CAP_BYTES else f"> {LDS_CAP_BYTES} B, too big"
    return (f"{buffers_text}; per buffer = (tile_m {s['tile_m']} + tile_n {s['tile_n']}) * tile_k {s['tile_k']} * 2 B "
            f"= {per_buffer} B; LDS = {per_buffer} * {buffers} = {value} B {verdict}")


def fits_lds(config: Mapping[str, Any]) -> bool:
    return shared_bytes(config) <= LDS_CAP_BYTES


def config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


def neighbours(config: Mapping[str, Any], domain: Mapping[str, Mapping[str, Sequence[Any]]]) -> Iterator[dict[str, Any]]:
    """One-axis moves to adjacent domain values, in ``AXIS_ORDER``."""
    for section, name in AXIS_ORDER:
        values = list(domain.get(section, {}).get(name, ()))
        if name in HIDDEN_FIXED_FIELDS.get(section, ()) or config[section].get(name) not in values:
            continue
        index = values.index(config[section][name])
        for step in (1, -1):
            if not 0 <= index + step < len(values):
                continue
            candidate = copy.deepcopy(dict(config))
            candidate[section][name] = values[index + step]
            linked = LINKED.get((section, name))
            if linked and candidate[section][name] in domain[linked[0]].get(linked[1], ()):
                candidate[linked[0]][linked[1]] = candidate[section][name]
            yield candidate


def submission_arguments(config: Mapping[str, Any], hypothesis: str) -> dict[str, Any]:
    """``submit_schedule`` arguments for ``config`` (hidden fixed fields removed)."""
    args: dict[str, Any] = {}
    for section in ("schedule", "realization"):
        hidden = HIDDEN_FIXED_FIELDS.get(section, ())
        args[section] = {k: v for k, v in config[section].items() if k not in hidden}
    args["hypothesis"] = hypothesis
    return args


class GreedySearch:
    """Stateful proposer: call :meth:`propose`, then :meth:`observe` with the outcome."""

    def __init__(self, baseline: Mapping[str, Any], domain: Mapping[str, Any], min_gain: float = 0.005) -> None:
        self.domain = domain
        self.min_gain = min_gain
        self.incumbent = copy.deepcopy(dict(baseline))
        self.best = 1.0
        self.tried = {config_key(baseline)}
        self._pending: Iterator[dict[str, Any]] = neighbours(self.incumbent, domain)

    def propose(self) -> dict[str, Any] | None:
        for candidate in self._pending:
            key = config_key(candidate)
            if key in self.tried or not fits_lds(candidate):
                continue
            self.tried.add(key)
            return candidate
        return None

    def observe(self, config: Mapping[str, Any], speedup: float | None) -> None:
        if speedup is not None and speedup > self.best * (1.0 + self.min_gain):
            self.best = speedup
            self.incumbent = copy.deepcopy(dict(config))
            self._pending = neighbours(self.incumbent, self.domain)
