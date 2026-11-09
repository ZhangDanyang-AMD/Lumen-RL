"""A/B evaluation of specialist generations: wasted exploration and end-to-end speed.

Three studies, each run for two arms (``base`` = untuned specialists,
``rsi`` = RSI-trained specialists) under identical prompts, budgets and
MICake evaluator:

* planner: one Planner episode on a fresh MICake session per (instruction, sample).
* critic: one Critic diagnosis on a replayed, fixed history per (context, sample).
* e2e: one frozen-Manager episode with direct submission disabled, so every
  candidate comes from a specialist.

This module holds the per-episode metrics and the statistics; the runner
lives in ``scripts/run_ab_eval.py``.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Callable, Mapping, Sequence

PRODUCTIVE = "measured"
NON_ACTION = frozenset({"report", "diagnosis"})
THRESHOLDS = (1.01, 1.03, 1.04, 1.05)


def step_class(step: Mapping[str, Any]) -> str:
    """Coarse outcome of one specialist step (``measured`` is the only productive one)."""
    result = step["result"]
    label = result.get("execution_result") or ""
    if label == "diagnosis_probed":
        probe = (result.get("observation") or {}).get("probe") or {}
        if probe.get("duplicate"):
            return "duplicate"
        label = probe.get("stage") or "unknown"
    if label == PRODUCTIVE:
        performance = (result.get("observation") or {}).get("performance") or {}
        if performance.get("validity") not in (None, "VALID"):
            return "invalid_measurement"
    return label


def planner_metrics(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Wasted actions and best speedup of one Planner episode."""
    actions = [step_class(s) for s in episode["steps"] if step_class(s) not in NON_ACTION]
    speedups = [
        float(((s["result"].get("observation") or {}).get("performance") or {}).get("geomean_speedup_vs_baseline") or 0.0)
        for s in episode["steps"] if step_class(s) == PRODUCTIVE
    ]
    wasted = sum(1 for a in actions if a != PRODUCTIVE)
    return {
        "actions": len(actions),
        "measured": len(actions) - wasted,
        "wasted": wasted,
        "wasted_rate": wasted / len(actions) if actions else 1.0,
        "resource_rejected": actions.count("resource_rejected"),
        "schema_invalid": actions.count("schema_invalid"),
        "duplicate": actions.count("duplicate"),
        "failed_candidate": sum(actions.count(k) for k in ("verify_fail", "compile_fail", "correctness_fail")),
        "best_speedup": max(speedups, default=1.0),
        "beat_baseline": float(max(speedups, default=1.0) > 1.0),
        "clean_finish": float(episode["terminal_state"] == "AGENT_FINISHED"),
        "turns": len(episode["turns"]),
    }


def critic_metrics(episode: Mapping[str, Any], context_best: float) -> dict[str, Any]:
    """Diagnosis validity and the outcome of the probed recommendation."""
    classes = [step_class(s) for s in episode["steps"]]
    probed = [s for s in episode["steps"] if s["result"].get("execution_result") == "diagnosis_probed"]
    speedup = 0.0
    if probed:
        probe = (probed[0]["result"].get("observation") or {}).get("probe") or {}
        speedup = float(((probe.get("performance") or {}).get("geomean_speedup_vs_baseline")) or 0.0)
    probe_class = step_class(probed[0]) if probed else "no_diagnosis"
    return {
        "attempts": len(classes),
        "schema_invalid": classes.count("schema_invalid"),
        "first_try_valid": float(bool(classes) and classes[0] != "schema_invalid"),
        "probe_class": probe_class,
        "probe_productive": float(probe_class == PRODUCTIVE),
        "probe_speedup": speedup,
        "probe_beats_context_best": float(speedup > context_best),
    }


def e2e_metrics(trace: Sequence[Mapping[str, Any]], budget: int) -> dict[str, Any]:
    """Anytime best curve of one Manager episode from its submission trace.

    ``trace`` rows: ``{"index", "t", "stage", "speedup", "counted"}`` for every
    MICake call in order; ``index`` is the budget used so far, ``t`` seconds
    since the episode started. Uncounted calls (duplicate, schema-invalid)
    cost no budget but still count as wasted actions.
    """
    best = 1.0
    best_at: dict[str, float | None] = {"submissions": None, "seconds": None}
    curve: list[float] = []
    reach: dict[str, dict[str, float | None]] = {f"{tau:.2f}": {"submissions": None, "seconds": None} for tau in THRESHOLDS}
    for row in trace:
        if not row.get("counted", True):
            continue
        if row["stage"] == PRODUCTIVE and row.get("speedup") and float(row["speedup"]) > best:
            best = float(row["speedup"])
            best_at = {"submissions": float(row["index"]), "seconds": float(row["t"])}
        curve.append(best)
        for tau in THRESHOLDS:
            key = f"{tau:.2f}"
            if reach[key]["submissions"] is None and best >= tau:
                reach[key] = {"submissions": float(row["index"]), "seconds": float(row["t"])}
    padded = (curve + [best] * budget)[:budget] if curve else [1.0] * budget
    wasted = sum(1 for row in trace if row["stage"] != PRODUCTIVE)
    return {
        "actions": len(trace),
        "submissions": len(curve),
        "wasted": wasted,
        "wasted_rate": wasted / len(trace) if trace else 1.0,
        "resource_rejected": sum(1 for row in trace if row["stage"] == "resource_rejected"),
        "final_best": best,
        "anytime_log_auc": statistics.fmean(math.log(v) for v in padded),
        "reach": reach,
        "best_at": best_at,
    }


def censored(value: float | None, ceiling: float) -> float:
    """Value for rank statistics; never reaching the threshold counts as ``ceiling``."""
    return ceiling if value is None else value


def bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """``stat(b) - stat(a)`` with a percentile 95% bootstrap interval."""
    rng = random.Random(seed)
    observed = stat(b) - stat(a)
    draws = sorted(
        stat([rng.choice(b) for _ in b]) - stat([rng.choice(a) for _ in a])
        for _ in range(resamples)
    )
    return {"diff": observed, "ci_low": draws[int(0.025 * resamples)], "ci_high": draws[int(0.975 * resamples) - 1]}


def permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for ``stat(b) - stat(a)``."""
    rng = random.Random(seed)
    observed = abs(stat(b) - stat(a))
    pooled = list(a) + list(b)
    hits = 0
    for _ in range(resamples):
        rng.shuffle(pooled)
        if abs(stat(pooled[len(a):]) - stat(pooled[: len(a)])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def stratified_compare(
    a: Mapping[Any, Sequence[float]],
    b: Mapping[Any, Sequence[float]],
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean of per-stratum ``mean(b) - mean(a)`` with stratified bootstrap CI and within-stratum permutation p.

    Strata are instructions (or contexts): arm labels are only exchanged
    inside a stratum, so between-instruction difficulty does not dilute the
    arm effect.
    """
    strata = [s for s in a if s in b and a[s] and b[s]]
    if not strata:
        return {}

    def effect(pairs):
        return statistics.fmean(statistics.fmean(y) - statistics.fmean(x) for x, y in pairs)

    observed = effect([(a[s], b[s]) for s in strata])
    rng = random.Random(seed)
    draws = sorted(
        effect([([rng.choice(a[s]) for _ in a[s]], [rng.choice(b[s]) for _ in b[s]]) for s in strata])
        for _ in range(resamples)
    )
    hits = 0
    for _ in range(resamples):
        pairs = []
        for s in strata:
            pooled = list(a[s]) + list(b[s])
            rng.shuffle(pooled)
            pairs.append((pooled[: len(a[s])], pooled[len(a[s]):]))
        if abs(effect(pairs)) >= abs(observed) - 1e-12:
            hits += 1
    return {"strata": len(strata), "diff": observed, "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "p": (hits + 1) / (resamples + 1)}


def compare(a: Sequence[float], b: Sequence[float], stat: Callable[[Sequence[float]], float] = statistics.fmean) -> dict[str, Any]:
    """Arm summaries plus the bootstrap interval and permutation p of ``b - a``."""
    if not a or not b:
        return {"n_base": len(a), "n_rsi": len(b)}
    return {
        "n_base": len(a),
        "n_rsi": len(b),
        "base": stat(a),
        "rsi": stat(b),
        **bootstrap_diff(a, b, stat),
        "p": permutation_p(a, b, stat),
    }
