"""Champion/challenger gate and DSV4.1 comparison for Planner RSI generations.

Scores come from interleaved end-to-end episodes (Manager and Critic are
DSV4.1 in every arm; only the Planner differs). A challenger replaces the
champion when it finds faster kernels, or reaches the target speedup sooner
without giving up kernel speed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from micake_rsi.ab_eval import censored, compare


@dataclass(frozen=True)
class PlannerGate:
    tau: float = 1.04
    min_best_gain: float = 0.005
    max_best_loss_for_speed: float = 0.002
    max_time_ratio: float = 0.85
    max_waste_increase: float = 0.05


def verified_best(record: Mapping[str, Any]) -> float:
    if (record.get("best") or {}).get("source") != "agent":
        return 1.0
    return float((record.get("verified") or {}).get("speedup") or record["metrics"]["final_best"])


UNCOUNTED = frozenset({"duplicate", "schema_invalid"})


def budget_waste(record: Mapping[str, Any]) -> float:
    """Failed share of budget-consuming submissions; uncounted stages cost only time, which ``time_to_tau`` measures."""
    counted = [row["stage"] for row in record.get("trace") or () if row["stage"] not in UNCOUNTED]
    return sum(stage != "measured" for stage in counted) / len(counted) if counted else 1.0


def episode_scores(records: Sequence[Mapping[str, Any]], arm: str, tau: float) -> dict[str, list[float]]:
    key = f"{tau:.2f}"
    rows = [r for r in records if r["arm"] == arm]
    return {
        "best": [verified_best(r) for r in rows],
        "reached": [float(r["metrics"]["reach"][key]["submissions"] is not None) for r in rows],
        "time_to_tau": [censored(r["metrics"]["reach"][key]["seconds"], 2 * r["wall_s"]) for r in rows],
        "subs_to_tau": [censored(r["metrics"]["reach"][key]["submissions"], 11.0) for r in rows],
        "time_to_best": [float((r["metrics"].get("best_at") or {}).get("seconds") or r["wall_s"]) for r in rows],
        "wasted_rate": [float(r["metrics"]["wasted_rate"]) for r in rows],
        "budget_waste": [budget_waste(r) for r in rows],
        "wall_s": [float(r["wall_s"]) for r in rows],
    }


def summarize(scores: Mapping[str, list[float]]) -> dict[str, float]:
    if not scores["best"]:
        return {"n": 0}
    return {"n": len(scores["best"]), "mean_best": statistics.fmean(scores["best"]), "max_best": max(scores["best"]),
            "reached_rate": statistics.fmean(scores["reached"]), "median_time_to_tau": statistics.median(scores["time_to_tau"]),
            "median_subs_to_tau": statistics.median(scores["subs_to_tau"]),
            "median_time_to_best": statistics.median(scores["time_to_best"]),
            "wasted_rate": statistics.fmean(scores["wasted_rate"]), "budget_waste": statistics.fmean(scores["budget_waste"]),
            "mean_wall_s": statistics.fmean(scores["wall_s"])}


SIGNIFICANCE = 0.05


def kernel_beats_reference(records: Sequence[Mapping[str, Any]], name: str, gate: PlannerGate, reference: str = "dsv4") -> bool:
    """Faster kernel than ``reference`` by the gate margin, with a permutation p below ``SIGNIFICANCE``."""
    best = versus(records, name, reference, gate.tau).get("best") or {}
    if not best or best.get("p") is None:
        return False
    return best["rsi"] - best["base"] >= gate.min_best_gain and best["p"] < SIGNIFICANCE


def as_fast_as_reference(records: Sequence[Mapping[str, Any]], name: str, gate: PlannerGate, reference: str = "dsv4") -> bool:
    """Median time to ``gate.tau`` is no slower than ``reference``. Episodes that miss the threshold are censored."""
    compared = versus(records, name, reference, gate.tau)
    arm, ref = compared.get("arm") or {}, compared.get("reference") or {}
    if "median_time_to_tau" not in arm or "median_time_to_tau" not in ref:
        return False
    return arm["median_time_to_tau"] <= ref["median_time_to_tau"]


def beats_reference(records: Sequence[Mapping[str, Any]], name: str, gate: PlannerGate, reference: str = "dsv4") -> bool:
    """Held-out stop: a significantly faster kernel overall and per dtype, and a median time to tau that is not slower."""
    if not kernel_beats_reference(records, name, gate, reference):
        return False
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get("dtype") or "fp16", []).append(record)
    return (all(kernel_beats_reference(group, name, gate, reference) for group in groups.values())
            and as_fast_as_reference(records, name, gate, reference))


def versus(records: Sequence[Mapping[str, Any]], arm: str, reference: str, tau: float) -> dict[str, Any]:
    """``arm`` minus ``reference`` on kernel speed and on the time to reach ``tau``."""
    a, b = episode_scores(records, reference, tau), episode_scores(records, arm, tau)
    if not a["best"] or not b["best"]:
        return {}
    return {"arm": summarize(b), "reference": summarize(a),
            "best": compare(a["best"], b["best"]),
            "time_to_tau": compare(a["time_to_tau"], b["time_to_tau"], statistics.median),
            "reached": compare(a["reached"], b["reached"])}


def gate_decision(records: Sequence[Mapping[str, Any]], champion: str, challenger: str, gate: PlannerGate) -> dict[str, Any]:
    old, new = summarize(episode_scores(records, champion, gate.tau)), summarize(episode_scores(records, challenger, gate.tau))
    if not old.get("n") or not new.get("n"):
        return {"decision": "HOLD", "reasons": ["missing episodes"], "champion": old, "challenger": new}
    gain = new["mean_best"] - old["mean_best"]
    reasons: list[str] = []
    if new["budget_waste"] > old["budget_waste"] + gate.max_waste_increase:
        reasons.append(f"budget waste {new['budget_waste']:.3f} > champion {old['budget_waste']:.3f} + {gate.max_waste_increase}")
    faster_kernel = gain >= gate.min_best_gain
    faster_path = (gain >= -gate.max_best_loss_for_speed
                   and new["median_time_to_tau"] <= gate.max_time_ratio * old["median_time_to_tau"]
                   and new["reached_rate"] >= old["reached_rate"])
    if not (faster_kernel or faster_path):
        reasons.append(f"best gain {gain:+.4f} < {gate.min_best_gain} and time to {gate.tau} "
                       f"{new['median_time_to_tau']:.0f}s vs {old['median_time_to_tau']:.0f}s does not qualify")
    return {"decision": "HOLD" if reasons else "PROMOTE", "reasons": reasons, "best_gain": gain,
            "route": "faster_kernel" if faster_kernel else ("faster_path" if faster_path else None),
            "champion": old, "challenger": new}
