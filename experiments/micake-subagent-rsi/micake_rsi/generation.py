"""Generation metrics, lineage manifest and champion/challenger promotion.

A generation = one set of specialist bindings rolled out under the frozen
Manager on a fixed task set. Its manifest records exactly which weights,
evaluator and config produced its data, which datasets it emitted, and the
promotion decision against the current champion generation.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.trajectory import Episode
from micake_rsi.rewards import SpecialistCredit

MANIFEST_SCHEMA = "micake_rsi_generation_v1"
CORRECT_STAGES = ("correct", "measured")


def git_commit(path: str | Path) -> str | None:
    """HEAD commit of the repo containing ``path`` (None outside git)."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def generation_metrics(
    managers: Sequence[Episode],
    children: Mapping[str, Sequence[Episode]],
    credits: Mapping[str, SpecialistCredit],
) -> dict[str, Any]:
    """Manager-level outcome and per-role specialist quality metrics."""
    def _specialists(manager: Episode) -> list[Episode]:
        return [child for child in children.get(manager.episode_id, ()) if child.role != "debugger"]

    debug_episodes = [child for manager in managers for child in children.get(manager.episode_id, ()) if child.role == "debugger"]
    summaries = [m.environment.get("final_summary") or {} for m in managers]
    claimed = [float(s.get("best_speedup", 1.0)) for s in summaries]
    best = [float(s.get("best_speedup_verified", s.get("best_speedup", 1.0))) for s in summaries]
    manager = {
        "episodes": len(managers),
        "mean_best_speedup": _mean(best),
        "mean_claimed_best_speedup": _mean(claimed),
        "median_best_speedup": statistics.median(best) if best else None,
        "beat_baseline_rate": _mean([float(b > 1.0) for b in best]),
        "terminal_states": dict(Counter(m.terminal_state for m in managers)),
        "delegations": dict(Counter(c.role for m in managers for c in _specialists(m))),
        "direct_submissions": sum(
            1 for m in managers for s in m.steps
            if s.action.name == "submit_schedule" and s.result.execution_result != "schema_invalid"
        ),
    }

    by_role: dict[str, list[Episode]] = defaultdict(list)
    for m in managers:
        for child in _specialists(m):
            by_role[child.role or ""].append(child)
    roles: dict[str, Any] = {}
    for role, episodes in sorted(by_role.items()):
        steps = [s for ep in episodes for s in ep.steps]
        submissions = [
            sub for ep in episodes for sub in ((ep.environment.get("final_summary") or {}).get("submissions") or [])
        ]
        roles[role] = {
            "episodes": len(episodes),
            "steps": len(steps),
            "schema_invalid_rate": _mean([float(s.result.execution_result == "schema_invalid") for s in steps]),
            "clean_finish_rate": _mean([float(ep.terminal_state == "AGENT_FINISHED") for ep in episodes]),
            "submissions": len(submissions),
            "correctness_rate": _mean([float(s.get("stage") in CORRECT_STAGES) for s in submissions]),
            "new_best_rate": _mean([float(bool(s.get("new_best"))) for s in submissions]),
            "mean_local_reward": _mean([credits[ep.episode_id].local for ep in episodes]),
            "mean_contribution": _mean([credits[ep.episode_id].contribution for ep in episodes]),
            "mean_total_reward": _mean([credits[ep.episode_id].total for ep in episodes]),
        }
    debug = {
        "episodes": len(debug_episodes),
        "retries": sum(1 for episode in debug_episodes if (episode.environment.get("final_summary") or {}).get("retry")),
        "files": [
            path
            for episode in debug_episodes
            for path in ((episode.environment.get("final_summary") or {}).get("files") or [])
        ],
    }
    return {"manager": manager, "roles": roles, "debug": debug}


@dataclass(frozen=True)
class PromotionGate:
    """Challenger must match the champion on quality and beat it on outcome (values from YAML)."""

    min_role_episodes: int
    max_schema_invalid_rate: float
    max_correctness_regression: float
    min_speedup_gain: float
    max_flagged_rate: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromotionGate":
        return cls(
            min_role_episodes=int(value["min_role_episodes"]),
            max_schema_invalid_rate=float(value["max_schema_invalid_rate"]),
            max_correctness_regression=float(value["max_correctness_regression"]),
            min_speedup_gain=float(value["min_speedup_gain"]),
            max_flagged_rate=float(value["max_flagged_rate"]),
        )


def promotion_decision(
    challenger: Mapping[str, Any],
    champion: Mapping[str, Any] | None,
    gate: PromotionGate,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``BOOTSTRAP`` (no champion yet), ``PROMOTE`` or ``HOLD`` with reasons.

    ``verification`` carries the guard results of this generation: ``flagged_rate``,
    ``noise_band`` and, when the harness changed, ``champion_mean_best_speedup``
    re-measured on the current harness.
    """
    verification = verification or {}
    reasons: list[str] = []
    flagged_rate = float(verification.get("flagged_rate") or 0.0)
    if flagged_rate > gate.max_flagged_rate:
        reasons.append(f"flagged_rate {flagged_rate:.3f} > {gate.max_flagged_rate}")
    for role, m in challenger["roles"].items():
        if m["episodes"] < gate.min_role_episodes:
            reasons.append(f"{role}: only {m['episodes']} episodes (< {gate.min_role_episodes})")
        if (m["schema_invalid_rate"] or 0.0) > gate.max_schema_invalid_rate:
            reasons.append(f"{role}: schema_invalid_rate {m['schema_invalid_rate']:.3f} > {gate.max_schema_invalid_rate}")
    if champion is None:
        return {"decision": "BOOTSTRAP", "reasons": reasons}

    for role, m in challenger["roles"].items():
        ref = champion["roles"].get(role)
        if ref and ref["correctness_rate"] is not None and m["correctness_rate"] is not None:
            if m["correctness_rate"] < ref["correctness_rate"] - gate.max_correctness_regression:
                reasons.append(f"{role}: correctness {m['correctness_rate']:.3f} regressed from {ref['correctness_rate']:.3f}")
    reference = verification.get("champion_mean_best_speedup")
    if reference is None:
        reference = champion["manager"]["mean_best_speedup"]
    gain = (challenger["manager"]["mean_best_speedup"] or 1.0) - (reference or 1.0)
    threshold = max(gate.min_speedup_gain, float(verification.get("noise_band") or 0.0))
    if gain < threshold:
        reasons.append(f"mean best speedup gain {gain:+.4f} < {threshold:.4f} (max of min gain and noise band)")
    return {
        "decision": "HOLD" if reasons else "PROMOTE",
        "reasons": reasons,
        "speedup_gain": gain,
        "gain_threshold": threshold,
        "champion_reference": reference,
    }


class Lineage:
    """Champion pointer and generation manifests under one RSI root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def champion_path(self) -> Path:
        return self.root / "champion.json"

    def champion(self) -> dict[str, Any] | None:
        if not self.champion_path.exists():
            return None
        return json.loads(self.champion_path.read_text(encoding="utf-8"))

    def record(self, manifest: dict[str, Any]) -> Path:
        """Write the manifest; move the champion pointer on BOOTSTRAP/PROMOTE."""
        path = self.root / manifest["generation_id"] / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if manifest["promotion"]["decision"] in ("BOOTSTRAP", "PROMOTE"):
            champion = {
                "generation_id": manifest["generation_id"],
                "specialists": manifest["specialists"],
                "metrics": manifest["metrics"],
                "guards": manifest.get("guards"),
                "manifest": str(path),
            }
            self.champion_path.write_text(json.dumps(champion, indent=2, sort_keys=True), encoding="utf-8")
        return path
