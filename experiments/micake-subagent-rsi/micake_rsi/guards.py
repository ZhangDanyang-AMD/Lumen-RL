"""Measurement guards adopted from GEAK's kernel workflow.

* Harness fingerprint: every generation records a digest of the MICake
  evaluator sources it was measured with. Two generations are only compared
  on the same fingerprint; when it moved (e.g. a debug repair), the champion's
  best configs are re-measured on the current harness first.
* Independent verification: the best config of every Manager episode is
  re-measured on a fresh worker ``verify_repeats`` times. The median is the
  official number; the in-episode measurement is only a claim.
* Arbitration: a verified result more than ``max_claim_shortfall`` below the
  claim is ``flagged``; a re-measurement that is not a valid correct
  measurement is ``correctness_failed`` (official speedup 1.0, baseline kept).
* Noise band: the run-to-run spread observed during verification widens the
  promotion gain threshold, so a win must clear measured noise.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.trajectory import Episode
from sandbox.micake.environment import MicakeGemmEnvironment

logger = logging.getLogger(__name__)

VERIFIED = "verified"
FLAGGED = "flagged"
CORRECTNESS_FAILED = "correctness_failed"
BASELINE = "baseline"
UNTRUSTED = frozenset({FLAGGED, CORRECTNESS_FAILED})
SKIP_PARTS = frozenset({"__pycache__", ".git", ".venv"})


@dataclass(frozen=True)
class GuardSettings:
    """Verification and promotion guards (values come from YAML)."""

    enabled: bool
    verify_repeats: int
    max_repeat_spread: float
    max_claim_shortfall: float
    noise_band_cap: float
    max_flagged_rate: float
    harness_roots: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GuardSettings":
        return cls(
            enabled=bool(value["enabled"]),
            verify_repeats=int(value["verify_repeats"]),
            max_repeat_spread=float(value["max_repeat_spread"]),
            max_claim_shortfall=float(value["max_claim_shortfall"]),
            noise_band_cap=float(value["noise_band_cap"]),
            max_flagged_rate=float(value["max_flagged_rate"]),
            harness_roots=tuple(value["harness_roots"]),
        )


def harness_fingerprint(roots: Sequence[str]) -> dict[str, Any]:
    """sha256 over every ``*.py`` file (path + content) under ``roots``."""
    digest = hashlib.sha256()
    count = 0
    for root in roots:
        base = Path(root).resolve()
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS.intersection(path.parts):
                continue
            digest.update(str(path.relative_to(base.parent)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def _speedup(result: Mapping[str, Any]) -> float | None:
    performance = result.get("performance") or {}
    speedup = performance.get("geomean_speedup")
    if result.get("stage") != "measured" or performance.get("validity") != "VALID" or not speedup:
        return None
    return float(speedup)


async def remeasure(micake: MicakeGemmEnvironment, config: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Measure ``config`` ``repeats`` times on a fresh worker; median of valid runs."""
    await micake.restart_evaluator()
    runs: list[float] = []
    stages: list[str] = []
    for _ in range(repeats):
        result = await micake.measure(config)
        stages.append(str(result.get("stage")))
        speedup = _speedup(result)
        if speedup is None:
            return {"runs": runs, "stages": stages, "speedup": None, "spread": None}
        runs.append(speedup)
    median = statistics.median(runs)
    return {"runs": runs, "stages": stages, "speedup": median, "spread": (max(runs) - min(runs)) / median}


def arbitrate(claim: float, measured: Mapping[str, Any], settings: GuardSettings) -> dict[str, Any]:
    """Director rule: verified numbers are official; a large shortfall is flagged."""
    if measured["speedup"] is None:
        return {"status": CORRECTNESS_FAILED, "claim": claim, "speedup": 1.0, **measured}
    status = FLAGGED if measured["speedup"] < claim * (1.0 - settings.max_claim_shortfall) else VERIFIED
    noisy = measured["spread"] > settings.max_repeat_spread
    return {"status": status, "claim": claim, "noisy": noisy, **measured}


async def verify_generation(
    micake: MicakeGemmEnvironment,
    managers: Sequence[Episode],
    settings: GuardSettings,
) -> list[dict[str, Any]]:
    """Re-measure each Manager episode's best config and attach the verdict to its summary."""
    records: list[dict[str, Any]] = []
    for manager in managers:
        summary = manager.environment.setdefault("final_summary", {})
        claim = float(summary.get("best_speedup", 1.0))
        if summary.get("best_source") != "agent" or not summary.get("best_config"):
            verdict = {"status": BASELINE, "claim": claim, "speedup": 1.0}
        else:
            measured = await remeasure(micake, summary["best_config"], settings.verify_repeats)
            verdict = arbitrate(claim, measured, settings)
        summary["verification"] = verdict
        summary["best_speedup_verified"] = verdict["speedup"]
        records.append({
            "episode_id": manager.episode_id,
            "task_id": manager.task.get("task_id"),
            "config": summary.get("best_config") if verdict["status"] != BASELINE else None,
            **verdict,
        })
        logger.info("verify %s: %s claim %.4f verified %.4f",
                    manager.episode_id[:8], verdict["status"], claim, verdict["speedup"])
    return records


def verification_summary(records: Sequence[Mapping[str, Any]], settings: GuardSettings) -> dict[str, Any]:
    """Counts, flagged rate and the noise band measured across verified episodes."""
    spreads = [float(r["spread"]) for r in records if r.get("spread") is not None]
    statuses = [r["status"] for r in records]
    measured = [s for s in statuses if s != BASELINE]
    return {
        "statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
        "flagged_rate": (sum(s in UNTRUSTED for s in measured) / len(measured)) if measured else 0.0,
        "noise_band": min(max(spreads), settings.noise_band_cap) if spreads else 0.0,
    }


async def reverify_champion(
    micake: MicakeGemmEnvironment,
    records: Sequence[Mapping[str, Any]],
    settings: GuardSettings,
) -> dict[str, Any]:
    """Re-measure the champion's verified bests on the current harness."""
    speedups: list[float] = []
    details: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") == BASELINE or not record.get("config"):
            speedups.append(1.0)
            continue
        measured = await remeasure(micake, record["config"], settings.verify_repeats)
        speedup = measured["speedup"] if measured["speedup"] is not None else 1.0
        speedups.append(speedup)
        details.append({"episode_id": record.get("episode_id"), "previous": record.get("speedup"), **measured})
    return {
        "mean_best_speedup": statistics.fmean(speedups) if speedups else None,
        "episodes": details,
    }
