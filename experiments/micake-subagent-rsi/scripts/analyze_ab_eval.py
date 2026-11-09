"""Summarize ``run_ab_eval.py`` results: per-arm metrics and ``rsi`` vs each other arm.

Example:
    python experiments/micake-subagent-rsi/scripts/analyze_ab_eval.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from micake_rsi.ab_eval import censored, compare, stratified_compare  # noqa: E402

PLANNER_KEYS = ("wasted_rate", "resource_rejected", "schema_invalid", "duplicate", "failed_candidate",
                "best_speedup", "beat_baseline", "clean_finish", "turns")
CRITIC_KEYS = ("first_try_valid", "schema_invalid", "probe_productive", "probe_speedup", "probe_beats_context_best")
E2E_KEYS = ("final_best", "wasted_rate", "resource_rejected", "anytime_log_auc")


def load(path: Path) -> list[dict]:
    """Records of ``<study>.jsonl`` and its shards ``<study>_*.jsonl``."""
    records = []
    for part in sorted(path.parent.glob(f"{path.stem}*.jsonl")):
        records += [json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records


def by_arm(records: list[dict], value) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        out[r["arm"]].append(float(value(r)))
    return out


REFERENCE = "rsi"


def contrast(series: dict[str, list[float]], reference: str | None = None) -> dict[str, dict]:
    reference = reference or REFERENCE
    if reference not in series:
        return {}
    return {other: compare(series[other], series[reference]) for other in sorted(series) if other != reference}


def stratified(records: list[dict], key: str) -> dict[str, dict]:
    """Per-instruction stratified contrast of ``REFERENCE`` against every other arm."""
    cells: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if "item" in r:
            cells[r["arm"]][r["item"]].append(float(r["metrics"][key]))
    if REFERENCE not in cells:
        return {}
    return {other: stratified_compare(cells[other], cells[REFERENCE]) for other in sorted(cells) if other != REFERENCE}


def summarize(records: list[dict], keys, extra=None) -> dict:
    out = {}
    for key in keys:
        series = by_arm(records, lambda r, k=key: r["metrics"][k])
        out[key] = {"mean": {a: statistics.fmean(v) for a, v in series.items()}, "n": {a: len(v) for a, v in series.items()},
                    "rsi_minus": contrast(series), "stratified": stratified(records, key)}
    for key, fn in (extra or {}).items():
        series = by_arm(records, fn)
        out[key] = {"mean": {a: statistics.fmean(v) for a, v in series.items()}, "n": {a: len(v) for a, v in series.items()},
                    "rsi_minus": contrast(series)}
    return out


def e2e_extra(budget: int = 10) -> dict:
    extra = {
        "wall_s": lambda r: r["wall_s"],
        "eval_s": lambda r: r["eval_s"],
        "llm_s": lambda r: r["manager_llm_s"] + r["specialist_llm_s"],
        "verified_best": lambda r: ((r.get("verified") or {}).get("speedup") or 1.0)
        if r["best"]["source"] == "agent" else 1.0,
    }
    for tau in ("1.01", "1.03", "1.04", "1.05"):
        extra[f"reached_{tau}"] = lambda r, t=tau: float(r["metrics"]["reach"][t]["submissions"] is not None)
        extra[f"subs_to_{tau}"] = lambda r, t=tau: censored(r["metrics"]["reach"][t]["submissions"], budget + 1)
        extra[f"wall_to_{tau}"] = lambda r, t=tau: censored(r["metrics"]["reach"][t]["seconds"], r["wall_s"] * 2)
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "ab_eval")
    parser.add_argument("--reference", default="rsi", help="arm every other arm is compared against")
    args = parser.parse_args()
    global REFERENCE
    REFERENCE = args.reference
    report = {
        "planner": summarize(load(args.dir / "planner.jsonl"), PLANNER_KEYS,
                             {"wall_s": lambda r: r["wall_s"], "llm_s": lambda r: r["llm_s"], "eval_s": lambda r: r["eval_s"]}),
        "critic": summarize(load(args.dir / "critic.jsonl"), CRITIC_KEYS,
                            {"wall_s": lambda r: r["wall_s"], "llm_s": lambda r: r["llm_s"]}),
        "e2e": summarize(load(args.dir / "e2e.jsonl"), E2E_KEYS, e2e_extra()),
    }
    (args.dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for study, metrics in report.items():
        print(f"== {study}")
        for key, value in metrics.items():
            means = "  ".join(f"{a}={m:.4f}(n={value['n'][a]})" for a, m in sorted(value["mean"].items()))
            diffs = "  ".join(
                f"{REFERENCE}-{o}={c['diff']:+.4f}[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]p={c['p']:.3f}"
                for o, c in value["rsi_minus"].items() if "diff" in c
            )
            strat = "  ".join(
                f"strat {REFERENCE}-{o}={c['diff']:+.4f}[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]p={c['p']:.3f}"
                for o, c in (value.get("stratified") or {}).items() if "diff" in c
            )
            print(f"  {key:24s} {means}  |  {diffs}  ||  {strat}")


if __name__ == "__main__":
    main()
