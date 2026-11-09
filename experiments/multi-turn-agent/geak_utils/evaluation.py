"""Evaluation result models and deterministic latency aggregation."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_PERF_RE = re.compile(
    r"Perf:\s*([0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*\(([^)]+)\))?", re.I
)
_GEAK_RE = re.compile(
    r"GEAK_RESULT_LATENCY_MS\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:[^\n]*?(?:case|id)\s*[=:]\s*([A-Za-z0-9_.:/-]+))?",
    re.I,
)


@dataclass
class CommandResult:
    mode: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    per_case_ms: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self, output_limit: int = 12000) -> dict[str, Any]:
        value = asdict(self)
        value["ok"] = self.ok
        value["stdout"] = self.stdout[-output_limit:]
        value["stderr"] = self.stderr[-output_limit:]
        return value


@dataclass
class EvaluationResult:
    compiled: bool
    correct: bool
    speedup_geomean: float
    speedup_arithmetic: float
    baseline_ms: dict[str, float]
    candidate_ms: dict[str, float]
    compile: CommandResult | None
    correctness: CommandResult
    performance: CommandResult | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "correct": self.correct,
            "speedup_geomean": self.speedup_geomean,
            "speedup_arithmetic": self.speedup_arithmetic,
            "baseline_ms": self.baseline_ms,
            "candidate_ms": self.candidate_ms,
            "compile": self.compile.to_dict() if self.compile else None,
            "correctness": self.correctness.to_dict(),
            "performance": self.performance.to_dict() if self.performance else None,
            "error": self.error,
        }


def parse_performance_output(output: str) -> dict[str, float]:
    """Parse named or positional per-case latency lines."""

    values: dict[str, float] = {}
    unnamed = 0
    for pattern in (_PERF_RE, _GEAK_RE):
        for match in pattern.finditer(output):
            name = match.group(2) or "case_%d" % unnamed
            if match.group(2) is None:
                unnamed += 1
            values[name] = float(match.group(1))
    return values


def parse_performance_report(path: Path) -> dict[str, float]:
    """Parse GEAK's optional JSON performance report."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}

    values: dict[str, float] = {}
    for index, item in enumerate(payload.get("test_cases") or []):
        if not isinstance(item, dict):
            continue
        latency = item.get("execution_time_ms")
        if not isinstance(latency, (int, float)) or latency <= 0:
            continue
        name = str(item.get("test_case_id") or "case_%d" % index)
        values[name] = float(latency)
    return values


def median_cases(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    common = set(samples[0])
    for sample in samples[1:]:
        common &= set(sample)
    return {
        name: statistics.median(float(sample[name]) for sample in samples)
        for name in sorted(common)
    }


def geomean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))
