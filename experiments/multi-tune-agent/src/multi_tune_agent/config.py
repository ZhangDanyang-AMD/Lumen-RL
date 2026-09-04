"""Configuration for the MultiTune orchestration and GEAK environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class MultiTuneConfig:
    geak_root: Path
    cases_path: Path
    trajectory_root: Path
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-Coder-Next"
    gpu_ids: str = "1"
    request_timeout: float = 600.0
    command_timeout: int = 300
    baseline_repeats: int = 3
    max_rounds: int = 3
    engineers_per_round: int = 3
    engineer_tool_rounds: int = 16
    integrator_tool_rounds: int = 10
    candidate_floor: float = 1.0
    min_improvement: float = 0.02
    target_speedup: float = 1.5
    keep_sessions: bool = True
    bootstrap_enabled: bool = True
    bootstrap_auto_promote: bool = True
    aiter_root: Path = Path("~/aiter")
    generated_template_root: Path = (
        Path(__file__).resolve().parents[2] / "examples" / "tasks" / "generated"
    )
    bootstrap_min_aiter_score: int = 85

    def __post_init__(self) -> None:
        object.__setattr__(self, "geak_root", self.geak_root.expanduser().resolve())
        object.__setattr__(self, "cases_path", self.cases_path.expanduser().resolve())
        object.__setattr__(
            self, "trajectory_root", self.trajectory_root.expanduser().resolve()
        )
        object.__setattr__(self, "aiter_root", Path(self.aiter_root).expanduser().resolve())
        object.__setattr__(
            self,
            "generated_template_root",
            Path(self.generated_template_root).expanduser().resolve(),
        )
        if self.max_rounds < 1 or self.engineers_per_round < 1:
            raise ValueError("max_rounds and engineers_per_round must be positive")
        if self.engineer_tool_rounds < 1 or self.integrator_tool_rounds < 1:
            raise ValueError("tool round limits must be positive")
        if self.baseline_repeats < 1:
            raise ValueError("baseline_repeats must be positive")
        if self.candidate_floor <= 0 or self.target_speedup <= 0:
            raise ValueError("speedup thresholds must be positive")
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative")
        for name in ("bootstrap_enabled", "bootstrap_auto_promote"):
            if isinstance(getattr(self, name), bool) is False:
                raise ValueError("%s must be a boolean" % name)
        if isinstance(self.bootstrap_min_aiter_score, bool):
            raise ValueError("bootstrap_min_aiter_score must be an integer")
        try:
            score = int(self.bootstrap_min_aiter_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("bootstrap_min_aiter_score must be an integer") from exc
        if score != self.bootstrap_min_aiter_score:
            raise ValueError("bootstrap_min_aiter_score must be an integer")
        if not 0 <= score <= 100:
            raise ValueError("bootstrap_min_aiter_score must be between 0 and 100")
        object.__setattr__(self, "bootstrap_min_aiter_score", score)

    @classmethod
    def from_yaml(cls, path: Path) -> "MultiTuneConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("configuration root must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown configuration keys: %s" % ", ".join(unknown))
        values: dict[str, Any] = dict(payload)
        env_base_url = os.environ.get("LUMEN_CODE_BASE_URL", "").strip()
        if env_base_url:
            values["base_url"] = env_base_url
        env_geak_root = (
            os.environ.get("LUMEN_CODE_GEAK_ROOT", "").strip()
            or os.environ.get("GEAK_HOME", "").strip()
        )
        if env_geak_root:
            values["geak_root"] = env_geak_root
        env_aiter_root = os.environ.get("AITER_HOME", "").strip()
        if env_aiter_root:
            values["aiter_root"] = env_aiter_root
        values.setdefault(
            "generated_template_root",
            Path(__file__).resolve().parents[2] / "examples" / "tasks" / "generated",
        )
        values.setdefault("aiter_root", Path("~/aiter"))
        for name in (
            "geak_root",
            "cases_path",
            "trajectory_root",
            "aiter_root",
            "generated_template_root",
        ):
            if name not in values:
                raise ValueError("missing required configuration key: %s" % name)
            raw = Path(str(values[name])).expanduser()
            if not raw.is_absolute():
                raw = config_path.parent / raw
            values[name] = raw
        return cls(**values)

