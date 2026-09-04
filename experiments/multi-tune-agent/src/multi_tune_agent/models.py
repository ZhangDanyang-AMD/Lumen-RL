"""Shared value objects for MultiTune sessions and role handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RewardBreakdown:
    total: float
    correctness: float
    performance: float
    improvement: float
    speedup: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "correctness": self.correctness,
            "performance": self.performance,
            "improvement": self.improvement,
            "speedup": self.speedup,
        }


@dataclass
class SessionState:
    session_id: str
    case_id: str
    role: str
    episode_dir: Path
    workspace: Path
    sandbox: Any
    parent_session_id: Optional[str] = None
    best_speedup: float = 1.0
    last_evaluation: Optional[dict[str, Any]] = None
    last_reward: float = 0.0
    closed: bool = False


@dataclass
class Direction:
    direction_id: str
    specialty: str
    strategy: str
    instructions: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any], index: int) -> "Direction":
        # GEAK's native TechLead schema uses id/title/prompt, while MultiTune's
        # internal handoff uses direction_id/strategy/instructions.
        direction_id = value.get("direction_id") or value.get("id")
        strategy = (
            value.get("strategy")
            or value.get("title")
            or value.get("instructions")
            or value.get("prompt")
        )
        instructions = (
            value.get("instructions")
            or value.get("prompt")
            or value.get("strategy")
            or value.get("title")
        )
        return cls(
            direction_id=str(direction_id or "direction_%d" % (index + 1)),
            specialty=str(value.get("specialty") or "algorithm"),
            strategy=str(strategy or ""),
            instructions=str(instructions or ""),
        )


@dataclass
class Candidate:
    candidate_id: str
    session_id: str
    direction: Direction
    evaluation: dict[str, Any]
    reward: RewardBreakdown
    accepted: bool
    agent_text: str = ""
    verifier: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        return float(self.evaluation.get("speedup_geomean") or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "direction": self.direction.__dict__,
            "evaluation": self.evaluation,
            "reward": self.reward.to_dict(),
            "accepted": self.accepted,
            "agent_text": self.agent_text,
            "verifier": self.verifier,
        }

