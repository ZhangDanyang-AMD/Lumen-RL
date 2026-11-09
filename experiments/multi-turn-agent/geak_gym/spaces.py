"""Action and observation space definitions for the GEAK gym environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOOL_NAMES = ("list_files", "read_file", "write_file", "evaluate")

EVALUATE_MODES = ("compile", "correctness", "performance", "full")


@dataclass
class GEAKAction:
    """Parsed agent action."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_sandbox_params(self) -> dict[str, Any]:
        """Convert to ``GEAKToolEnvironment.execute`` parameter dict."""
        params: dict[str, Any] = {"action": self.tool_name}
        params.update(self.arguments)
        return params


@dataclass
class GEAKObservation:
    """Structured observation returned to the agent."""

    tool_result: dict[str, Any]
    allowed_write_paths: list[str]
    baseline_ms: dict[str, float]
    current_speedup: float
    turn_index: int
    done: bool

    def to_text(self, max_length: int = 8000) -> str:
        """Flatten to text for LLM context."""
        import json

        text = json.dumps(
            {
                "result": self.tool_result,
                "speedup": self.current_speedup,
                "turn": self.turn_index,
                "done": self.done,
            },
            default=str,
        )
        if len(text) > max_length:
            half = max_length // 2
            text = text[:half] + "...(truncated)..." + text[-half:]
        return text


def parse_action(raw: str | dict[str, Any]) -> GEAKAction:
    """Parse a raw action (JSON dict or text) into a :class:`GEAKAction`."""
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return GEAKAction(tool_name="evaluate", arguments={"mode": "full"})
    if not isinstance(raw, dict):
        return GEAKAction(tool_name="evaluate", arguments={"mode": "full"})
    tool_name = str(raw.get("tool_name") or raw.get("action") or raw.get("name") or "evaluate")
    arguments = dict(raw.get("arguments") or raw.get("params") or {})
    if tool_name not in TOOL_NAMES:
        tool_name = "evaluate"
        arguments = {"mode": "full"}
    return GEAKAction(tool_name=tool_name, arguments=arguments)
