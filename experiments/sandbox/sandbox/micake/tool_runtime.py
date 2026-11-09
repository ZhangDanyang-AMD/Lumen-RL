"""Model-visible MICake tools and fail-closed argument parsing.

The model may only submit a typed schedule/realization from MICake's closed
domain plus a free-text hypothesis. It cannot submit source code, paths,
compiler flags, correctness verdicts or promotion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SUBMIT_TOOL = "submit_schedule"
FINISH_TOOL = "finish"
HIDDEN_FIXED_FIELDS = {"realization": ("kpack",)}
HYPOTHESIS_LIMIT = 600


@dataclass(frozen=True)
class Submission:
    """Validated submit_schedule arguments."""

    config: dict[str, Any]
    hypothesis: str


class ToolArgumentError(ValueError):
    """Arguments violate the tool schema."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _visible_fields(domain: Mapping[str, Mapping[str, list[Any]]], section: str) -> dict[str, list[Any]]:
    hidden = HIDDEN_FIXED_FIELDS.get(section, ())
    return {name: values for name, values in domain[section].items() if name not in hidden}


def tool_schemas(domain: Mapping[str, Mapping[str, list[Any]]]) -> list[dict[str, Any]]:
    """OpenAI function schemas derived from MICake's candidate domain."""

    def section_schema(section: str) -> dict[str, Any]:
        fields = _visible_fields(domain, section)
        return {
            "type": "object",
            "properties": {
                name: {"type": "string" if isinstance(values[0], str) else "integer", "enum": values}
                for name, values in fields.items()
            },
            "required": sorted(fields),
            "additionalProperties": False,
        }

    return [
        {
            "type": "function",
            "function": {
                "name": SUBMIT_TOOL,
                "description": (
                    "Submit one typed GEMM schedule candidate. MICake validates it, materializes Triton "
                    "source, compiles for gfx950, checks correctness on all workloads and (if enabled) "
                    "measures it against the baseline. Returns compiler facts, correctness and speedup."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schedule": section_schema("schedule"),
                        "realization": section_schema("realization"),
                        "hypothesis": {
                            "type": "string",
                            "description": "What you expect this change to do and why (one or two sentences).",
                        },
                    },
                    "required": ["schedule", "realization", "hypothesis"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": FINISH_TOOL,
                "description": "End the optimization task when you believe no further submission will help.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "description": "Short summary of findings."}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_submission(
    arguments: Mapping[str, Any] | None,
    domain: Mapping[str, Mapping[str, list[Any]]],
) -> Submission:
    """Validate submit_schedule arguments against the closed domain.

    Raises:
        ToolArgumentError: On any schema violation.
    """
    if arguments is None:
        raise ToolArgumentError("ARGUMENTS_NOT_JSON_OBJECT", "tool arguments must be a JSON object")
    extra = set(arguments) - {"schedule", "realization", "hypothesis"}
    if extra:
        raise ToolArgumentError("UNEXPECTED_FIELD", ", ".join(sorted(extra)))
    hypothesis = arguments.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ToolArgumentError("HYPOTHESIS_REQUIRED", "hypothesis must be a non-empty string")

    config: dict[str, Any] = {}
    for section in ("schedule", "realization"):
        value = arguments.get(section)
        if not isinstance(value, Mapping):
            raise ToolArgumentError("SECTION_REQUIRED", f"{section} must be an object")
        fields = _visible_fields(domain, section)
        unknown = set(value) - set(fields)
        missing = set(fields) - set(value)
        if unknown or missing:
            raise ToolArgumentError(
                "SECTION_FIELDS_INVALID",
                f"{section}: unknown={sorted(unknown)} missing={sorted(missing)}",
            )
        for name, allowed in fields.items():
            if type(value[name]) is not type(allowed[0]) or value[name] not in allowed:
                raise ToolArgumentError("VALUE_OUT_OF_DOMAIN", f"{section}.{name}={value[name]!r}; allowed {allowed}")
        config[section] = dict(value)
        for name in HIDDEN_FIXED_FIELDS.get(section, ()):
            config[section][name] = domain[section][name][0]
    return Submission(config=config, hypothesis=hypothesis.strip()[:HYPOTHESIS_LIMIT])
