"""Role agents backed by the internal agent-loop runtime."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .geak_tool import GEAKStatefulTool, GEAKToolEnvironment
from .models import Direction
from .runtime import AgentLoopOutput, ModelBackend, ToolAgentLoop
from .trajectory import TrajectoryWriter


_ROLE_FILES = {
    "director": "director.md",
    "tech_lead": "tech_lead.md",
    "engineer": "engineer.md",
    "verifier": "verify_engineer.md",
    "integrator": "integrator.md",
}

_CASE_KNOWLEDGE = {
    "gemm": ("dense_gemm",),
    "fused_attention": ("attention_prefill_fmha",),
    "grouped_gemm": ("grouped_gemm_moe", "fused_moe_grouped_gemm"),
    "scaled_quant_gemm": ("scaled_quant_gemm",),
    "quant_fp4_mxfp": ("quant_fp4_mxfp",),
    "aiter_generated": (),
}


class RolePromptLibrary:
    def __init__(self, geak_root: Path) -> None:
        self.geak_root = Path(geak_root).resolve()

    def system(self, role: str, case_type: str = "") -> str:
        try:
            role_file = _ROLE_FILES[role]
        except KeyError as exc:
            raise ValueError("unsupported role: %s" % role) from exc
        role_text = (
            self.geak_root / "kernel_workflow" / "roles" / role_file
        ).read_text(encoding="utf-8")
        contract = (
            "You run inside MultiTune. GEAK is exposed as one stateful tool. "
            "Never edit tests, task configuration, metadata, or oracle files. "
            "Correctness is a hard gate and all speedups are measured against a "
            "frozen baseline. Do not claim unmeasured improvements."
        )
        if role not in {"engineer", "integrator"} or not case_type:
            return contract + "\n\n## GEAK role\n" + role_text
        knowledge = self._knowledge(case_type)
        return (
            contract
            + "\n\n## GEAK role\n"
            + role_text
            + "\n\n## Selected performance knowledge\n"
            + knowledge
        )

    def _knowledge(self, case_type: str) -> str:
        sections = []
        workflow = self.geak_root / "kernel_workflow" / "knowledge"
        for name in ("self_monitoring.md", "optimization_strategies.md", "amd_instinct.md"):
            path = workflow / name
            if path.is_file():
                sections.append("### %s\n%s" % (name, path.read_text(encoding="utf-8")))
        perf = self.geak_root / "perf_knowledge" / "operators"
        for operator in _CASE_KNOWLEDGE.get(case_type, ()):
            for name in ("overview.md", "tuning.md", "numerics.md"):
                path = perf / operator / name
                if path.is_file():
                    sections.append(
                        "### %s/%s\n%s"
                        % (operator, name, path.read_text(encoding="utf-8"))
                    )
        return "\n\n".join(sections)


class StructuredRoleAgent:
    """Read-only role invocation with resilient JSON extraction."""

    def __init__(
        self,
        backend: ModelBackend,
        prompts: RolePromptLibrary,
        trajectory: TrajectoryWriter,
    ) -> None:
        self.backend = backend
        self.prompts = prompts
        self.trajectory = trajectory
        self._case_input: dict[str, Any] | None = None

    async def run(
        self,
        role: str,
        phase: str,
        payload: Mapping[str, Any],
        *,
        case_type: str = "",
        round_index: int = 0,
    ) -> dict[str, Any]:
        effective_payload = dict(payload)
        supplied_case = effective_payload.get("case")
        if isinstance(supplied_case, Mapping):
            self._case_input = dict(supplied_case)
        elif case_type == "aiter_generated" and self._case_input is not None:
            effective_payload["case"] = dict(self._case_input)
        messages = [
            {
                "role": "system",
                "content": self.prompts.system(role, case_type)
                + "\nReturn one JSON object only. Do not call tools.",
            },
            {
                "role": "user",
                "content": "PHASE=%s\nINPUT=%s"
                % (
                    phase,
                    json.dumps(
                        effective_payload, indent=2, sort_keys=True, default=str
                    ),
                ),
            },
        ]
        started = time.monotonic()
        turn = await asyncio.to_thread(self.backend.generate, messages, ())
        elapsed = time.monotonic() - started
        parsed = extract_json(turn.text)
        self.trajectory.append(
            "role_response",
            {
                "input": effective_payload,
                "text": turn.text,
                "structured": parsed,
                "usage": turn.usage,
                "logprobs": turn.logprobs,
                "elapsed_seconds": elapsed,
            },
            role=role,
            phase=phase,
            round_index=round_index,
        )
        return parsed


class CodeRoleAgent:
    """Engineer or Integrator with a private long-lived GEAK tool session."""

    def __init__(
        self,
        backend: ModelBackend,
        prompts: RolePromptLibrary,
        environment: GEAKToolEnvironment,
        trajectory: TrajectoryWriter,
    ) -> None:
        self.backend = backend
        self.prompts = prompts
        self.environment = environment
        self.trajectory = trajectory

    async def engineer(
        self,
        case_id: str,
        case_type: str,
        parent_session_id: str,
        direction: Direction,
        *,
        max_turns: int,
        round_index: int,
    ) -> AgentLoopOutput:
        messages = [
            {"role": "system", "content": self.prompts.system("engineer", case_type)},
            {
                "role": "user",
                "content": (
                    "Implement and measure this assigned direction in your private "
                    "workspace. Use the GEAK tool; finish only after correctness and "
                    "performance evidence. Preserve every field in the generated "
                    "kernel contract when one is supplied.\n\nCASE=\n"
                    + json.dumps(
                        self.environment.case_observation(case_id),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n\nDIRECTION=\n"
                    + json.dumps(direction.__dict__, indent=2, sort_keys=True)
                ),
            },
        ]
        output = await ToolAgentLoop(
            self.backend,
            GEAKStatefulTool(self.environment),
            max_assistant_turns=max_turns,
            retain_session=True,
        ).run(
            messages,
            create_kwargs={
                "case_id": case_id,
                "role": "engineer",
                "parent_session_id": parent_session_id,
            },
        )
        self._record_loop(output, "engineer", "optimize", round_index, direction.direction_id)
        return output

    async def integrator(
        self,
        case_id: str,
        case_type: str,
        parent_session_id: str,
        candidates: list[dict[str, Any]],
        *,
        max_turns: int,
        round_index: int,
    ) -> AgentLoopOutput:
        visible = [
            {
                "candidate_id": item["candidate_id"],
                "session_id": item["session_id"],
                "direction": item["direction"],
                "speedup": item["evaluation"].get("speedup_geomean"),
                "candidate_ms": item["evaluation"].get("candidate_ms"),
            }
            for item in candidates
        ]
        messages = [
            {"role": "system", "content": self.prompts.system("integrator", case_type)},
            {
                "role": "user",
                "content": (
                    "Combine only compatible measured wins. Use read_candidate with "
                    "the listed session ids to inspect their source, edit your private "
                    "workspace, and run a full evaluation. If merging is unsafe, copy "
                    "the strongest single strategy instead. Preserve every field in "
                    "the generated kernel contract when one is supplied.\n\nCASE=\n"
                    + json.dumps(
                        self.environment.case_observation(case_id),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n\nCANDIDATES=\n"
                    + json.dumps(visible, indent=2, sort_keys=True)
                ),
            },
        ]
        output = await ToolAgentLoop(
            self.backend,
            GEAKStatefulTool(self.environment),
            max_assistant_turns=max_turns,
            retain_session=True,
        ).run(
            messages,
            create_kwargs={
                "case_id": case_id,
                "role": "integrator",
                "parent_session_id": parent_session_id,
            },
        )
        self._record_loop(output, "integrator", "integrate", round_index, "integrated")
        return output

    def _record_loop(
        self,
        output: AgentLoopOutput,
        role: str,
        phase: str,
        round_index: int,
        direction_id: str,
    ) -> None:
        self.trajectory.append(
            "agent_loop",
            {
                "direction_id": direction_id,
                "session_id": output.session_id,
                "final_text": output.final_text,
                "policy_logprobs": output.policy_logprobs,
                "tool_rewards": output.tool_rewards,
                "reward_score": output.reward_score,
                "metrics": output.metrics.__dict__,
                "events": output.events,
            },
            role=role,
            phase=phase,
            round_index=round_index,
        )


def extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    candidates = [value]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return {"_parse_error": "role did not return valid JSON", "_raw": text}

