"""Framework-independent long-horizon agent and tool runtime.

The lifecycle and state machine mirror the useful ideas from verl/Uni-Agent,
but this module has no dependency on either project.
"""

from __future__ import annotations

import abc
import asyncio
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence

import requests


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    assistant_message: dict[str, Any]
    tool_calls: list[ToolCall]
    logprobs: list[float]
    usage: dict[str, Any]


class ModelBackend(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelTurn: ...


def _messages_for_generation(
    messages: Sequence[Mapping[str, Any]], max_chars: int = 80000
) -> list[dict[str, Any]]:
    """Compact replayable tool history while retaining recent evidence.

    Strategy: keep first 3 messages (system + user + env init) and the
    most recent ``tail_messages`` messages verbatim.  Everything in
    between is replaced by a single summary.  Within the retained tail,
    large write_file contents and tool observations are truncated.
    """
    tail_messages = 20

    prepared = [copy.deepcopy(dict(message)) for message in messages]
    # Truncate write_file content and tool observations everywhere
    for index, message in enumerate(prepared):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = function.get("arguments")
                if not isinstance(raw, str):
                    continue
                try:
                    arguments = json.loads(raw)
                except ValueError:
                    continue
                content = arguments.get("content") if isinstance(arguments, dict) else None
                if (
                    isinstance(arguments, dict)
                    and arguments.get("action") == "write_file"
                    and isinstance(content, str)
                    and len(content) > 512
                ):
                    arguments["content"] = (
                        "<omitted %d chars; use read_file for current source>"
                        % len(content)
                    )
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str) and len(content) > 2000:
                message["content"] = (
                    content[:1000]
                    + "\n...<truncated>...\n"
                    + content[-1000:]
                )

    total = json.dumps(prepared, default=str, ensure_ascii=False)
    if len(total) <= max_chars:
        return prepared

    prefix_count = min(3, len(prepared))
    tail_start = max(prefix_count, len(prepared) - tail_messages)
    while tail_start > prefix_count and prepared[tail_start].get("role") == "tool":
        tail_start -= 1
    omitted = prepared[prefix_count:tail_start]
    summary = {
        "role": "user",
        "content": (
            "[Context compacted] %d older messages omitted. "
            "Use read_file to inspect the current workspace state. "
            "Continue from the latest visible feedback."
            % len(omitted)
        ),
    }
    result = prepared[:prefix_count] + [summary] + prepared[tail_start:]
    # If still over budget, drop more aggressively
    while (
        len(json.dumps(result, default=str, ensure_ascii=False)) > max_chars
        and len(result) > prefix_count + 4
    ):
        result = result[:prefix_count] + [summary] + result[prefix_count + 2:]
    return result


def _detect_kernel_code(text: str, language: str = "triton") -> str | None:
    """Detect if model text output contains kernel code and extract it.

    SFT models often output code as text instead of calling write_file.
    This function detects such output and returns the code to auto-write.
    """
    text = text.strip()
    if len(text) < 50:
        return None

    # Check for code indicators by language
    triton_markers = ["@triton.jit", "@triton.autotune", "import triton", "tl.load", "tl.store"]
    opus_markers = ["__global__", "#include", "hip_runtime", "hipLaunchKernelGGL"]
    flydsl_markers = ["@flyc.kernel", "@flyc.jit", "import flydsl", "fx.Tensor"]

    markers = {
        "triton": triton_markers,
        "opus": opus_markers,
        "flydsl": flydsl_markers,
    }.get(language, triton_markers + opus_markers + flydsl_markers)

    hit_count = sum(1 for m in markers if m in text)
    if hit_count < 1:
        return None

    # Extract code: if text has markdown code blocks, extract them
    code_block_match = re.findall(r"```(?:python|cpp|cuda|hip)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        # Join all code blocks
        code = "\n\n".join(code_block_match)
        if len(code) > 50:
            return code

    # If no code blocks, check if the text IS code (starts with import/include/def/etc.)
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) > 5:
        code_kws = ("import ", "from ", "def ", "#include", "#", "@", "class ",
                     "    ", "\t", "//", "/*", "__", "void ", "int ", "float ",
                     "return ", "for ", "if ", "while ", "elif ", "else:")
        code_indicators = sum(1 for l in non_empty[:40] if any(
            l.strip().startswith(kw) or l.startswith(kw) for kw in code_kws
        ))
        if code_indicators > len(non_empty[:40]) * 0.3:
            # Strip leading prose lines (e.g., "Here is the kernel:")
            start = 0
            for i, l in enumerate(lines):
                s = l.strip()
                if s and any(s.startswith(kw) for kw in ("import ", "from ", "#include", "@", "def ", "class ", "__", "void ", "int ")):
                    start = i
                    break
                if i > 5:
                    break
            return "\n".join(lines[start:]).strip()

    return None


def _parse_text_tool_calls(text: str) -> list["ToolCall"]:
    """Best-effort parse of tool calls that vLLM returned as text content.

    Handles three formats the SFT model produces:
    1. JSON array/object: [{"name":"geak","arguments":{...}}]
    2. XML: <tool_call><function=geak><parameter=action=read_file>...
    3. Truncated JSON from max_tokens cutoff
    """
    parsed: list["ToolCall"] = []
    text = text.strip()

    # --- Format 1: JSON (possibly truncated) ---
    if text.startswith(("{", "[")):
        # First try direct parse
        try:
            maybe = json.loads(text)
            if isinstance(maybe, dict):
                maybe = [maybe]
            if isinstance(maybe, list):
                for item in maybe:
                    if not isinstance(item, dict):
                        continue
                    # Handle two formats:
                    # A) Flat: {"name": "geak", "arguments": {...}}
                    # B) OpenAI nested: {"function": {"name": "geak", "arguments": "{...}"}, "id": "..."}
                    _fn = item.get("function")
                    if isinstance(_fn, dict) and "name" in _fn:
                        name = str(_fn["name"])
                        raw_args = _fn.get("arguments", {})
                        _call_id = str(item.get("id", uuid.uuid4().hex))
                    elif "name" in item:
                        name = str(item["name"])
                        raw_args = item.get("arguments", {})
                        _call_id = str(item.get("id", uuid.uuid4().hex))
                    else:
                        continue
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except (ValueError, TypeError):
                            raw_args = {"_raw": raw_args}
                    if isinstance(raw_args, dict):
                        parsed.append(ToolCall(
                            call_id=_call_id,
                            name=name,
                            arguments=raw_args,
                        ))
        except (ValueError, TypeError, KeyError):
            # JSON is truncated — try to extract what we can.
            # Strategy: find the "name" and "action"/"path" fields via regex,
            # then extract the "content" as everything after "content":"
            _name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            _action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text)
            _path_match = re.search(r'"path"\s*:\s*"([^"]+)"', text)
            if _name_match:
                _tc_name = _name_match.group(1)
                _tc_args: dict[str, Any] = {}
                if _action_match:
                    _tc_args["action"] = _action_match.group(1)
                if _path_match:
                    _tc_args["path"] = _path_match.group(1)
                # Extract content field (everything after "content":" until end)
                _content_match = re.search(r'"content"\s*:\s*"', text)
                if _content_match:
                    _content_start = _content_match.end()
                    _raw_content = text[_content_start:]
                    # Remove trailing incomplete JSON chars
                    _raw_content = _raw_content.rstrip("}] \n\r\t")
                    if _raw_content.endswith('"'):
                        _raw_content = _raw_content[:-1]
                    # Unescape JSON string escapes
                    try:
                        _tc_args["content"] = _raw_content.encode().decode("unicode_escape")
                    except (UnicodeDecodeError, ValueError):
                        _tc_args["content"] = _raw_content.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
                _id_match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
                if _tc_args:
                    parsed.append(ToolCall(
                        call_id=_id_match.group(1) if _id_match else uuid.uuid4().hex,
                        name=_tc_name,
                        arguments=_tc_args,
                    ))

    # --- Format 2: XML <tool_call> format (Qwen3-Coder alternative) ---
    if not parsed and "<tool_call>" in text:
        _tool_call_pattern = re.compile(
            r"<function=(\w+)>(.*?)</function>", re.DOTALL
        )
        _param_pattern = re.compile(
            r"<parameter=(\w+)(?:=|\s*>\s*\n?)(.*?)\s*\n?\s*</parameter>", re.DOTALL
        )
        for fn_match in _tool_call_pattern.finditer(text):
            fn_name = fn_match.group(1)
            fn_body = fn_match.group(2)
            args = {}
            for p_match in _param_pattern.finditer(fn_body):
                args[p_match.group(1)] = p_match.group(2).strip()
            if fn_name and args:
                parsed.append(ToolCall(
                    call_id=uuid.uuid4().hex,
                    name=fn_name,
                    arguments=args,
                ))

    return parsed


class OpenAIModelBackend:
    """Small OpenAI-compatible backend with tool calls and token logprobs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "unused",
        timeout: float = 600.0,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.repetition_penalty = float(repetition_penalty)
        self.frequency_penalty = float(frequency_penalty)
        self.session = session or requests.Session()

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_generation(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self.frequency_penalty
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        response = self.session.post(
            self.base_url + "/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer " + self.api_key},
            timeout=self.timeout,
        )
        if response.status_code == 400:
            detail = response.text
            match = re.search(
                r"maximum context length is (\d+) tokens.*request has (\d+) input tokens",
                detail,
                re.IGNORECASE,
            )
            if match:
                available = int(match.group(1)) - int(match.group(2)) - 64
                retry_tokens = min(self.max_tokens, available)
                if 128 <= retry_tokens < self.max_tokens:
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = retry_tokens
                    response = self.session.post(
                        self.base_url + "/chat/completions",
                        json=retry_payload,
                        headers={"Authorization": "Bearer " + self.api_key},
                        timeout=self.timeout,
                    )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if len(detail) > 2000:
                detail = detail[:2000] + "...(truncated)"
            raise requests.HTTPError(
                "%s; response=%s" % (exc, detail), response=response
            ) from exc
        body = response.json()
        # Detect vLLM "Operation not allowed" or llm_call_failed errors
        if "error" in body and "choices" not in body:
            err_msg = str(body.get("error", body))[:300]
            raise RuntimeError(f"vLLM error: {err_msg}")
        choice = body["choices"][0]
        message = dict(choice.get("message") or {})
        # Detect error wrapped as content (vLLM concurrency rejection)
        _raw_content = str(message.get("content") or "")
        if '"llm_call_failed"' in _raw_content or '"Operation not allowed"' in _raw_content:
            raise RuntimeError(f"vLLM rejected request: {_raw_content[:200]}")
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must be an object")
            except (TypeError, ValueError) as exc:
                parsed = {"_malformed_json": str(exc), "_raw": arguments}
            calls.append(
                ToolCall(
                    call_id=str(raw.get("id") or uuid.uuid4().hex),
                    name=str(function.get("name") or ""),
                    arguments=parsed,
                )
            )

        # Fallback: if vLLM returned tool call as text (qwen3_coder parser miss),
        # parse it manually from the content field.
        content_text = str(message.get("content") or "")
        if not calls and content_text:
            _parsed_calls = _parse_text_tool_calls(content_text)
            if _parsed_calls:
                calls.extend(_parsed_calls)
                content_text = ""

        logprobs = []
        for token in (choice.get("logprobs") or {}).get("content") or []:
            value = token.get("logprob")
            if isinstance(value, (int, float)):
                logprobs.append(float(value))
        return ModelTurn(
            text=content_text,
            assistant_message=message,
            tool_calls=calls,
            logprobs=logprobs,
            usage=dict(body.get("usage") or {}),
        )


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass
class AgentLoopMetrics:
    total_seconds: float = 0.0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    assistant_turns: int = 0
    tool_calls: int = 0


@dataclass
class AgentLoopOutput:
    messages: list[dict[str, Any]]
    final_text: str
    policy_logprobs: list[float]
    tool_rewards: list[float]
    reward_score: float
    metrics: AgentLoopMetrics
    session_id: str
    events: list[dict[str, Any]] = field(default_factory=list)


class StatefulTool(abc.ABC):
    name: str

    @abc.abstractmethod
    def schemas(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def create(self, create_kwargs: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]: ...

    @abc.abstractmethod
    def execute(
        self, instance_id: str, parameters: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, Mapping[str, Any]]: ...

    @abc.abstractmethod
    def calc_reward(self, instance_id: str) -> float: ...

    @abc.abstractmethod
    def release(self, instance_id: str) -> None: ...


class AgentLoopBase(abc.ABC):
    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> AgentLoopOutput: ...


class ToolAgentLoop(AgentLoopBase):
    """One policy with one long-lived tool session across all turns."""

    def __init__(
        self,
        backend: ModelBackend,
        tool: StatefulTool,
        *,
        max_assistant_turns: int = 16,
        retain_session: bool = True,
    ) -> None:
        self.backend = backend
        self.tool = tool
        self.max_assistant_turns = int(max_assistant_turns)
        self.retain_session = bool(retain_session)

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        create_kwargs: Mapping[str, Any],
    ) -> AgentLoopOutput:
        started = time.monotonic()
        session_id, initial = await asyncio.to_thread(self.tool.create, create_kwargs)
        history = [dict(message) for message in messages]
        # Summarize initial state without raw JSON to avoid model hallucinating tool results
        workspace = initial.get("workspace", "")
        allowed = initial.get("allowed_write_paths", [])
        baseline_geo = initial.get("baseline", {}).get("geomean_ms", 0.0)
        case_id = initial.get("case_id", "")

        # Extract operator/language/shape from workspace config.yaml
        task_context = ""
        _task_lang = "triton"
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            cfg_path = _Path(workspace) / "config.yaml"
            if cfg_path.exists():
                cfg = _yaml.safe_load(cfg_path.read_text())
                op = cfg.get("operator", "unknown")
                lang = cfg.get("language", "triton")
                _task_lang = lang
                shapes = cfg.get("benchmark_shapes", [])
                shape_str = ""
                if shapes and isinstance(shapes[0], dict):
                    s = shapes[0]
                    shape_parts = [f"{k}={v}" for k, v in s.items()
                                   if k not in ("case_id", "dtype", "kernel_prompt")]
                    dtype = s.get("dtype", "bfloat16")
                    shape_str = f" | Shape: {', '.join(shape_parts)}, dtype={dtype}"
                    kp = s.get("kernel_prompt", "")
                    if kp:
                        task_context += f"Kernel description: {kp}\n"
                task_context += f"Operator: {op} | Language: {lang}{shape_str}\n"
        except Exception:
            pass

        api_hint = ""
        if lang == "opus":
            api_hint = (
                "\nCRITICAL: Before writing kernel.cu, run list_files('include/') to discover "
                "available headers. Then read_file the relevant ones for your operator. "
                "Do NOT invent headers — only #include files that actually exist in include/.\n"
            )
        elif lang == "flydsl":
            api_hint = (
                "\nCRITICAL: FlyDSL uses @flyc.kernel/@flyc.jit decorators and fx.* types. "
                "Real APIs: fx.Tensor, fx.make_tiled_mma, fx.gemm, fx.copy, fx.barrier, fx.block_idx. "
                "NONEXISTENT: fx.index(), fx.shared_tensor(), fx.block_barrier(). "
                "Define a build_*() or public wrapper function.\n"
            )

        history.append(
            {
                "role": "user",
                "content": (
                    f"Environment ready. Workspace: {workspace}\n"
                    f"Case: {case_id}\n"
                    f"{task_context}"
                    f"Writable files: {', '.join(allowed)}\n"
                    f"Baseline performance: {baseline_geo:.4f} ms (target: speedup >= 0.9x = {baseline_geo / 0.9:.4f} ms or faster)\n\n"
                    f"{api_hint}"
                    f"START NOW: call read_file('scripts/task_runner.py') to understand the correctness test, "
                    f"then read_file('config.yaml') to see shape details, "
                    f"then write your kernel, then evaluate.\n"
                    f"IMPORTANT: Your kernel must implement the '{task_context.split('|')[0].replace('Operator:', '').strip() if task_context else 'specified'}' operator. Do NOT write a different operator.\n"
                    f"When evaluate returns 'correct=false', read the 'correctness.stdout' field for the exact error and traceback."
                ),
            }
        )
        metrics = AgentLoopMetrics()
        policy_logprobs: list[float] = []
        tool_rewards: list[float] = []
        events: list[dict[str, Any]] = []
        final_text = ""
        state = AgentState.PENDING
        pending_calls: list[ToolCall] = []
        consecutive_no_tools = 0
        max_consecutive_no_tools = 10
        turns_since_write = 0
        has_ever_written = False

        try:
            while state is not AgentState.TERMINATED:
                if state is AgentState.PENDING:
                    state = AgentState.GENERATING
                    continue
                if state is AgentState.GENERATING:
                    model_started = time.monotonic()
                    try:
                        turn = await asyncio.to_thread(
                            self.backend.generate, history, self.tool.schemas()
                        )
                    except Exception as gen_err:
                        elapsed = time.monotonic() - model_started
                        err_str = str(gen_err)
                        # Retry transient vLLM errors (Operation not allowed, timeout)
                        # without counting as a turn
                        if "vLLM" in err_str or "Operation not allowed" in err_str or "llm_call_failed" in err_str:
                            import time as _time, sys as _sys
                            print(f"[GEN_RETRY] {err_str[:100]}", file=_sys.stderr, flush=True)
                            _time.sleep(2)
                            metrics.model_seconds += elapsed
                            continue  # retry without incrementing turns
                        metrics.model_seconds += elapsed
                        metrics.assistant_turns += 1
                        consecutive_no_tools += 1
                        if consecutive_no_tools >= max_consecutive_no_tools:
                            _err_last_sp = 0.0
                            for _e in reversed(events):
                                if _e.get("type") == "tool" and "evaluation" in _e.get("result", {}):
                                    _err_last_sp = float(_e["result"]["evaluation"].get("speedup_geomean", 0))
                                    break
                            if _err_last_sp > 0:
                                consecutive_no_tools = 0
                            else:
                                state = AgentState.TERMINATED
                        elif metrics.assistant_turns >= self.max_assistant_turns:
                            state = AgentState.TERMINATED
                        else:
                            history.append({"role": "user", "content": f"Generation error: {err_str[:200]}. Use the geak tool to continue."})
                        continue
                    elapsed = time.monotonic() - model_started
                    metrics.model_seconds += elapsed
                    metrics.assistant_turns += 1
                    policy_logprobs.extend(turn.logprobs)
                    final_text = turn.text
                    history.append(turn.assistant_message)
                    events.append(
                        {
                            "type": "model",
                            "elapsed_seconds": elapsed,
                            "text": turn.text,
                            "tool_calls": [call.__dict__ for call in turn.tool_calls],
                            "logprobs": turn.logprobs,
                            "usage": turn.usage,
                        }
                    )
                    pending_calls = turn.tool_calls
                    if not pending_calls and turn.text:
                        import sys as _sys
                        print(f"[NO_TOOL turn={metrics.assistant_turns}] text={repr(turn.text[:120])}", file=_sys.stderr, flush=True)
                        # Auto-capture: if model output contains kernel code, write it
                        _kernel_code = _detect_kernel_code(turn.text, _task_lang)
                        if _kernel_code and len(_kernel_code) > 80:
                            _write_path = "kernel.cu" if _task_lang == "opus" else "kernel.py"
                            try:
                                _aw_result, _aw_reward, _ = await asyncio.to_thread(
                                    self.tool.execute, session_id,
                                    {"action": "write_file", "path": _write_path, "content": _kernel_code},
                                )
                                if isinstance(_aw_result, dict) and _aw_result.get("ok"):
                                    has_ever_written = True
                                    turns_since_write = 0
                                    tool_rewards.append(float(_aw_reward))
                                    print(f"[AUTO_WRITE] Captured {len(_kernel_code)} chars to {_write_path}", file=_sys.stderr, flush=True)
                                    # Tell model we wrote it and to evaluate
                                    history.append({
                                        "role": "user",
                                        "content": (
                                            f"I automatically saved your kernel code to {_write_path}. "
                                            "Now call evaluate to test it:\n"
                                            '  geak(action="evaluate")'
                                        ),
                                    })
                                    consecutive_no_tools = 0
                                    state = AgentState.GENERATING
                                    continue
                            except Exception as _aw_err:
                                print(f"[AUTO_WRITE_ERR] {_aw_err}", file=_sys.stderr, flush=True)
                    if pending_calls:
                        consecutive_no_tools = 0
                        state = AgentState.PROCESSING_TOOLS
                    elif metrics.assistant_turns >= self.max_assistant_turns:
                        state = AgentState.TERMINATED
                    else:
                        # Check if latest evaluate achieved speedup >= 1.0
                        last_sp = 0.0
                        for e in reversed(events):
                            if e.get("type") == "tool" and "evaluation" in e.get("result", {}):
                                last_sp = float(e["result"]["evaluation"].get("speedup_geomean", 0))
                                break
                        if last_sp >= 1.0:
                            # Re-benchmark to confirm beating baseline
                            verify_result = None
                            try:
                                verify_result, _, _ = await asyncio.to_thread(
                                    self.tool.execute, session_id,
                                    {"action": "evaluate", "mode": "full"},
                                )
                            except Exception:
                                pass
                            verified_sp = float(
                                (verify_result or {}).get("evaluation", {}).get("speedup_geomean", 0)
                            ) if verify_result else 0
                            if verified_sp >= 1.0:
                                state = AgentState.TERMINATED
                            else:
                                history.append({
                                    "role": "user",
                                    "content": f"Re-benchmark: speedup={verified_sp:.2f}x — did not hold up. "
                                        "Target is >=1.0x (beat baseline). Continue optimizing.",
                                })
                                state = AgentState.GENERATING
                        else:
                            consecutive_no_tools += 1
                            if consecutive_no_tools >= max_consecutive_no_tools:
                                if last_sp > 0:
                                    # Model achieved some speedup but < 1.0 — don't terminate,
                                    # force it to keep optimizing until max_assistant_turns.
                                    consecutive_no_tools = 0
                                else:
                                    state = AgentState.TERMINATED
                                    continue
                            history.append({
                                "role": "user",
                                "content": (
                                    f"Current best speedup: {last_sp:.2f}x (target: >=1.0x). "
                                    "You MUST keep optimizing. Call the geak tool now.\n"
                                    "Try a different algorithm, tiling strategy, or memory access pattern.\n"
                                    '  geak(action="read_file", path="scripts/task_runner.py")\n'
                                    "Then rewrite kernel.py and call evaluate."
                                ),
                            })
                            state = AgentState.GENERATING
                    continue
                if state is AgentState.PROCESSING_TOOLS:
                    _GEAK_ACTIONS = {"list_files", "read_file", "read_candidate", "write_file", "evaluate", "state"}
                    for call in pending_calls:
                        tool_started = time.monotonic()
                        # Normalize tool calls: model may generate "read_file"
                        # or "geak.read_file" instead of "geak" with action param
                        _effective_name = call.name
                        _effective_args = dict(call.arguments)
                        if _effective_name != self.tool.name:
                            bare = _effective_name.removeprefix(self.tool.name + ".")
                            if bare in _GEAK_ACTIONS:
                                _effective_name = self.tool.name
                                _effective_args.setdefault("action", bare)
                        # Infer missing path for write_file based on language
                        if _effective_args.get("action") == "write_file" and "path" not in _effective_args:
                            _effective_args["path"] = "kernel.py"
                        call = ToolCall(call.call_id, _effective_name, _effective_args)
                        import sys as _sys
                        _act = _effective_args.get("action", call.name)
                        _pth = _effective_args.get("path", "")
                        print(f"[TOOL t={metrics.assistant_turns}] {_act} {_pth}", file=_sys.stderr, flush=True)

                        if call.name != self.tool.name:
                            result = {
                                "ok": False,
                                "error": "unknown tool %r; expected %r"
                                % (call.name, self.tool.name),
                            }
                            reward, tool_metrics = 0.0, {"unknown_tool": call.name}
                        else:
                            try:
                                result, reward, tool_metrics = await asyncio.to_thread(
                                    self.tool.execute,
                                    session_id,
                                    call.arguments,
                                )
                            except Exception as tool_err:
                                result = {"ok": False, "error": f"Tool execution error: {tool_err}"}
                                reward, tool_metrics = 0.0, {"tool_error": str(tool_err)}
                        elapsed = time.monotonic() - tool_started
                        metrics.tool_seconds += elapsed
                        metrics.tool_calls += 1
                        tool_rewards.append(float(reward))
                        content = json.dumps(result, sort_keys=True, default=str)
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "name": call.name,
                                "content": content,
                            }
                        )
                        events.append(
                            {
                                "type": "tool",
                                "name": call.name,
                                "elapsed_seconds": elapsed,
                                "parameters": call.arguments,
                                "result": result,
                                "reward": reward,
                                "metrics": dict(tool_metrics),
                            }
                        )
                        # Track write_file calls
                        _action = call.arguments.get("action", "") if isinstance(call.arguments, dict) else ""
                        if _action == "write_file" and isinstance(result, dict) and result.get("ok"):
                            has_ever_written = True
                            turns_since_write = 0
                            # Repetition detection on written content
                            _written = str(call.arguments.get("content", ""))
                            _lines = _written.strip().split("\n")
                            if len(_lines) > 20:
                                from collections import Counter as _Counter
                                _counts = _Counter(_lines)
                                _top_count = _counts.most_common(1)[0][1] if _counts else 0
                                if _top_count > len(_lines) * 0.3:
                                    history.append({
                                        "role": "user",
                                        "content": (
                                            "WARNING: Your kernel contains highly repetitive lines "
                                            f"({_top_count}/{len(_lines)} lines are identical). "
                                            "This is a generation error. Rewrite the kernel with "
                                            "actual implementation logic, not repeated comments."
                                        ),
                                    })

                        eval_data = result.get("evaluation", {}) if isinstance(result, dict) else {}
                        sp = float(eval_data.get("speedup_geomean", 0))
                        if sp >= 0.9 and eval_data.get("correct"):
                            state = AgentState.TERMINATED
                            break

                        # Post-evaluate nudge: if evaluation failed, tell model to fix
                        if _action == "evaluate" and isinstance(result, dict):
                            _compiled = eval_data.get("compiled", False)
                            _correct = eval_data.get("correct", False)
                            if not _compiled or not _correct:
                                _err_msg = ""
                                _stdout = str(eval_data.get("correctness", {}).get("stdout", ""))
                                _compile_out = str(eval_data.get("compile", {}).get("stdout", ""))
                                if not _compiled and _compile_out:
                                    _err_msg = f"Compile error:\n{_compile_out[:500]}"
                                elif not _correct and _stdout:
                                    _err_msg = f"Correctness error:\n{_stdout[:500]}"
                                if _err_msg:
                                    history.append({
                                        "role": "user",
                                        "content": (
                                            f"Your kernel failed. {_err_msg}\n\n"
                                            "Fix the error in your kernel code and call write_file "
                                            "with the corrected version, then evaluate again. "
                                            "Do NOT re-read files you already read."
                                        ),
                                    })

                    # Detect repeated reads of the same file
                    _recent_actions = [
                        (e.get("parameters", {}).get("action", ""), e.get("parameters", {}).get("path", ""))
                        for e in events[-6:]
                        if e.get("type") == "tool"
                    ]
                    if len(_recent_actions) >= 4:
                        _last_paths = [p for a, p in _recent_actions if a == "read_file"]
                        if len(_last_paths) >= 4 and len(set(_last_paths[-4:])) <= 1:
                            _kf = "kernel.cu" if _task_lang == "opus" else "kernel.py"
                            history.append({
                                "role": "user",
                                "content": (
                                    f"You are reading the same file repeatedly. Stop reading and take action:\n"
                                    f"- If you haven't written your kernel yet: call write_file with your implementation\n"
                                    f"- If you already wrote the kernel: call evaluate to test it\n"
                                    f"- If evaluate failed: fix the error and call write_file with the corrected code"
                                ),
                            })

                    # Write-file stall detection — escalating interventions
                    turns_since_write += 1
                    if state != AgentState.TERMINATED and not has_ever_written:
                        if turns_since_write >= 50:
                            import sys as _sys
                            print(f"[WRITE_STALL] Terminating: {turns_since_write} turns without write_file", file=_sys.stderr, flush=True)
                            state = AgentState.TERMINATED
                        elif turns_since_write >= 15:
                            _kf = "kernel.cu" if _task_lang == "opus" else "kernel.py"
                            history.append({
                                "role": "user",
                                "content": (
                                    f"CRITICAL: You have read files for {turns_since_write} turns without writing. "
                                    "You have enough information. Write your COMPLETE kernel NOW.\n"
                                    f'Call: geak(action="write_file", path="{_kf}", content="<your full kernel>")\n'
                                    "Do NOT read any more files. Write the kernel immediately."
                                ),
                            })
                        elif turns_since_write >= 8:
                            history.append({
                                "role": "user",
                                "content": (
                                    "You should now write your kernel. Call:\n"
                                    '  geak(action="write_file", path="kernel.py", content="...")\n'
                                    "with your complete implementation."
                                ),
                            })

                    if state != AgentState.TERMINATED:
                        state = (
                            AgentState.TERMINATED
                            if metrics.assistant_turns >= self.max_assistant_turns
                            else AgentState.GENERATING
                        )
                    continue
                raise RuntimeError("invalid agent state: %s" % state)
        finally:
            if not self.retain_session:
                await asyncio.to_thread(self.tool.release, session_id)

        metrics.total_seconds = time.monotonic() - started
        reward_score = float(self.tool.calc_reward(session_id))
        return AgentLoopOutput(
            messages=history,
            final_text=final_text,
            policy_logprobs=policy_logprobs,
            tool_rewards=tool_rewards,
            reward_score=reward_score,
            metrics=metrics,
            session_id=session_id,
            events=events,
        )


async def gather_limited(
    coroutines: Sequence[Any], limit: int
) -> list[Any]:
    """Gather work concurrently while bounding active sessions."""

    semaphore = asyncio.Semaphore(max(1, int(limit)))

    async def guarded(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return list(await asyncio.gather(*(guarded(item) for item in coroutines)))

