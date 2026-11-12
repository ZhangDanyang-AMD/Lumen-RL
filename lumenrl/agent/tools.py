"""Tools an agent loop can call, and the SandboxFusion code interpreter."""

from __future__ import annotations

import ast
import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    text: str
    # False when the call did not run to completion: bad arguments, a timeout,
    # or the sandbox itself failing. An exception in the model's code is a
    # successful call whose output is the traceback.
    ok: bool = True


class Tool(Protocol):
    name: str

    @property
    def schema(self) -> dict[str, Any]:
        """The OpenAI function schema the chat template renders into the prompt."""

    async def __call__(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolSession(Protocol):
    """One rollout's calls to a stateful tool; they share state until ``close``."""

    async def __call__(self, arguments: dict[str, Any]) -> ToolResult: ...

    async def close(self) -> None: ...


class StatefulTool(Protocol):
    """A tool whose calls in one rollout see each other's effects.

    The agent loop opens a session on a rollout's first call to the tool and
    closes it when the rollout ends, however it ends.
    """

    name: str

    @property
    def schema(self) -> dict[str, Any]: ...

    def open(self) -> ToolSession: ...


_CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_RESULT_NAME = "_lumenrl_last_value"


def prepare_python_code(code: str) -> str:
    """Unwrap a fenced block and print the value of a trailing expression.

    Models often end a snippet with a bare expression and expect a notebook to
    show it. As in a notebook, the value is printed unless it is None, so a
    trailing ``solve()`` that prints for itself does not add a stray "None".
    """
    fenced = _CODE_FENCE.findall(code)
    if fenced:
        code = fenced[0]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    value = last.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
    ):
        return code
    lines = code.splitlines()
    first, final = last.lineno - 1, last.end_lineno
    head = lines[first][: last.col_offset]
    tail = lines[final - 1][last.end_col_offset:]
    segment = ast.get_source_segment(code, last)
    replacement = (
        f"{head}{_RESULT_NAME} = ({segment}){tail}\n"
        f"if {_RESULT_NAME} is not None:\n"
        f"    print({_RESULT_NAME})"
    )
    return "\n".join([*lines[:first], replacement, *lines[final:]])


PostFn = Callable[[str, dict[str, Any], float], Awaitable[dict[str, Any]]]


class SandboxFusionCodeTool:
    """``code_interpreter(code)`` run by a SandboxFusion ``/run_code`` service.

    The schema and output format follow verl's ReTool recipe: the tool returns
    the program's stdout followed by its stderr.
    """

    name = "code_interpreter"

    def __init__(
        self,
        url: str,
        *,
        language: str = "python",
        run_timeout_s: float = 30.0,
        memory_limit_mb: int = 1024,
        max_concurrency: int = 64,
        request_retries: int = 3,
        post: PostFn | None = None,
    ) -> None:
        self._url = url
        self._language = language
        self._run_timeout_s = float(run_timeout_s)
        self._memory_limit_mb = int(memory_limit_mb)
        self._request_retries = int(request_retries)
        self._slots = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._post = post or self._aiohttp_post
        self._session: Any = None

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "A tool for executing code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to execute."},
                    },
                    "required": ["code"],
                },
            },
        }

    async def __call__(self, arguments: dict[str, Any]) -> ToolResult:
        code = self._code(arguments)
        if isinstance(code, ToolResult):
            return code
        result, _ = await self._run(code, self._run_timeout_s)
        return result

    def _code(self, arguments: dict[str, Any]) -> str | ToolResult:
        """The program to run for ``arguments``, or the error the model gets."""
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                f"Error: {self.name} needs a non-empty string argument 'code'.", ok=False,
            )
        return prepare_python_code(code) if self._language == "python" else code

    async def _run(self, code: str, run_timeout_s: float) -> tuple[ToolResult, float]:
        """Run ``code``; return the result and the seconds it ran."""
        payload = {
            "code": code,
            "language": self._language,
            "run_timeout": run_timeout_s,
            "compile_timeout": self._run_timeout_s,
            "memory_limit_MB": self._memory_limit_mb,
        }
        async with self._slots:
            body = await self._request(payload)
        if body is None:
            return ToolResult("Error: the code sandbox is unavailable.", ok=False), 0.0
        run_s = float((body.get("run_result") or {}).get("execution_time") or run_timeout_s)
        return self._format(body), run_s

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        timeout = 2 * self._run_timeout_s + 30.0
        for attempt in range(self._request_retries + 1):
            try:
                return await self._post(self._url, payload, timeout)
            except Exception as e:  # noqa: BLE001 - any transport failure is retried
                if attempt == self._request_retries:
                    logger.warning("SandboxFusion request to %s failed: %r", self._url, e)
                    return None
                await asyncio.sleep(0.5 * 2**attempt)
        return None

    def _format(self, body: dict[str, Any]) -> ToolResult:
        run = body.get("run_result") or {}
        output = (run.get("stdout") or "") + (run.get("stderr") or "")
        status = run.get("status")
        if status == "Finished":
            return ToolResult(output)
        if status == "TimeLimitExceeded":
            return ToolResult(
                f"{output}Error: execution timed out after {self._run_timeout_s:g} seconds.",
                ok=False,
            )
        compile_run = body.get("compile_result") or {}
        detail = (compile_run.get("stderr") or "") + output or body.get("message") or ""
        return ToolResult(
            f"Error: {status or body.get('status', 'sandbox error')}. {detail}".rstrip(), ok=False,
        )

    async def _aiohttp_post(
        self, url: str, payload: dict[str, Any], timeout: float,
    ) -> dict[str, Any]:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with self._session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


_REPLAY_PROGRAM = """import contextlib as _lumenrl_contextlib, io as _lumenrl_io, linecache as _lumenrl_linecache
import random as _lumenrl_random, traceback as _lumenrl_traceback
_lumenrl_random.seed({seed})
try:
    import numpy as _lumenrl_numpy
    _lumenrl_numpy.random.seed({seed})
except ImportError:
    pass
_lumenrl_cells = {cells!r}
for _lumenrl_i, _lumenrl_cell in enumerate(_lumenrl_cells, 1):
    _lumenrl_linecache.cache["<cell %d>" % _lumenrl_i] = (0, None, _lumenrl_cell.splitlines(True), "<cell>")
for _lumenrl_i, _lumenrl_cell in enumerate(_lumenrl_cells[:-1], 1):
    with _lumenrl_contextlib.redirect_stdout(_lumenrl_io.StringIO()), \\
            _lumenrl_contextlib.redirect_stderr(_lumenrl_io.StringIO()):
        try:
            exec(compile(_lumenrl_cell, "<cell %d>" % _lumenrl_i, "exec"))
        except BaseException:
            pass
try:
    exec(compile(_lumenrl_cells[-1], "<cell %d>" % len(_lumenrl_cells), "exec"))
except SystemExit:
    raise
except BaseException as _lumenrl_error:
    _lumenrl_traceback.print_exception(
        type(_lumenrl_error), _lumenrl_error, _lumenrl_error.__traceback__.tb_next,
    )
"""


class _ReplaySession:
    """A Python session rebuilt for every call by re-running the earlier cells.

    SandboxFusion runs each program in a fresh process, and its Jupyter
    endpoint does not start in the image used here, so the state lives in the
    session: each call runs the cells that finished before it with their output
    discarded, then the new cell. ``random`` and NumPy are seeded once per
    session, so earlier draws come out the same on every replay. What a cell
    took from the clock or the outside world is not reproduced, and every call
    pays for the replay, so it gets the replayed cells' time on top of its own
    timeout. A cell that timed out or failed in the sandbox is not replayed; a
    cell that raised is, up to where it raised.
    """

    def __init__(self, tool: SandboxFusionCodeTool, seed: int) -> None:
        self._tool = tool
        self._seed = seed
        self._cells: list[str] = []
        self._replay_s = 0.0
        self._lock = asyncio.Lock()

    async def __call__(self, arguments: dict[str, Any]) -> ToolResult:
        code = self._tool._code(arguments)
        if isinstance(code, ToolResult):
            return code
        async with self._lock:
            program = _REPLAY_PROGRAM.format(seed=self._seed, cells=[*self._cells, code])
            result, run_s = await self._tool._run(program, self._tool._run_timeout_s + self._replay_s)
            if result.ok:
                self._cells.append(code)
                self._replay_s = run_s
        return result

    async def close(self) -> None:
        self._cells.clear()


class StatefulSandboxFusionCodeTool(SandboxFusionCodeTool):
    """``code_interpreter`` whose calls in one rollout share a Python session.

    Variables, functions and imports from one call are there in the next, as
    in a notebook. See ``_ReplaySession`` for how the state is kept.
    """

    def __init__(self, url: str, **kwargs: Any) -> None:
        if kwargs.get("language", "python") != "python":
            raise ValueError("a stateful code interpreter runs Python only")
        super().__init__(url, **kwargs)

    def open(self) -> ToolSession:
        return _ReplaySession(self, seed=random.randrange(2**31))
