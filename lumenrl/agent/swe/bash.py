"""mini-swe-agent's bash tool, run in a sandbox: what the model sees and how it submits."""

from __future__ import annotations

import asyncio
from typing import Any

from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.sandbox_client import SandboxSession
from lumenrl.agent.tools import ToolResult, ToolSession


def render(template: str, **variables: Any) -> str:
    from jinja2 import StrictUndefined, Template

    return Template(template, undefined=StrictUndefined).render(**variables)


class MiniSweBashSession:
    """One rollout's `bash` calls in its sandbox.

    Every command runs in a fresh `bash -c` in `cwd` with mini-swe-agent's environment; calls
    run one at a time, in order. A command whose output starts with the submission marker and
    that exits 0 is the submission: the rest of its output is the patch, kept in `submission`,
    and the result is `done`. A timeout is reported the way mini-swe-agent reports one: return
    code -1 and the exception.
    """

    def __init__(self, sandbox: SandboxSession, *, cwd: str = mini_swe.CWD,
                 timeout_s: float = mini_swe.COMMAND_TIMEOUT_S, env: dict[str, str] | None = None) -> None:
        self._sandbox, self._cwd, self._timeout_s = sandbox, cwd, float(timeout_s)
        self._env = dict(mini_swe.ENV if env is None else env)
        self._lock = asyncio.Lock()
        self.submission: str | None = None

    async def __call__(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str):
            error = "Missing 'command' argument in bash tool call."
            return ToolResult(render(mini_swe.FORMAT_ERROR_TEMPLATE, error=error, actions=[], has_tool_calls=True),
                              ok=False)
        async with self._lock:
            if self.submission is not None:
                return ToolResult("The task has already been submitted.", ok=False, done=True)
            output, ok = await self._execute(command)
            lines = output["output"].lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() == mini_swe.SUBMIT_MARKER and output["returncode"] == 0:
                self.submission = "".join(lines[1:])
                return ToolResult("", done=True)
        return ToolResult(render(mini_swe.OBSERVATION_TEMPLATE, output=output), ok=ok)

    async def _execute(self, command: str) -> tuple[dict[str, Any], bool]:
        try:
            r = await self._sandbox.run(command, self._timeout_s, workdir=self._cwd, env=self._env)
        except Exception as e:  # noqa: BLE001 - the model sees what went wrong, as in mini-swe-agent
            return {"output": "", "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: {e}"}, False
        if r.timed_out:
            return {"output": r.output, "returncode": -1,
                    "exception_info": "An error occurred while executing the command: "
                                      f"Command timed out after {self._timeout_s:g} seconds"}, False
        return {"output": r.output, "returncode": r.exit_code, "exception_info": ""}, True

    async def close(self) -> None:
        """The environment that opened the sandbox closes it, after grading."""


class MiniSweBashTool:
    """The `bash` tool's schema for the prompt. Its sessions come from the SWE environment,
    which opens the sandbox before the rollout and grades it after, so it does not open one."""

    name = "bash"

    @property
    def schema(self) -> dict[str, Any]:
        return mini_swe.BASH_TOOL

    def open(self) -> ToolSession:
        raise RuntimeError("bash sessions are opened by the SWE environment; pass them to ToolAgentLoop.run")
