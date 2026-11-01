"""The SandboxFusion code interpreter against a fake ``/run_code`` endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import io

import pytest

from lumenrl.agent import tools as tools_module
from lumenrl.agent.tools import SandboxFusionCodeTool, prepare_python_code

# The retry backoff is patched out below; the fake sandbox still has to yield.
_yield_to_loop = asyncio.sleep


def run_python(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<snippet>", "exec"), {})  # noqa: S102 - test-only snippets
    return out.getvalue()


@pytest.mark.parametrize(
    ("code", "printed"),
    [
        ("x = 6\nx * 7", "42\n"),
        ("print(42)", "42\n"),
        ("(40 +\n 2)", "42\n"),
        ("x = 42; x", "42\n"),
        ("x = 42\nx  # the answer", "42\n"),
        ("def f():\n    print('hi')\nf()", "hi\n"),
        ("x = 42", ""),
        ("for i in range(2):\n    i", ""),
        ("```python\nx = 41\nx + 1\n```", "42\n"),
    ],
)
def test_prepared_code_prints_a_trailing_value(code, printed):
    assert run_python(prepare_python_code(code)) == printed


def test_code_that_does_not_parse_is_sent_unchanged():
    assert prepare_python_code("print(1") == "print(1"


class FakeSandbox:
    def __init__(self, responses=None, failures: int = 0) -> None:
        self.requests: list[dict] = []
        self.responses = list(responses or [])
        self.failures = failures
        self.active = 0
        self.peak = 0

    async def post(self, url, payload, timeout):
        self.requests.append(payload)
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("sandbox down")
        self.active += 1
        self.peak = max(self.peak, self.active)
        await _yield_to_loop(0.01)
        self.active -= 1
        if self.responses:
            return self.responses.pop(0)
        return finished("ok\n")


def finished(stdout: str, stderr: str = "") -> dict:
    return {
        "status": "Success",
        "run_result": {"status": "Finished", "return_code": 0, "stdout": stdout, "stderr": stderr},
    }


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(tools_module.asyncio, "sleep", instant)


def make_tool(sandbox: FakeSandbox, **kwargs) -> SandboxFusionCodeTool:
    return SandboxFusionCodeTool("http://sandbox/run_code", post=sandbox.post, **kwargs)


def test_request_carries_the_prepared_code_and_limits():
    sandbox = FakeSandbox([finished("42\n")])
    tool = make_tool(sandbox, run_timeout_s=5, memory_limit_mb=512)

    result = asyncio.run(tool({"code": "6 * 7"}))

    assert result.ok and result.text == "42\n"
    [request] = sandbox.requests
    assert run_python(request["code"]) == "42\n"
    assert request["language"] == "python"
    assert request["run_timeout"] == 5
    assert request["memory_limit_MB"] == 512


def test_output_is_stdout_then_stderr_even_when_the_code_raises():
    traceback = "Traceback (most recent call last):\nZeroDivisionError: division by zero\n"
    sandbox = FakeSandbox([finished("partial\n", traceback)])

    result = asyncio.run(make_tool(sandbox)({"code": "1/0"}))

    assert result.ok
    assert result.text == "partial\n" + traceback


def test_timeout_is_reported_to_the_model():
    sandbox = FakeSandbox([
        {"status": "Failed", "run_result": {"status": "TimeLimitExceeded", "stdout": "", "stderr": ""}},
    ])

    result = asyncio.run(make_tool(sandbox, run_timeout_s=3)({"code": "while True: pass"}))

    assert not result.ok
    assert result.text == "Error: execution timed out after 3 seconds."


def test_sandbox_error_is_reported_to_the_model():
    sandbox = FakeSandbox([{"status": "SandboxError", "message": "no runner", "run_result": None}])

    result = asyncio.run(make_tool(sandbox)({"code": "1"}))

    assert not result.ok
    assert result.text == "Error: SandboxError. no runner"


def test_transport_failures_are_retried():
    sandbox = FakeSandbox(failures=2)

    result = asyncio.run(make_tool(sandbox, request_retries=2)({"code": "1"}))

    assert result.ok
    assert len(sandbox.requests) == 3


def test_an_unreachable_sandbox_is_a_failed_call():
    sandbox = FakeSandbox(failures=10)

    result = asyncio.run(make_tool(sandbox, request_retries=2)({"code": "1"}))

    assert not result.ok
    assert result.text == "Error: the code sandbox is unavailable."
    assert len(sandbox.requests) == 3


@pytest.mark.parametrize("arguments", [{}, {"code": ""}, {"code": 7}])
def test_missing_code_fails_without_a_request(arguments):
    sandbox = FakeSandbox()

    result = asyncio.run(make_tool(sandbox)(arguments))

    assert not result.ok
    assert sandbox.requests == []


def test_concurrent_calls_are_capped():
    sandbox = FakeSandbox()
    tool = make_tool(sandbox, max_concurrency=2)

    async def many():
        return await asyncio.gather(*(tool({"code": str(i)}) for i in range(6)))

    results = asyncio.run(many())

    assert all(result.ok for result in results)
    assert sandbox.peak == 2
