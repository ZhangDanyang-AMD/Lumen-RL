"""The SandboxFusion code interpreter against a fake ``/run_code`` endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import io
import subprocess
import sys
import time

import pytest

from lumenrl.agent import tools as tools_module
from lumenrl.agent.tools import SandboxFusionCodeTool, StatefulSandboxFusionCodeTool, prepare_python_code

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


# -- stateful code interpreter ----------------------------------------------------


class ExecutingSandbox:
    """Runs each program in a fresh Python process, as SandboxFusion does."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def post(self, url, payload, timeout):
        self.requests.append(payload)
        await _yield_to_loop(0.01)
        t0 = time.perf_counter()
        run = subprocess.run(
            [sys.executable, "-c", payload["code"]], capture_output=True, text=True,
            timeout=payload["run_timeout"],
        )
        return {
            "status": "Success",
            "run_result": {
                "status": "Finished", "return_code": run.returncode, "stdout": run.stdout,
                "stderr": run.stderr, "execution_time": time.perf_counter() - t0,
            },
        }


def stateful_tool(sandbox, **kwargs) -> StatefulSandboxFusionCodeTool:
    return StatefulSandboxFusionCodeTool("http://sandbox/run_code", post=sandbox.post, **kwargs)


def run_session(sandbox, *codes, **kwargs):
    session = stateful_tool(sandbox, **kwargs).open()

    async def calls():
        return [await session({"code": code}) for code in codes]

    return asyncio.run(calls())


def test_a_session_keeps_variables_functions_and_imports():
    results = run_session(ExecutingSandbox(), "import math\nx = 6\ndef f(y):\n    return y * 7", "f(x) + math.floor(0.5)")

    assert [result.text for result in results] == ["", "42\n"]


def test_earlier_output_is_not_shown_again():
    results = run_session(ExecutingSandbox(), "print('a')", "print('b')")

    assert [result.text for result in results] == ["a\n", "b\n"]


def test_random_draws_replay_as_they_first_came_out():
    first, again = run_session(ExecutingSandbox(), "import random\nv = random.random()\nv", "v")

    assert first.text == again.text != ""


def test_a_cell_that_raised_keeps_what_ran_before_the_error():
    raised, after = run_session(ExecutingSandbox(), "y = 2\n1/0", "y")

    assert raised.ok and raised.text.endswith("ZeroDivisionError: division by zero\n")
    assert after.text == "2\n"


def test_a_traceback_shows_the_cells_and_their_lines():
    _, result = run_session(ExecutingSandbox(), "def g(y):\n    return y / 0", "g(1)")

    assert result.text.startswith('Traceback (most recent call last):\n  File "<cell 2>", line 1, in <module>\n')
    assert 'File "<cell 1>", line 2, in g\n    return y / 0\n' in result.text


def test_a_call_that_did_not_finish_is_not_replayed():
    sandbox = FakeSandbox([
        {"status": "Failed", "run_result": {"status": "TimeLimitExceeded", "stdout": "", "stderr": ""}},
        finished(""),
    ])

    timed_out, _ = run_session(sandbox, "while True: pass", "x = 1")

    assert not timed_out.ok
    assert "while True" not in sandbox.requests[1]["code"]


def test_each_call_gets_the_replayed_time_on_top_of_its_timeout():
    took_4s = finished("")
    took_4s["run_result"]["execution_time"] = 4.0
    sandbox = FakeSandbox([took_4s, finished("")])

    run_session(sandbox, "x = 1", "x", run_timeout_s=10)

    assert [request["run_timeout"] for request in sandbox.requests] == [10, 14.0]


def test_sessions_do_not_share_state():
    tool = stateful_tool(ExecutingSandbox())
    first, second = tool.open(), tool.open()

    asyncio.run(first({"code": "x = 1"}))
    result = asyncio.run(second({"code": "x"}))

    assert result.text.endswith("NameError: name 'x' is not defined\n")


def test_parallel_calls_in_one_session_run_one_after_the_other():
    session = stateful_tool(ExecutingSandbox()).open()

    async def both():
        return await asyncio.gather(session({"code": "x = 1"}), session({"code": "x + 1"}))

    _, second = asyncio.run(both())

    assert second.text == "2\n"


def test_a_stateful_interpreter_runs_python_only():
    with pytest.raises(ValueError, match="Python only"):
        StatefulSandboxFusionCodeTool("http://sandbox/run_code", language="cpp")
