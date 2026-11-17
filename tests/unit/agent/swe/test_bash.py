import asyncio

from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.bash import MiniSweBashSession, MiniSweBashTool
from lumenrl.agent.swe.sandbox_client import ExecResult


class FakeSandbox:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    async def run(self, command, timeout_s=60.0, *, workdir=None, env=None, output_cap_bytes=None):
        self.commands.append((command, timeout_s, workdir, env))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _call(session, arguments):
    return asyncio.run(session(arguments))


def test_a_command_runs_in_testbed_with_mini_swe_env_and_its_output_is_rendered():
    sandbox = FakeSandbox(ExecResult("hi\n", 0, False))
    session = MiniSweBashSession(sandbox)

    result = _call(session, {"command": "echo hi"})

    assert sandbox.commands == [("echo hi", 60.0, "/testbed", mini_swe.ENV)]
    assert result.ok and not result.done
    assert result.text == "<returncode>0</returncode>\n<output>\nhi\n</output>"


def test_a_long_output_keeps_head_and_tail_with_the_warning():
    session = MiniSweBashSession(FakeSandbox(ExecResult("a" * 6000 + "b" * 6000, 1, False)))

    text = _call(session, {"command": "cat big"}).text

    assert text.startswith("<returncode>1</returncode>\n<warning>\nThe output of your last command was too long.")
    assert "<output_head>\n" + "a" * 5000 + "\n</output_head>" in text
    assert "<elided_chars>\n2000 characters elided\n</elided_chars>" in text
    assert text.endswith("<output_tail>\n" + "b" * 5000 + "\n</output_tail>")


def test_a_timeout_reads_as_mini_swe_reports_it():
    session = MiniSweBashSession(FakeSandbox(ExecResult("partial", -1, True)))

    result = _call(session, {"command": "sleep 100"})

    assert not result.ok
    assert result.text.startswith(
        "<exception>An error occurred while executing the command: Command timed out after 60 seconds</exception>\n"
        "<returncode>-1</returncode>"
    )


def test_the_marker_with_exit_zero_submits_the_rest_of_the_output():
    patch = "diff --git a/x b/x\n+fix\n"
    session = MiniSweBashSession(FakeSandbox(ExecResult(f"\n{mini_swe.SUBMIT_MARKER}\n{patch}", 0, False)))

    result = _call(session, {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"})

    assert result.done and session.submission == patch


def test_the_marker_from_a_failing_command_does_not_submit():
    session = MiniSweBashSession(FakeSandbox(ExecResult(f"{mini_swe.SUBMIT_MARKER}\n", 1, False)))

    result = _call(session, {"command": "echo ... && cat missing"})

    assert not result.done and session.submission is None


def test_a_call_without_a_command_gets_mini_swe_format_error():
    session = MiniSweBashSession(FakeSandbox())

    result = _call(session, {"cmd": "ls"})

    assert not result.ok
    assert "<error>\nMissing 'command' argument in bash tool call.\n</error>" in result.text


def test_nothing_runs_after_the_submission():
    sandbox = FakeSandbox(ExecResult(f"{mini_swe.SUBMIT_MARKER}\n", 0, False))
    session = MiniSweBashSession(sandbox)
    _call(session, {"command": "submit"})

    result = _call(session, {"command": "rm -rf /"})

    assert result.done and len(sandbox.commands) == 1


def test_the_tool_has_mini_swe_schema_and_does_not_open_sessions_itself():
    tool = MiniSweBashTool()

    assert tool.name == "bash" and tool.schema["function"]["parameters"]["required"] == ["command"]
    try:
        tool.open()
    except RuntimeError as e:
        assert "SWE environment" in str(e)
    else:
        raise AssertionError("open() must refuse")
