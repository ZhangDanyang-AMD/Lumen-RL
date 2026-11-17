import asyncio
import contextlib

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.evaluate import evaluate, mini_swe_agent, summarize
from lumenrl.agent.swe.swebench import Grade, SweBenchTask
from lumenrl.agent.tools import ToolResult
from tests.unit.agent.fakes import IM_END, ChatMLTokenizer, tool_call_text


class FakeBash:
    def __init__(self):
        self.submission = None
        self.commands = []

    async def __call__(self, arguments):
        self.commands.append(arguments["command"])
        if arguments["command"].startswith("echo " + mini_swe.SUBMIT_MARKER):
            self.submission = "diff --git a/f b/f\n"
            return ToolResult("", done=True)
        return ToolResult("<returncode>0</returncode>\n<output>\nok\n</output>")

    async def close(self):
        pass


class FakeEnv:
    """Resolves a submitted patch for tasks named `good-*`; `broken-*` cannot open a sandbox."""

    def __init__(self):
        self.sessions = []
        self.open_now = 0
        self.max_open = 0

    @contextlib.asynccontextmanager
    async def agent_session(self, task):
        if task.instance_id.startswith("broken"):
            raise RuntimeError("no sandbox")
        self.open_now += 1
        self.max_open = max(self.max_open, self.open_now)
        bash = FakeBash()
        self.sessions.append((task.instance_id, bash))
        try:
            await asyncio.sleep(0.01)
            yield bash
        finally:
            self.open_now -= 1

    async def grade(self, task, patch):
        if not patch:
            return Grade(False, False, "no patch submitted")
        return Grade(task.instance_id.startswith("good"), True, "resolved" if task.instance_id.startswith("good") else "tests failed")


def _scripted(tok, submit):
    turns = [tool_call_text("bash", {"command": "ls"})]
    if submit:
        turns.append(tool_call_text("bash", {"command": f"echo {mini_swe.SUBMIT_MARKER} && cat patch.txt"}))
    else:
        turns.append("I give up.")

    def generate_for(task):
        queue = list(turns)

        async def generate(prompt_ids, params):
            ids = tok.encode(queue.pop(0)) + [IM_END]
            return {"token_ids": ids, "logprobs": None, "finish_reason": "stop"}

        return generate

    return generate_for


def _task(name):
    return SweBenchTask({"instance_id": name, "image": "img", "problem_statement": f"fix {name}"})


def test_each_attempt_runs_in_its_own_session_and_is_graded():
    tok, env = ChatMLTokenizer(), FakeEnv()
    agent = mini_swe_agent(ChatTemplateTokenizer(tok), None)
    seen = []

    results = asyncio.run(evaluate(
        [_task("good-1"), _task("bad-1"), _task("broken-1")], agent, env, _scripted(tok, submit=True),
        {"max_tokens": 1000}, n=2, concurrency=2, max_response_tokens=4096, on_result=seen.append,
    ))

    assert len(results) == len(seen) == 6 and env.max_open <= 2
    good = [r for r in results if r["instance_id"] == "good-1"]
    assert all(r["resolved"] and r["stop_reason"] == "submitted" and r["turns"] == 2 for r in good)
    assert {r["sample"] for r in good} == {0, 1}
    assert all(not r["resolved"] and r["reason"] == "tests failed" for r in results if r["instance_id"] == "bad-1")
    assert all(r["error"] == "RuntimeError: no sandbox" for r in results if r["instance_id"] == "broken-1")
    first_task, first_bash = env.sessions[0]
    assert first_task == "good-1"
    assert first_bash.commands == ["ls", f"echo {mini_swe.SUBMIT_MARKER} && cat patch.txt"]


def test_an_attempt_that_never_submits_grades_as_no_patch():
    tok = ChatMLTokenizer()
    agent = mini_swe_agent(ChatTemplateTokenizer(tok), None)

    (r,) = asyncio.run(evaluate([_task("good-1")], agent, FakeEnv(), _scripted(tok, submit=False),
                                {"max_tokens": 1000}, max_response_tokens=4096))

    assert not r["resolved"] and r["stop_reason"] == "answer" and r["reason"] == "no patch submitted"


def test_summary_finds_the_tasks_with_both_outcomes():
    results = [
        {"instance_id": "a", "resolved": True, "stop_reason": "submitted", "reason": "resolved"},
        {"instance_id": "a", "resolved": False, "stop_reason": "submitted", "reason": "tests failed"},
        {"instance_id": "b", "resolved": True, "stop_reason": "submitted", "reason": "resolved"},
        {"instance_id": "c", "resolved": False, "error": "x"},
    ]

    s = summarize(results)

    assert s["mixed"] == ["a"] and s["tasks_ever_resolved"] == 2 and s["errors"] == 1
    assert s["resolved_rate"] == 0.5 and s["stop_reasons"] == {"submitted": 3, "error": 1}
