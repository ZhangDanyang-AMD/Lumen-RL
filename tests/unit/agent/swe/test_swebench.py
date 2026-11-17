import asyncio
import os
import subprocess

import pytest

from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.sandbox_client import ExecResult
from lumenrl.agent.swe.swebench import SweBenchEnvironment, SweBenchTask, write_file


class LocalSandbox:
    """Runs commands with the local bash, HOME pointed at a scratch directory. The task images'
    `BASH_ENV=/root/.bashrc` (which activates their conda env) is dropped: there is none here."""

    def __init__(self, home):
        self.home = str(home)
        self.closed = False

    async def run(self, command, timeout_s=60.0, *, workdir=None, env=None, output_cap_bytes=None):
        env = {k: v for k, v in (env or {}).items() if k != "BASH_ENV"}
        p = await asyncio.to_thread(
            subprocess.run, ["bash", "-c", command], cwd=workdir or self.home, capture_output=True, text=True,
            env={**os.environ, "HOME": self.home, **env}, timeout=timeout_s, check=False,
        )
        return ExecResult(p.stdout + p.stderr, p.returncode, False)

    async def close(self):
        self.closed = True


class LocalClient:
    def __init__(self, home):
        self.home = home
        self.opened = []

    async def open(self, image, *, env=None, workdir="/", timeout_s=1800.0):
        self.opened.append((image, workdir))
        self.sandbox = LocalSandbox(self.home)
        return self.sandbox


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@b", "GIT_COMMITTER_NAME": "a",
                        "GIT_COMMITTER_EMAIL": "a@b"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "testbed"
    r.mkdir()
    _git(r, "init", "-q")
    (r / ".gitignore").write_text("build/\n")
    (r / "src.py").write_text("x = 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "one")
    (r / "src.py").write_text("x = 2\n")
    (r / "build").mkdir()
    (r / "build" / "forced.txt").write_text("tracked although ignored\n")
    _git(r, "add", "-A")
    _git(r, "add", "-f", "build/forced.txt")
    _git(r, "commit", "-qm", "two -- the fix the agent must not find")
    (r / "build" / "artifact.so").write_text("untracked build output\n")
    (r / "pkg" / "__pycache__").mkdir(parents=True)
    (r / "pkg" / "__pycache__" / "m.pyc").write_text("cached\n")
    return r


def test_write_file_round_trips_text_longer_than_one_command(tmp_path):
    sandbox = LocalSandbox(tmp_path)
    text = "".join(f"line {i} é\n" for i in range(20_000))

    asyncio.run(write_file(sandbox, str(tmp_path / "out.txt"), text))

    assert (tmp_path / "out.txt").read_text() == text and not (tmp_path / "out.txt.b64").exists()


def test_the_agent_sandbox_has_no_history_but_the_same_tracked_files(repo, tmp_path):
    client = LocalClient(tmp_path)
    env = SweBenchEnvironment(client, cwd=str(repo))
    task = SweBenchTask({"instance_id": "x__y-1", "image": "swebench/x:latest", "problem_statement": "p"})

    async def scenario():
        async with env.agent_session(task) as bash:
            log = (await bash({"command": "git log --oneline | wc -l"})).text
            files = (await bash({"command": "git ls-files"})).text
            status = (await bash({"command": "git status --porcelain"})).text
            return log, files, status

    log, files, status = asyncio.run(scenario())

    assert client.opened == [("swebench/x:latest", str(repo))] and client.sandbox.closed
    assert "<output>\n1\n</output>" in log
    assert "<output>\n.gitignore\nbuild/forced.txt\nsrc.py\n</output>" in files
    assert "<output>\n</output>" in status
    assert not (repo / "pkg" / "__pycache__").exists() and (repo / "build" / "artifact.so").exists()


def test_the_prompt_is_mini_swe_with_the_problem_statement():
    messages = SweBenchTask({"problem_statement": "Fix the frobnicator."}).messages()

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == mini_swe.SYSTEM_TEMPLATE.rstrip("\n")
    assert "Consider the following PR description:\nFix the frobnicator.\n</pr_description>" in messages[1]["content"]
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" in messages[1]["content"]


def test_no_patch_grades_as_unresolved_without_a_sandbox(tmp_path):
    client = LocalClient(tmp_path)
    grade = asyncio.run(SweBenchEnvironment(client).grade(SweBenchTask({"instance_id": "x"}), "  \n"))

    assert not grade.resolved and grade.reason == "no patch submitted" and client.opened == []
