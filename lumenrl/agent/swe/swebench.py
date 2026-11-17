"""SWE-bench tasks as rollout environments: the prompt, a sandbox for the agent, and grading.

Rows are in the v5 dataset format (`SWE-bench/SWE-bench_Verified` and its kin), which carry the
task image, the eval script, the log parser and the eval type themselves.

The agent's sandbox has the repository's history removed -- `/testbed` becomes a single commit
of the same tracked files -- and its `__pycache__` purged, as DeepSeek-V4.1's evaluation does
against reward hacking. Grading follows the official harness: the submitted patch goes into a
*fresh* container of the task image (so nothing else the agent left behind counts), is applied
the ways the harness tries, the eval script runs, and `swebench`'s own report decides.
"""

from __future__ import annotations

import base64
import contextlib
import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.bash import MiniSweBashSession, render
from lumenrl.agent.swe.sandbox_client import SandboxClient, SandboxSession

# swebench.harness.run_evaluation.GIT_APPLY_CMDS, in its order
GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
)
TESTS_TIMEOUT = ">>>>> Tests Timed Out"  # swebench.harness.constants.TESTS_TIMEOUT
START_TEST_OUTPUT = ">>>>> Start Test Output"

PREPARE = r"""set -euo pipefail
cd {cwd}
git config --global --add safe.directory {cwd}
git ls-files -z > /tmp/.lumenrl_tracked
rm -rf .git
git init -q
git add -f --pathspec-from-file=/tmp/.lumenrl_tracked --pathspec-file-nul
git -c user.name=swe -c user.email=swe@localhost commit -q --no-verify -m "Initial commit"
rm -f /tmp/.lumenrl_tracked
find {cwd} -name __pycache__ -type d -prune -exec rm -rf {{}} +
git rev-list --count HEAD
"""

_CHUNK = 60_000  # base64 characters per command: one argv string must stay under 128 KiB


@dataclass(frozen=True)
class SweBenchTask:
    row: dict[str, Any] = field(repr=False)

    @property
    def instance_id(self) -> str:
        return self.row["instance_id"]

    @property
    def image(self) -> str:
        return self.row["image"]

    @property
    def setup_script(self) -> str:
        """Run first in every sandbox of the task, where its image needs repairing (SWE-rebench's)."""
        return self.row.get("setup_script") or ""

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": render(mini_swe.SYSTEM_TEMPLATE)},
            {"role": "user", "content": render(mini_swe.INSTANCE_TEMPLATE, task=self.row["problem_statement"])},
        ]


@dataclass(frozen=True)
class Grade:
    resolved: bool
    applied: bool
    reason: str
    tests_status: dict[str, Any] | None = None


async def write_file(sandbox: SandboxSession, path: str, text: str) -> None:
    data = base64.b64encode(text.encode()).decode()
    staged = path + ".b64"
    await _check(sandbox, f": > {staged}")
    for i in range(0, len(data), _CHUNK):
        await _check(sandbox, f"printf %s {data[i:i + _CHUNK]} >> {staged}")
    await _check(sandbox, f"base64 -d {staged} > {path} && rm -f {staged}")


async def _check(sandbox: SandboxSession, command: str, timeout_s: float = 60.0) -> str:
    r = await sandbox.run(command, timeout_s)
    if r.exit_code != 0:
        raise RuntimeError(f"`{command[:80]}` exited {r.exit_code}: {r.output[-2000:]}")
    return r.output


class SweBenchEnvironment:
    def __init__(self, client: SandboxClient, *, cwd: str = mini_swe.CWD, eval_timeout_s: float = 1800.0,
                 command_timeout_s: float = mini_swe.COMMAND_TIMEOUT_S) -> None:
        self._client, self._cwd = client, cwd
        self._eval_timeout_s, self._command_timeout_s = float(eval_timeout_s), float(command_timeout_s)

    @contextlib.asynccontextmanager
    async def agent_session(self, task: SweBenchTask) -> AsyncIterator[MiniSweBashSession]:
        """The agent's `bash` session in a prepared sandbox of `task`, closed on exit."""
        sandbox = await self._client.open(task.image, workdir=self._cwd)
        try:
            if task.setup_script:
                await _check(sandbox, task.setup_script, timeout_s=600.0)
            commits = (await _check(sandbox, PREPARE.format(cwd=self._cwd), timeout_s=600.0)).strip()
            if commits.splitlines()[-1:] != ["1"]:
                raise RuntimeError(f"{task.instance_id}: history not removed: {commits[-500:]}")
            yield MiniSweBashSession(sandbox, cwd=self._cwd, timeout_s=self._command_timeout_s)
        finally:
            await sandbox.close()

    async def grade(self, task: SweBenchTask, patch: str | None) -> Grade:
        """Whether `patch` resolves `task`, judged as the SWE-bench harness judges it."""
        if not patch or not patch.strip():
            return Grade(False, False, "no patch submitted")
        from swebench.harness.grading import get_eval_report
        from swebench.harness.utils import make_test_spec

        spec = make_test_spec(task.row)
        sandbox = await self._client.open(task.image, workdir=self._cwd)
        try:
            if task.setup_script:
                await _check(sandbox, task.setup_script, timeout_s=600.0)
            await write_file(sandbox, "/tmp/patch.diff", patch)
            applied = None
            for cmd in GIT_APPLY_CMDS:
                r = await sandbox.run(f"cd {self._cwd} && {cmd} /tmp/patch.diff", 300.0)
                if r.exit_code == 0:
                    applied = cmd
                    break
            if applied is None:
                return Grade(False, False, "patch does not apply")
            await write_file(sandbox, "/tmp/eval.sh", spec.eval_script)
            run = await sandbox.run("bash /tmp/eval.sh > /tmp/eval.log 2>&1", self._eval_timeout_s)
            log = (await sandbox.run("cat /tmp/eval.log", 300.0, output_cap_bytes=64 << 20)).output
            if run.timed_out:
                log += f"\n{TESTS_TIMEOUT}\n"
        finally:
            await sandbox.close()
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log)
        try:
            prediction = {"instance_id": task.instance_id, "model_name_or_path": "lumenrl", "model_patch": patch}
            report = get_eval_report(spec, prediction, f.name, include_tests_status=True)[task.instance_id]
        finally:
            os.unlink(f.name)
        reason = "resolved" if report["resolved"] else ("tests failed" if START_TEST_OUTPUT in log else "no test output")
        return Grade(bool(report["resolved"]), True, reason, report.get("tests_status"))
