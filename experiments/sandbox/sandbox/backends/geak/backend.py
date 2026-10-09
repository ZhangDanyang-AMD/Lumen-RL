"""GEAK sandbox backend implementation.

Uses the GEAK task_runner subprocess pipeline for kernel evaluation.
Each task directory must contain scripts/task_runner.py with compile,
correctness, and performance modes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from sandbox.base import CommandResult, EvalResult, SandboxBackend


_PERF_RE = re.compile(
    r"Perf:\s*([0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*\(([^)]+)\))?", re.I
)


class GEAKBackend(SandboxBackend):
    """Evaluate kernels using GEAK task_runner.py subprocess pipeline.

    Each task must have a directory with:
      - kernel.py: the kernel source to evaluate
      - scripts/task_runner.py: harness with compile/correctness/performance modes
    """

    def __init__(self, command_timeout: int = 120, **kwargs):
        self.command_timeout = command_timeout

    def prepare_workspace(
        self,
        task_id: str,
        kernel_source: str,
        task_dir: Path,
        gpu_id: int,
    ) -> Path:
        workdir = Path(tempfile.mkdtemp(prefix=f"sandbox_{task_id[:20]}_"))
        shutil.copytree(task_dir, workdir / "w", dirs_exist_ok=True)
        ws = workdir / "w"
        (ws / "kernel.py").write_text(kernel_source)
        return ws

    def run_stage(
        self,
        workspace: Path,
        stage: str,
        gpu_id: int,
        timeout: int = 0,
    ) -> CommandResult:
        if timeout <= 0:
            timeout = self.command_timeout

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        try:
            proc = subprocess.run(
                ["python3", "scripts/task_runner.py", stage],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            per_case = {}
            if stage == "performance":
                per_case = self._parse_perf(proc.stdout)

            return CommandResult(
                ok=proc.returncode == 0,
                mode=stage,
                stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:],
                elapsed_s=0.0,
                per_case_ms=per_case,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                ok=False, mode=stage, stderr="timeout",
                timed_out=True,
            )
        except Exception as e:
            return CommandResult(ok=False, mode=stage, stderr=str(e))

    def cleanup(self, workspace: Path) -> None:
        parent = workspace.parent
        if parent.name.startswith("sandbox_"):
            shutil.rmtree(parent, ignore_errors=True)
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _parse_perf(stdout: str) -> dict[str, float]:
        values: dict[str, float] = {}
        idx = 0
        for m in _PERF_RE.finditer(stdout):
            name = m.group(2) or f"case_{idx}"
            values[name] = float(m.group(1))
            idx += 1
        return values
