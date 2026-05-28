"""Lightweight distributed profiler dispatcher for LumenRL.

This module mirrors the most-used VERL profiler semantics:
- rank filtering (`all_ranks` / `ranks`)
- step-gated activation
- explicit `start(profile_step=...)` / `stop()` control

Current backend support:
- `torch`: writes chrome traces via ``torch.profiler``.
- `rocprof`: emits step ranges (via optional roctx) and generates launch
  scripts/flags for external `rocprof` wrapping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shlex
from typing import Any

import torch

from lumenrl.core.config import ProfilerConfig, RocprofToolConfig, TorchProfilerToolConfig

logger = logging.getLogger(__name__)


class _NoOpProfiler:
    def start(self, **kwargs: Any) -> None:
        return

    def stop(self) -> None:
        return


class _TorchStepProfiler:
    def __init__(self, rank: int, config: ProfilerConfig) -> None:
        self.rank = rank
        self.config = config
        self._ctx: torch.profiler.profile | None = None
        self._step: int | None = None

    def start(self, **kwargs: Any) -> None:
        profile_step = kwargs.get("profile_step")
        tool_cfg = self.config.tool_config
        if not isinstance(tool_cfg, TorchProfilerToolConfig):
            tool_cfg = TorchProfilerToolConfig()
        contents = set(tool_cfg.contents)
        activities: list[torch.profiler.ProfilerActivity] = []
        if "cpu" in contents:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in contents and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if not activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)

        self._ctx = torch.profiler.profile(
            activities=activities,
            record_shapes="shapes" in contents,
            profile_memory="memory" in contents,
            with_stack="stack" in contents,
        )
        self._ctx.__enter__()
        self._step = int(profile_step) if profile_step is not None else None

    def stop(self) -> None:
        if self._ctx is None:
            return
        # Advance once so profiler has a finalized step to export.
        self._ctx.step()
        self._ctx.__exit__(None, None, None)
        out_dir = Path(self.config.save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        step_str = f"step{self._step}" if self._step is not None else "step_unknown"
        trace_path = out_dir / f"{step_str}_rank{self.rank}_torch.json"
        try:
            self._ctx.export_chrome_trace(str(trace_path))
            logger.info("Profiler trace exported: %s", trace_path)
        except Exception as exc:  # pragma: no cover - best effort only
            logger.warning("Failed to export profiler trace to %s: %s", trace_path, exc)
        finally:
            self._ctx = None
            self._step = None


class _RocprofParamProfiler:
    """Generate rocprof launch artifacts and optional roctx step markers.

    `rocprof` is typically launched outside the Python process:
      rocprof [flags] python -m lumenrl.trainer.main ...

    This profiler helps by:
    1) mapping config -> rocprof CLI flags
    2) writing per-rank launch helper scripts under `save_path`
    3) optionally pushing roctx ranges at profiled steps (if `roctx` is installed)
    """

    def __init__(self, rank: int, config: ProfilerConfig) -> None:
        self.rank = rank
        self.config = config
        tool_cfg = config.tool_config
        if isinstance(tool_cfg, RocprofToolConfig):
            self.tool_config = tool_cfg
        else:
            self.tool_config = RocprofToolConfig()
        self._range_started = False
        self._step: int | None = None
        self._roctx = self._try_import_roctx()
        self._wrote_hints = False

    @staticmethod
    def _try_import_roctx() -> Any | None:
        try:
            import roctx  # type: ignore[import-not-found]

            return roctx
        except Exception:
            return None

    def _build_rocprof_flags(self, rank: int) -> list[str]:
        cfg = self.tool_config
        flags: list[str] = []
        if cfg.hip_trace:
            flags.append("--hip-trace")
        if cfg.hsa_trace:
            flags.append("--hsa-trace")
        if cfg.kernel_trace:
            flags.append("--kernel-trace")
        if cfg.memory_copy_trace:
            flags.append("--memory-copy-trace")
        if cfg.sys_trace:
            flags.append("--sys-trace")
        if cfg.timestamp_on:
            flags.append("--timestamp")
        if cfg.stats:
            flags.append("--stats")
        if cfg.output_format:
            flags.extend(["--format", cfg.output_format])
        if cfg.kernel_regex:
            flags.extend(["--kernel-regex", cfg.kernel_regex])
        if cfg.output_file:
            out_dir = Path(self.config.save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{cfg.output_file}_rank{rank}"
            flags.extend(["--output", str(out_file)])
        if cfg.extra_args:
            flags.extend(cfg.extra_args)
        return flags

    def _write_launch_hints(self) -> None:
        if self._wrote_hints:
            return
        out_dir = Path(self.config.save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        flags = self._build_rocprof_flags(self.rank)
        payload = {
            "tool": "rocprof",
            "rank": self.rank,
            "flags": flags,
            "note": "Launch profiler externally: rocprof <flags> python -m lumenrl.trainer.main ...",
        }
        (out_dir / f"rocprof_rank{self.rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        cmd = "rocprof " + " ".join(shlex.quote(f) for f in flags) + " python -m lumenrl.trainer.main --config <config>"
        (out_dir / f"rocprof_rank{self.rank}.sh").write_text(cmd + "\n", encoding="utf-8")
        logger.info("rocprof launch hints written under %s", out_dir)
        self._wrote_hints = True

    def start(self, **kwargs: Any) -> None:
        self._write_launch_hints()
        profile_step = kwargs.get("profile_step")
        self._step = int(profile_step) if profile_step is not None else None
        if self._roctx is not None:
            tag = f"lumenrl_step_{self._step}" if self._step is not None else "lumenrl_step"
            try:
                self._roctx.range_push(tag)
                self._range_started = True
            except Exception:
                self._range_started = False

    def stop(self) -> None:
        if self._roctx is not None and self._range_started:
            try:
                self._roctx.range_pop()
            except Exception:
                pass
        self._range_started = False
        self._step = None


class DistProfiler:
    """Dispatcher that gates profiling by rank and tool."""

    def __init__(self, rank: int, config: ProfilerConfig | None = None) -> None:
        self.rank = rank
        self.config = config or ProfilerConfig(enable=False)
        self._enable = bool(self.config.enable)
        self._this_step = False
        self._this_rank = self._resolve_rank_gate()

        tool = (self.config.tool or "").lower()
        if tool == "torch":
            self._impl = _TorchStepProfiler(rank=rank, config=self.config)
        elif tool == "rocprof":
            self._impl = _RocprofParamProfiler(rank=rank, config=self.config)
        else:
            self._impl = _NoOpProfiler()
            if self._enable:
                logger.warning("Unsupported profiler tool '%s'; using no-op profiler.", tool)

    def _resolve_rank_gate(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        return self.rank == 0

    def check_enable(self) -> bool:
        return self._enable

    def check_this_rank(self) -> bool:
        return self._this_rank

    def check_this_step(self) -> bool:
        return self._this_step

    def start(self, **kwargs: Any) -> None:
        if not (self.check_enable() and self.check_this_rank()):
            return
        self._this_step = True
        self._impl.start(**kwargs)

    def stop(self) -> None:
        if not (self.check_enable() and self.check_this_rank()):
            return
        self._this_step = False
        self._impl.stop()
