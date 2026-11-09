"""Client for the MICake evaluation worker subprocess.

The worker runs under MICake's Python (``micake_python``) with one visible GPU.
A request that times out or loses the worker is reported as an unknown outcome
and is never retried; the worker is restarted lazily for the next request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).with_name("worker.py")
UNKNOWN_STAGES = frozenset({"evaluation_unknown", "compile_unknown", "gpu_unknown"})
STREAM_LIMIT = 16 * 1024 * 1024


class MicakeWorkerError(RuntimeError):
    """The worker could not be started or returned a protocol error."""


class MicakeWorkerClient:
    """Serializes JSON-lines requests to one long-lived MICake worker."""

    def __init__(
        self,
        micake_root: str,
        micake_python: str,
        gpu_id: int,
        request_timeout_s: float,
        rocm_path: str,
    ) -> None:
        self._micake_root = micake_root
        self._micake_python = micake_python
        self._gpu_id = gpu_id
        self._timeout = request_timeout_s
        self._rocm_path = rocm_path
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._cache_dir = tempfile.mkdtemp(prefix=f"micake-triton-gpu{gpu_id}-")

    def _environment(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "USER", "LANG")}
        env.update(
            {
                "HIP_VISIBLE_DEVICES": str(self._gpu_id),
                "ROCM_PATH": self._rocm_path,
                "HIP_PATH": self._rocm_path,
                "TRITON_LIBHIP_PATH": f"{self._rocm_path}/lib/libamdhip64.so",
                "TRITON_CACHE_DIR": self._cache_dir,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return env

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        logger.info("starting MICake worker on GPU %d", self._gpu_id)
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._micake_python, "-B", str(WORKER_PATH), "--micake-root", self._micake_root,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                env=self._environment(),
                limit=STREAM_LIMIT,
            )
        except OSError as exc:
            raise MicakeWorkerError(f"cannot start MICake worker: {exc}") from exc
        return self._process

    async def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        async with self._lock:
            process = await self._ensure_started()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout)
            if not line:
                raise ConnectionError("MICake worker exited")
            return json.loads(line)

    async def describe(self) -> Mapping[str, Any]:
        """Return the candidate domain, baseline config and workload portfolio."""
        response = await self._request({"op": "describe"})
        if not response.get("ok"):
            raise MicakeWorkerError(response.get("error", "describe failed"))
        return response["result"]

    async def evaluate(
        self,
        config: Mapping[str, Any],
        fidelity: str,
        measurement: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate one typed candidate; never retries.

        Returns:
            Worker result with a ``stage`` field. Timeouts and worker loss yield
            ``stage == "evaluation_unknown"``.
        """
        payload = {"op": "evaluate", "config": dict(config), "fidelity": fidelity, "measurement": dict(measurement)}
        try:
            response = await self._request(payload)
        except (asyncio.TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            logger.error("MICake evaluation outcome unknown: %s", exc)
            await self.close()
            return {"stage": "evaluation_unknown", "diagnostic": {"code": type(exc).__name__}}
        if not response.get("ok"):
            return {"stage": "evaluation_unknown", "diagnostic": {"code": "WORKER_FAULT", "message": response.get("error")}}
        return dict(response["result"])

    async def close(self) -> None:
        """Terminate the worker if running."""
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
