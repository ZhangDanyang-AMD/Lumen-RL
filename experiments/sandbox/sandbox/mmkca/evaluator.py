"""Client for the MMKCA GEMM worker running in the gfx950 GPU image.

The host glibc cannot run MMKCA's locked LLVM tools, so the worker lives in a
container (one per GPU). A request that times out or loses the worker is an
unknown outcome and is never retried; the container is restarted lazily.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Mapping

from sandbox.micake.evaluator import STREAM_LIMIT, UNKNOWN_STAGES, MicakeWorkerError

logger = logging.getLogger(__name__)

__all__ = ["MmkcaWorkerClient", "UNKNOWN_STAGES"]


class MmkcaWorkerClient:
    """Serializes JSON-lines requests to one long-lived ``scripts/mmkca_worker.py``."""

    def __init__(
        self,
        mmkca_root: str,
        image: str,
        gpu_id: int,
        rocm_llvm: str,
        request_timeout_s: float,
    ) -> None:
        self._root = mmkca_root
        self._image = image
        self._gpu_id = gpu_id
        self._rocm_llvm = rocm_llvm
        self._timeout = request_timeout_s
        self._name = f"mmkca-worker-gpu{gpu_id}"
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    def _command(self) -> list[str]:
        rocm_root = os.path.dirname(os.path.dirname(self._rocm_llvm.rstrip("/")))
        return [
            "docker", "run", "-i", "--rm", "--name", self._name,
            "-e", f"HIP_VISIBLE_DEVICES={self._gpu_id}", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", f"MMKCA_ROCM_LLVM={self._rocm_llvm}",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--device", "/dev/kfd", "--device", "/dev/dri",
            "--group-add", str(os.stat("/dev/kfd").st_gid),
            "--group-add", str(os.stat("/dev/dri/renderD128").st_gid),
            "--security-opt", "seccomp=unconfined",
            "-v", f"{self._root}:/workspace/MMKCA", "-v", f"{rocm_root}:{rocm_root}:ro",
            "-w", "/workspace/MMKCA", self._image, "python3", "-u", "scripts/mmkca_worker.py",
        ]

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        subprocess.run(["docker", "rm", "-f", self._name], check=False, capture_output=True)
        logger.info("starting MMKCA worker %s", self._name)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=STREAM_LIMIT,
            )
        except OSError as exc:
            raise MicakeWorkerError(f"cannot start MMKCA worker: {exc}") from exc
        return self._process

    async def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        async with self._lock:
            process = await self._ensure_started()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout)
            if not line:
                raise ConnectionError("MMKCA worker exited")
            return json.loads(line)

    async def describe(self, dtype: str, portfolio: str, split: str, bundle: int | None) -> Mapping[str, Any]:
        """Domain, baseline, opening and workloads of one regime/dtype portfolio."""
        response = await self._request({"op": "describe", "dtype": dtype, "portfolio": portfolio,
                                        "split": split, "bundle": bundle})
        if not response.get("ok"):
            raise MicakeWorkerError(response.get("error", "describe failed"))
        return response["result"]

    async def evaluate(
        self,
        config: Mapping[str, Any],
        fidelity: str,
        measurement: Mapping[str, Any],
        dtype: str,
        portfolio: str,
        split: str,
        bundle: int | None,
    ) -> dict[str, Any]:
        """Evaluate one typed schedule on one portfolio; never retries."""
        payload = {"op": "evaluate", "config": dict(config), "fidelity": fidelity, "measurement": dict(measurement),
                   "dtype": dtype, "portfolio": portfolio, "split": split, "bundle": bundle}
        try:
            response = await self._request(payload)
        except (asyncio.TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            logger.error("MMKCA evaluation outcome unknown on %s: %s", self._name, exc)
            await self.close()
            return {"stage": "evaluation_unknown", "diagnostic": {"code": type(exc).__name__}}
        if not response.get("ok"):
            return {"stage": "evaluation_unknown", "diagnostic": {"code": "WORKER_FAULT", "message": response.get("error")}}
        return dict(response["result"])

    async def close(self) -> None:
        """Stop the worker; ``--rm`` removes its container and compile cache."""
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=20)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        subprocess.run(["docker", "rm", "-f", self._name], check=False, capture_output=True)
