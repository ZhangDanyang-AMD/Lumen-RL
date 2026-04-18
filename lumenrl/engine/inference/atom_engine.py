"""vLLM-backed inference engine with subprocess isolation for RL training.

Uses vLLM's ``LLM`` class for efficient generation with PagedAttention
and continuous batching.  Runs vLLM in a **separate subprocess** via
``subprocess.Popen`` to get full process isolation from torchrun's
NCCL process groups.

Communication: uses a pair of named FIFOs (named pipes) for JSON
commands/responses, bypassing stdout/stderr which can be polluted
by C++ libraries (Gloo, NCCL).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

import torch

from lumenrl.core.config import AtomConfig

logger = logging.getLogger(__name__)

_WEIGHT_SYNC_DIR = os.environ.get(
    "LUMENRL_WEIGHT_SYNC_DIR",
    "/tmp/lumenrl_weight_sync",
)

_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
for key in list(os.environ.keys()):
    if any(key.startswith(p) for p in [
        "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
        "OMP_NUM_THREADS",
    ]):
        del os.environ[key]

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

import torch
from vllm import LLM, SamplingParams

cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
gpu_mem = float(sys.argv[4])
max_model_len = int(sys.argv[5]) if sys.argv[5] != "None" else None
kv_cache_dtype = sys.argv[6]

kwargs = {
    "model": model_path,
    "gpu_memory_utilization": gpu_mem,
    "enforce_eager": True,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "tensor_parallel_size": 1,
}
if max_model_len is not None:
    kwargs["max_model_len"] = max_model_len
if kv_cache_dtype != "auto":
    kwargs["kv_cache_dtype"] = kv_cache_dtype

llm = LLM(**kwargs)

resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({"status": "ready"}) + "\\n")
resp_f.flush()

cmd_f = open(cmd_fifo, "r")

for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "generate":
        prompts = msg["prompts"]
        sp_dict = msg.get("sampling_params", {})
        vllm_sp = SamplingParams(
            max_tokens=int(sp_dict.get("max_tokens", sp_dict.get("max_new_tokens", 128))),
            temperature=float(sp_dict.get("temperature", 1.0)),
            top_p=float(sp_dict.get("top_p", 1.0)),
            top_k=int(sp_dict.get("top_k", -1)),
        )
        outputs = llm.generate(prompts, vllm_sp, use_tqdm=False)
        results = [o.outputs[0].text if o.outputs else "" for o in outputs]
        resp_f.write(json.dumps({"results": results}) + "\\n")
        resp_f.flush()

    elif cmd == "sleep":
        del llm
        llm = None
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "wake":
        new_path = msg.get("model_path")
        if new_path:
            kwargs["model"] = new_path
        if llm is not None:
            del llm
            torch.cuda.empty_cache()
            gc.collect()
        llm = LLM(**kwargs)
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "shutdown":
        try:
            del llm
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class AtomEngine:
    """vLLM inference engine with subprocess isolation for colocated RL.

    Spawns vLLM in a completely separate process via ``subprocess.Popen``.
    Uses named FIFOs for JSON communication to avoid stdout/stderr pollution
    from C++ libraries (Gloo, NCCL).
    """

    def __init__(self, config: AtomConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._proc: subprocess.Popen | None = None
        self._initialized = False
        self._weight_dir: str | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_awake(self) -> bool:
        return self._initialized and self._proc is not None and self._proc.poll() is None

    def _start_worker(self, model_path: str | None = None) -> None:
        """Start the vLLM worker subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            return

        path = model_path or self._weight_dir or self._model_name
        gpu_mem = self._config.gpu_memory_utilization
        env_mem = os.environ.get("ATOM_GPU_MEMORY_UTILIZATION")
        if env_mem is not None:
            gpu_mem = float(env_mem)

        gpu_id = int(os.environ.get("LOCAL_RANK", "0"))

        fifo_dir = tempfile.mkdtemp(prefix="lumenrl_fifo_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_CONFIGURE_LOGGING"] = "0"
        for key in list(env.keys()):
            if any(key.startswith(p) for p in [
                "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
                "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
                "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
                "OMP_NUM_THREADS",
            ]):
                del env[key]

        max_model_len_str = str(self._config.max_model_len) if self._config.max_model_len is not None else "None"

        logger.info(
            "AtomEngine: starting vLLM worker for %s (gpu=%d, mem=%.0f%%)",
            path, gpu_id, gpu_mem * 100,
        )

        self._proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", _WORKER_SCRIPT,
                self._cmd_fifo, self._resp_fifo,
                path, str(gpu_mem), max_model_len_str,
                self._config.kv_cache_dtype,
            ],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env=env,
            start_new_session=True,
        )

        self._resp_f = open(self._resp_fifo, "r")

        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("vLLM worker subprocess exited before ready")
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"vLLM worker failed to start: {resp}")

        self._cmd_f = open(self._cmd_fifo, "w")

        self._initialized = True
        logger.info("AtomEngine: vLLM worker ready (pid=%d).", self._proc.pid)

    def _send_cmd(self, cmd: dict) -> dict:
        """Send a JSON command to the worker and read the response."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("vLLM worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("vLLM worker closed response FIFO unexpectedly")
        return json.loads(resp_line)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Generate text completions via the vLLM worker."""
        if not self.is_awake:
            self._start_worker()

        sp = dict(sampling_params or {})
        str_prompts = [p if isinstance(p, str) else str(p) for p in prompts]

        resp = self._send_cmd({
            "cmd": "generate",
            "prompts": str_prompts,
            "sampling_params": sp,
        })
        return resp["results"]

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Save a new weight snapshot for the next ``wake()`` cycle."""
        sync_dir = Path(_WEIGHT_SYNC_DIR)
        sync_dir.mkdir(parents=True, exist_ok=True)

        try:
            from safetensors.torch import save_file
        except ImportError:
            ckpt_path = sync_dir / "model_weights.pt"
            torch.save(state_dict, ckpt_path)
            self._weight_dir = str(sync_dir)
            logger.info("AtomEngine.update_weights: saved %d tensors to %s", len(state_dict), ckpt_path)
            return

        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            tensors[name] = tensor.contiguous().cpu()

        save_path = sync_dir / "model.safetensors"
        save_file(tensors, str(save_path))
        self._weight_dir = str(sync_dir)
        logger.info("AtomEngine.update_weights: saved %d tensors to %s", len(tensors), save_path)

    def sleep(self) -> None:
        """Kill the vLLM subprocess to free all GPU memory for training.

        On ROCm, ``del llm`` within the subprocess doesn't release all
        GPU memory because the HIP context retains it.  Terminating
        the process is the only reliable way to free everything.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send_cmd({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except Exception:
                pass
        self._proc = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("AtomEngine: sleep complete (subprocess terminated).")

    def wake(self) -> None:
        """Start a fresh vLLM subprocess for generation."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("AtomEngine: already awake.")
            return
        model_path = self._weight_dir or self._model_name
        self._start_worker(model_path)
        if self._weight_dir is not None:
            self._weight_dir = None
        logger.info("AtomEngine: wake complete (fresh subprocess).")

    def shutdown(self) -> None:
        """Release all resources permanently."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_cmd({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except Exception:
                pass
        for p in [self._cmd_fifo, self._resp_fifo]:
            try:
                if p:
                    os.unlink(p)
            except Exception:
                pass
        self._proc = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        self._weight_dir = None
        logger.info("AtomEngine: shutdown complete.")
