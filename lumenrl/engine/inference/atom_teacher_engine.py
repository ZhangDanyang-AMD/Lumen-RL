"""ATOM-based teacher inference engine for draft model distillation.

Runs the teacher model in a **separate subprocess** on dedicated GPUs
with tensor parallelism and optional MXFP4/FP8 quantization.  Designed
for the 4+4 GPU split strategy:

- GPUs 0-3: ATOM subprocess, TP=4, teacher model (frozen, forward-only)
- GPUs 4-7: torchrun training ranks, FSDP2, draft model training

Hidden states are transferred via **MORI-IO P2P RDMA** — the teacher
subprocess writes forward-pass outputs to pre-registered GPU buffers,
and the training process pulls them directly to its own GPU via
``session.read()``.  No CPU round-trip, no ``torch.save``/``torch.load``.

The subprocess directly instantiates ATOM's ``Config`` + model class
for pure prefill forward passes -- it **does not** go through
``LLMEngine`` (no scheduler, no KV cache, no generation loop).
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
import time
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_HIDDEN_XFER_DIR = os.environ.get(
    "LUMENRL_TEACHER_HIDDEN_DIR",
    "/dev/shm/lumenrl_teacher_hidden",
)

# ---------------------------------------------------------------------------
# Worker subprocess script
#
# This runs in a separate process on the teacher GPUs.  It loads the model
# via ATOM's infrastructure (Config -> model class -> load_model) which
# handles TP, MXFP4/FP8 quantization, and expert fusion automatically.
#
# Communication:
# - Named FIFOs for JSON commands + responses (small control messages)
# - MORI-IO P2P RDMA for GPU-to-GPU hidden state transfer (large tensors)
# - /dev/shm for input_ids (small, a few KB per batch)
# ---------------------------------------------------------------------------

_TEACHER_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging, time
import numpy as np

# ---- Isolate from torchrun environment ----
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
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
logger = logging.getLogger("atom_teacher")

import torch

# Parse arguments
cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
tp_size = int(sys.argv[4])
hidden_dir = sys.argv[5]
mori_host = sys.argv[6]
mori_port = int(sys.argv[7])
mori_qp = int(sys.argv[8])
max_batch = int(sys.argv[9])
max_seq = int(sys.argv[10])

os.makedirs(hidden_dir, exist_ok=True)

# ---- Load model via ATOM infrastructure ----
from atom.config import Config
from atom.model_loader.loader import load_model
from atom.utils import resolve_obj_by_qualname

# Import support_model_arch_dict to resolve model class
from atom.model_engine.model_runner import support_model_arch_dict

config = Config(
    model_path,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    trust_remote_code=True,
    max_num_batched_tokens=32768,
    max_num_seqs=64,
)

hf_config = config.hf_config
arch = hf_config.architectures[0]
if arch not in support_model_arch_dict:
    raise ValueError(f"Unsupported architecture: {arch}")

model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])

# Remap quantization layers before model construction
config.quant_config.remap_layer_name(
    config.hf_config,
    packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
    quant_exclude_name_mapping=getattr(model_class, "quant_exclude_name_mapping", {}),
)

# For TP > 1, we need distributed init.  For TP = 1, run directly.
if tp_size > 1:
    from atom.utils import init_dist_env, get_distributed_init_method
    init_method = get_distributed_init_method(config.master_addr, config.port)
    init_dist_env(tp_size, rankID=0, backend="nccl",
                  distributed_init_method=init_method)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.set_default_dtype(config.torch_dtype)
torch.set_default_device(device)

logger.info(f"Loading teacher model: {model_path} (tp={tp_size})")
model = model_class(config)

fused_fn = None
if hasattr(model, "load_fused_expert_weights"):
    fused_fn = model.load_fused_expert_weights

torch.set_default_device(None)
load_model(model, config.model, config.hf_config, config.load_dummy,
           load_fused_expert_weights_fn=fused_fn)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

logger.info(f"Teacher model loaded: {arch}")

# Determine hidden dim from model
D = None
for name, param in model.named_parameters():
    if "lm_head" in name and "weight" in name:
        D = param.shape[1]
        break
if D is None:
    raise RuntimeError("Could not determine hidden_dim from lm_head.weight")

# Extract lm_head weight and save to shared memory
lm_head_weight = None
for name, param in model.named_parameters():
    if "lm_head" in name and "weight" in name:
        lm_head_weight = param.detach().cpu()
        break

if lm_head_weight is not None:
    lm_head_path = os.path.join(hidden_dir, "lm_head_weight.pt")
    torch.save(lm_head_weight, lm_head_path)
    logger.info(f"lm_head.weight saved: {lm_head_weight.shape}")

# ---- Initialize MORI-IO (target role) ----
from mori.io import (
    IOEngine, IOEngineConfig, BackendType, RdmaBackendConfig,
    EngineDesc, MemoryDesc,
)

mori_config = IOEngineConfig(host=mori_host, port=mori_port)
mori_engine = IOEngine(key="teacher_worker", config=mori_config)
mori_engine.create_backend(BackendType.RDMA, RdmaBackendConfig(
    qp_per_transfer=mori_qp,
))

# Pre-allocate GPU buffers for hidden state transfer
max_elems = max_batch * max_seq * D
hidden_buf = torch.zeros(max_elems, dtype=torch.bfloat16, device=device)
embed_buf = torch.zeros(max_elems, dtype=torch.bfloat16, device=device)

hidden_mem = mori_engine.register_torch_tensor(hidden_buf)
embed_mem = mori_engine.register_torch_tensor(embed_buf)

logger.info(
    f"MORI-IO initialized: max_elems={max_elems}, "
    f"buf_size={max_elems * 2 / 1024**2:.1f} MB each"
)

# ---- Signal ready with MORI-IO metadata ----
resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({
    "status": "ready",
    "mori_engine_desc": mori_engine.get_engine_desc().pack().hex(),
    "mori_hidden_mem": hidden_mem.pack().hex(),
    "mori_embed_mem": embed_mem.pack().hex(),
    "hidden_dim": D,
    "max_batch": max_batch,
    "max_seq": max_seq,
}) + "\\n")
resp_f.flush()

cmd_f = open(cmd_fifo, "r")

# ---- Command loop ----
for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "register_engine":
        # Register the reader's (training rank 0) engine for RDMA
        reader_desc_bytes = bytes.fromhex(msg["engine_desc"])
        reader_desc = EngineDesc.unpack(reader_desc_bytes)
        mori_engine.register_remote_engine(reader_desc)
        logger.info("Registered reader engine for RDMA")
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "extract_hidden":
        input_path = msg["input_path"]

        data = torch.load(input_path, map_location="cpu", weights_only=True)
        input_ids = data["input_ids"].to(device)

        B, T = input_ids.shape
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.reshape(-1)
        flat_ids = input_ids.reshape(-1)

        with torch.no_grad():
            # Get input embeddings
            token_embeds = model.model.get_input_embeddings(flat_ids)

            # Forward pass through the model
            hidden_states = model.model(flat_ids, positions)

            # Handle aux_hidden_states return
            if isinstance(hidden_states, tuple):
                hidden_states, aux = hidden_states
            else:
                aux = []

        # Write directly into pre-registered GPU buffers (no CPU copy)
        actual_elems = B * T * D
        hidden_buf[:actual_elems].copy_(hidden_states.reshape(-1))
        embed_buf[:actual_elems].copy_(token_embeds.reshape(-1)[:actual_elems])

        # Synchronize to ensure writes are visible for RDMA read
        torch.cuda.synchronize(device)

        resp_f.write(json.dumps({
            "status": "ok", "B": B, "T": T, "D": D,
        }) + "\\n")
        resp_f.flush()

        del data, input_ids, positions, flat_ids, hidden_states, token_embeds
        torch.cuda.empty_cache()

    elif cmd == "shutdown":
        # Cleanup MORI-IO
        try:
            mori_engine.deregister_memory(hidden_mem)
            mori_engine.deregister_memory(embed_mem)
        except Exception:
            pass
        try:
            del model
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


class AtomTeacherEngine:
    """ATOM-based teacher inference for draft model distillation.

    Runs the teacher model in a separate subprocess on dedicated GPUs,
    with tensor parallelism and MXFP4/FP8 quantization.  Hidden states
    are transferred via MORI-IO P2P RDMA (GPU Direct).

    Communication protocol:
    - Named FIFOs for JSON commands (small control messages)
    - MORI-IO ``session.read()`` for GPU-to-GPU tensor transfer
    - ``/dev/shm`` files for ``input_ids`` (a few KB, not worth RDMA)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        gpu_ids: list[int] | None = None,
        *,
        mori_io_host: str = "127.0.0.1",
        mori_io_port: int = 0,
        mori_io_qp_per_transfer: int = 2,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        local_device: torch.device | None = None,
    ) -> None:
        self._model_name = model_name
        self._tp_size = tensor_parallel_size
        self._gpu_ids = gpu_ids or list(range(tensor_parallel_size))
        self._hidden_dir = _HIDDEN_XFER_DIR
        self._mori_host = mori_io_host
        self._mori_port = mori_io_port
        self._mori_qp = mori_io_qp_per_transfer
        self._max_batch = max_batch_size
        self._max_seq = max_seq_len
        self._local_device = local_device or torch.device("cuda:0")

        self._proc: subprocess.Popen | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        self._req_counter = 0

        # MORI-IO state
        self._mori_engine = None
        self._hidden_session = None
        self._embed_session = None
        self._hidden_buf: torch.Tensor | None = None
        self._embed_buf: torch.Tensor | None = None
        self._local_hidden_mem = None
        self._local_embed_mem = None
        self._hidden_dim: int = 0

    @property
    def is_alive(self) -> bool:
        return (
            self._initialized
            and self._proc is not None
            and self._proc.poll() is None
        )

    def start(self) -> None:
        """Start the teacher worker subprocess and set up MORI-IO RDMA."""
        if self.is_alive:
            return

        os.makedirs(self._hidden_dir, exist_ok=True)

        fifo_dir = tempfile.mkdtemp(prefix="lumenrl_teacher_fifo_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        # Build environment: isolate from torchrun, set teacher GPUs
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self._gpu_ids)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        for key in list(env.keys()):
            if any(key.startswith(p) for p in [
                "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
                "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
                "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
                "OMP_NUM_THREADS",
            ]):
                del env[key]

        attn_backend = os.environ.get("VLLM_ROCM_ATTN_BACKEND")
        if attn_backend:
            env["VLLM_ROCM_ATTN_BACKEND"] = attn_backend

        logger.info(
            "AtomTeacherEngine: starting teacher worker for %s "
            "(tp=%d, gpus=%s, mori=%s:%d)",
            self._model_name, self._tp_size, self._gpu_ids,
            self._mori_host, self._mori_port,
        )

        self._proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", _TEACHER_WORKER_SCRIPT,
                self._cmd_fifo, self._resp_fifo,
                self._model_name, str(self._tp_size),
                self._hidden_dir,
                self._mori_host, str(self._mori_port), str(self._mori_qp),
                str(self._max_batch), str(self._max_seq),
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
            raise RuntimeError(
                "Teacher worker subprocess exited before ready. "
                "Check stderr for model loading errors."
            )
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"Teacher worker failed to start: {resp}")

        self._cmd_f = open(self._cmd_fifo, "w")

        # ---- Set up MORI-IO (initiator / reader role) ----
        from mori.io import (
            IOEngine, IOEngineConfig, BackendType, RdmaBackendConfig,
            EngineDesc, MemoryDesc,
        )

        self._hidden_dim = resp["hidden_dim"]
        max_elems = self._max_batch * self._max_seq * self._hidden_dim

        # Parse remote (worker) MORI-IO metadata
        remote_engine_desc = EngineDesc.unpack(
            bytes.fromhex(resp["mori_engine_desc"])
        )
        remote_hidden_mem = MemoryDesc.unpack(
            bytes.fromhex(resp["mori_hidden_mem"])
        )
        remote_embed_mem = MemoryDesc.unpack(
            bytes.fromhex(resp["mori_embed_mem"])
        )

        # Initialize local MORI-IO engine
        local_config = IOEngineConfig(host=self._mori_host, port=0)
        self._mori_engine = IOEngine(key="teacher_reader", config=local_config)
        self._mori_engine.create_backend(
            BackendType.RDMA, RdmaBackendConfig(qp_per_transfer=self._mori_qp)
        )

        # Exchange engine descriptors
        local_desc = self._mori_engine.get_engine_desc()
        self._mori_engine.register_remote_engine(remote_engine_desc)

        # Tell worker to register our engine
        resp2 = self._send_cmd({
            "cmd": "register_engine",
            "engine_desc": local_desc.pack().hex(),
        })
        if resp2.get("status") != "ok":
            raise RuntimeError(f"Failed to register reader engine: {resp2}")

        # Pre-allocate local GPU buffers on training device
        self._hidden_buf = torch.zeros(
            max_elems, dtype=torch.bfloat16, device=self._local_device,
        )
        self._embed_buf = torch.zeros(
            max_elems, dtype=torch.bfloat16, device=self._local_device,
        )
        self._local_hidden_mem = self._mori_engine.register_torch_tensor(
            self._hidden_buf
        )
        self._local_embed_mem = self._mori_engine.register_torch_tensor(
            self._embed_buf
        )

        # Create RDMA sessions (local <- remote)
        self._hidden_session = self._mori_engine.create_session(
            self._local_hidden_mem, remote_hidden_mem,
        )
        self._embed_session = self._mori_engine.create_session(
            self._local_embed_mem, remote_embed_mem,
        )

        if self._hidden_session is None or self._embed_session is None:
            raise RuntimeError(
                "Failed to create MORI-IO RDMA sessions. "
                "Check RDMA backend configuration."
            )

        self._initialized = True
        logger.info(
            "AtomTeacherEngine: teacher worker ready (pid=%d, hidden_dim=%d, "
            "MORI-IO RDMA sessions established).",
            self._proc.pid, self._hidden_dim,
        )

    def _send_cmd(self, cmd: dict) -> dict:
        """Send a JSON command and read the response."""
        if not self.is_alive:
            raise RuntimeError("Teacher worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("Teacher worker closed response FIFO")
        return json.loads(resp_line)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run teacher forward pass, return hidden states on training GPU.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` mask (1 = valid, 0 = pad).

        Returns:
            Dict with ``hidden_states`` ``[B, T, D]``,
            ``token_embeds`` ``[B, T, D]``, and ``input_ids``.
            All tensors are on ``self._local_device`` (training GPU).
        """
        if not self.is_alive:
            self.start()

        self._req_counter += 1
        tag = f"req_{self._req_counter}"
        input_path = os.path.join(self._hidden_dir, f"{tag}_input.pt")

        # Save input_ids to shared memory (small, a few KB)
        torch.save(
            {"input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu()},
            input_path,
        )

        # Tell worker to run forward pass
        resp = self._send_cmd({
            "cmd": "extract_hidden",
            "input_path": input_path,
        })

        if resp.get("status") != "ok":
            raise RuntimeError(f"extract_hidden failed: {resp}")

        B = resp["B"]
        T = resp["T"]
        D = resp["D"]
        actual_bytes = B * T * D * 2  # bfloat16 = 2 bytes

        # MORI-IO RDMA read: pull hidden_states from worker GPU to local GPU
        uid_h = self._hidden_session.allocate_transfer_uid()
        status_h = self._hidden_session.read(0, 0, actual_bytes, uid_h)
        uid_e = self._embed_session.allocate_transfer_uid()
        status_e = self._embed_session.read(0, 0, actual_bytes, uid_e)

        # Wait for both transfers to complete
        status_h.Wait()
        status_e.Wait()

        if not status_h.Succeeded():
            raise RuntimeError(
                f"MORI-IO hidden_states read failed: {status_h.Message()}"
            )
        if not status_e.Succeeded():
            raise RuntimeError(
                f"MORI-IO token_embeds read failed: {status_e.Message()}"
            )

        # Cleanup input file
        try:
            os.unlink(input_path)
        except OSError:
            pass

        # Return views of local GPU buffers (zero-copy)
        return {
            "hidden_states": self._hidden_buf[:B * T * D].reshape(B, T, D),
            "token_embeds": self._embed_buf[:B * T * D].reshape(B, T, D),
            "input_ids": input_ids,
        }

    def get_lm_head_weight(self) -> torch.Tensor:
        """Load the teacher's lm_head.weight from shared memory.

        Must be called after ``start()`` -- the worker saves lm_head
        weight during model loading.  This is a one-time transfer
        (small enough that /dev/shm is fine).
        """
        if not self.is_alive:
            self.start()

        path = os.path.join(self._hidden_dir, "lm_head_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=True)
            time.sleep(0.5)
        raise FileNotFoundError(
            f"lm_head_weight.pt not found at {path} after 15s. "
            "Teacher worker may have failed to load the model."
        )

    def shutdown(self) -> None:
        """Terminate the teacher worker subprocess and clean up MORI-IO."""
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

        # Clean up MORI-IO resources
        if self._mori_engine is not None:
            try:
                if self._local_hidden_mem is not None:
                    self._mori_engine.deregister_memory(self._local_hidden_mem)
                if self._local_embed_mem is not None:
                    self._mori_engine.deregister_memory(self._local_embed_mem)
            except Exception:
                pass
        self._mori_engine = None
        self._hidden_session = None
        self._embed_session = None
        self._hidden_buf = None
        self._embed_buf = None
        self._local_hidden_mem = None
        self._local_embed_mem = None

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
        logger.info("AtomTeacherEngine: shutdown complete.")

    def __del__(self) -> None:
        self.shutdown()


__all__ = ["AtomTeacherEngine"]
