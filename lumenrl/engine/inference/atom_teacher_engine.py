"""ATOM-based teacher inference engine for speculative distillation.

Runs the teacher in a **separate subprocess** using ATOM's ``AsyncLLMEngine``,
which spawns one model-runner process per tensor-parallel rank internally. The
auxiliary hidden states the draft model trains on are captured with
``register_forward_hook`` on the selected decoder layers and shipped over
**Mooncake TCP** — the same transport ``VllmTeacherEngine`` uses.

Because capture rides on hooks rather than on a speculative-decoding config, one
engine serves both on-policy sweeps:

1. ``generate_tokens()`` — decode each prompt's continuation, submitting requests
   with no external id. ATOM keys hidden-state writes on that id, so withholding
   it parks capture and this is the stock ATOM decode path.
2. ``extract_hidden_states()`` — prefill the finished sequences with data ids
   supplied, writing aux + last hidden states to Mooncake.

Switching between them costs nothing: capture is decided per request, so there is
no mode to set and no restart. That is the reason this engine exists — the vLLM
path needed a separate engine per sweep and reloaded K3's 1.5 TB of weights on
every switch, and its K3 decode kernels faulted the GPU well before a 50-batch
round finished.

Which layers are captured comes from the training config and must match the
draft model's feature contract; nothing here infers them from depth.

Expects an ATOM build exposing ``atom.rollout`` (``AsyncLLMEngine``,
``RLHFModelRunner``), plus a ``torchspec`` module providing ``MooncakeConfig``
and ``EagleMooncakeStore`` — see this example's Dockerfile, which maps that name
onto ``lumenrl.transfer``.
"""

from __future__ import annotations

import json
import logging
import os
import errno
import select
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _pad_stack(tensors: list[torch.Tensor], length: int) -> torch.Tensor:
    """Right-pad each tensor's first dim to ``length``, then stack into a batch."""
    padded = []
    for t in tensors:
        if t.shape[0] > length:
            raise ValueError(
                f"sequence of {t.shape[0]} positions exceeds the batch width "
                f"{length}; hidden states would have to be truncated",
            )
        if t.shape[0] < length:
            pad = torch.zeros(
                length - t.shape[0], *t.shape[1:],
                dtype=t.dtype, device=t.device,
            )
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return torch.stack(padded)


_HIDDEN_XFER_DIR = os.environ.get(
    "LUMENRL_TEACHER_HIDDEN_DIR",
    "/dev/shm/lumenrl_teacher_hidden",
)
_READY_TIMEOUT_SECONDS = float(
    os.environ.get("LUMENRL_TEACHER_READY_TIMEOUT_SECONDS", "600"),
)
_CMD_FIFO_OPEN_TIMEOUT_SECONDS = 10.0
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0
# Decoding is the longest-running command by far, so it gets its own budget
# instead of tripping the control-message timeout. A fixed budget does not
# survive sweeps that merge batches, so the budget is derived from the work the
# sweep could do; this env var overrides it outright when set.
_GENERATE_TIMEOUT_OVERRIDE = os.environ.get(
    "LUMENRL_TEACHER_GENERATE_TIMEOUT_SECONDS"
)
_GENERATE_TIMEOUT_FLOOR_SECONDS = 3600.0
# Measured on K3 at TP=8, B=64, max_tokens=1024, 768 prompts merged into one
# sweep with CUDA graphs on: 515.7 s, i.e. 0.66 ms per prompt-token. 4x that
# absorbs a colder cache and the eager fallback (which measured ~1.4 ms) while
# still catching a hung engine.
#
# The earlier 9.1e-3 was calibrated against the batch-serial eager path and is
# now 14x the real cost: a whole-round sweep would have sat for 8 hours before
# reporting a hang. Recalibrate this whenever the decode path gets faster --
# a timeout that never fires is not a safe default, it is a silent one.
_GENERATE_SECONDS_PER_PROMPT_TOKEN = 2.6e-3


def _generate_timeout_seconds(num_prompts: int, max_tokens: int) -> float:
    """Wall-clock budget for one decode sweep."""
    if _GENERATE_TIMEOUT_OVERRIDE:
        return float(_GENERATE_TIMEOUT_OVERRIDE)
    return max(
        _GENERATE_TIMEOUT_FLOOR_SECONDS,
        _GENERATE_SECONDS_PER_PROMPT_TOKEN * num_prompts * max_tokens,
    )
_SHUTDOWN_COMMAND_TIMEOUT_SECONDS = 5.0
_WORKER_TERMINATE_GRACE_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Worker subprocess script
#
# Runs in a separate process on the teacher GPUs.  Uses ATOM's
# AsyncLLMEngine which internally spawns tp_size ModelRunner processes
# via AsyncIOProcManager, handling NCCL distributed init correctly.
#
# Communication:
# - Named FIFOs for JSON commands + responses (small control messages)
# - Mooncake TCP for hidden state transfer (via RLHFModelRunner)
# - /dev/shm for input_ids (small, a few KB per batch)
# - /dev/shm for teacher weights (lm_head, embed, norm)
# ---------------------------------------------------------------------------

_TEACHER_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging, time, socket, glob

# ---- Ensure lumenrl is importable (for ATOM's fallback imports) ----
# third_party/ATOM is deliberately NOT on the path: the image installs a newer
# ATOM at /app/ATOM, and the checked-in copy would shadow it.
for _p in ["/root/lumenrl", os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

# Suppress noisy logs from mooncake C++ (glog) and aiter
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_v"] = "0"
os.environ["MOONCAKE_LOG_LEVEL"] = "FATAL"
os.environ["AITER_LOG_LEVEL"] = "WARNING"

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("atom_teacher")

import torch
from transformers import AutoConfig

# Parse arguments
cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
tp_size = int(sys.argv[4])
hidden_dir = sys.argv[5]
max_batch = int(sys.argv[6])
max_seq = int(sys.argv[7])
atom_args_json = sys.argv[8] if len(sys.argv) > 8 else "{}"
atom_extra = json.loads(atom_args_json)
aux_ids_json = sys.argv[9] if len(sys.argv) > 9 else "[]"
aux_layer_ids_arg = json.loads(aux_ids_json)
start_mode = sys.argv[10] if len(sys.argv) > 10 else "extract"

os.makedirs(hidden_dir, exist_ok=True)

# ---- Extract weights from checkpoint (CPU only, before engine) ----
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
hf_text = getattr(hf_config, "text_config", hf_config)
num_layers = hf_text.num_hidden_layers
hidden_dim = hf_text.hidden_size
norm_eps = getattr(hf_text, "rms_norm_eps", 1e-6)

from safetensors.torch import load_file as st_load
st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

lm_head_weight = embed_weight = norm_weight = None

for sf in st_files:
    tensors = st_load(sf, device="cpu")
    for name, tensor in tensors.items():
        if "lm_head" in name and "weight" in name and lm_head_weight is None:
            lm_head_weight = tensor
        if "embed_tokens" in name and "weight" in name and embed_weight is None:
            embed_weight = tensor
        if norm_weight is None and "norm.weight" in name and "layer" not in name:
            if any(p in name for p in ["model.norm.weight",
                                        "language_model.model.norm.weight"]):
                norm_weight = tensor
    del tensors
    if all(w is not None for w in [lm_head_weight, embed_weight, norm_weight]):
        break

if lm_head_weight is None or embed_weight is None:
    raise RuntimeError(
        f"Could not find lm_head/embed_tokens weights in {model_path}. "
        f"Searched {len(st_files)} safetensors files."
    )

torch.save(lm_head_weight, os.path.join(hidden_dir, "lm_head_weight.pt"))
torch.save(embed_weight, os.path.join(hidden_dir, "embed_weight.pt"))
if norm_weight is not None:
    torch.save({"weight": norm_weight, "eps": norm_eps},
               os.path.join(hidden_dir, "norm_weight.pt"))
logger.info(
    "Weights saved: lm_head=%s, embed=%s, norm=%s",
    lm_head_weight.shape, embed_weight.shape,
    norm_weight.shape if norm_weight is not None else "N/A",
)
del lm_head_weight, embed_weight, norm_weight

# ---- Create AsyncLLMEngine (handles TP internally) ----
from atom.rollout.async_engine import AsyncLLMEngine
logging.getLogger("atom").setLevel(logging.WARNING)

engine_kwargs = dict(
    tensor_parallel_size=tp_size,
    enforce_eager=atom_extra.pop("enforce_eager", True),
    trust_remote_code=atom_extra.pop("trust_remote_code", True),
    max_num_batched_tokens=atom_extra.pop("max_num_batched_tokens", 32768),
    max_num_seqs=atom_extra.pop("max_num_seqs", 64),
    # LumenRL's runner adds a capture toggle and a rank-0 write guard.
    # AsyncLLMEngine only setdefaults this, so passing it wins.
    runner_qualname=atom_extra.pop(
        "runner_qualname",
        "lumenrl.engine.inference.atom_runner_ext.LumenRLModelRunner",
    ),
)
for k, v in atom_extra.items():
    engine_kwargs[k] = v

# ATOM filters kwargs against its Config fields and drops the rest without a
# word, so a mistyped tuning knob would silently do nothing. Fail instead.
from dataclasses import fields as _dc_fields
from atom.config import Config as _AtomConfig
_known = {f.name for f in _dc_fields(_AtomConfig)} | {
    "data_parallel_size", "data_parallel_master_port",
}
_unknown = sorted(set(engine_kwargs) - _known)
if _unknown:
    raise ValueError(
        f"ATOM would ignore these engine settings: {_unknown}. "
        f"They are not fields of atom.config.Config -- check for typos."
    )

logger.info("Creating AsyncLLMEngine: %s", engine_kwargs)
engine = AsyncLLMEngine(model_path, **engine_kwargs)
logger.info("AsyncLLMEngine created successfully")

# ---- Configure hidden states extraction ----
# Layer ids come from the training config: the draft model's feature contract
# fixes both how many aux layers it consumes and which ones. Falling back to
# Eagle3's 3-layer heuristic would silently produce features of the wrong width.
if aux_layer_ids_arg:
    aux_layer_ids = sorted(int(i) for i in aux_layer_ids_arg)
    out_of_range = [i for i in aux_layer_ids if i >= num_layers or i < 0]
    if out_of_range:
        raise ValueError(
            f"aux_hidden_state_layer_ids {out_of_range} are outside the "
            f"teacher's {num_layers} layers; ATOM silently skips hooks for "
            f"such layers, which would yield too few aux features"
        )
else:
    aux_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
    logger.warning(
        "No aux_hidden_state_layer_ids configured; falling back to the Eagle3 "
        "3-layer heuristic %s", aux_layer_ids,
    )

max_seq = engine_kwargs.get("max_model_len", max_seq)

# ATOM writes a request's hidden states once per scheduler step, covering only
# the tokens scheduled in that step, under a key that a later write overwrites.
# It does not consult is_final_chunk. So anything that stops a prefill from
# computing the whole sequence in one step silently truncates the features, and
# the draft model would train on them without complaint. Three ways that happens,
# all of which ATOM enables by default:
batched_token_budget = engine_kwargs.get("max_num_batched_tokens", 0)
if batched_token_budget and batched_token_budget < max_seq:
    raise ValueError(
        f"max_num_batched_tokens={batched_token_budget} is below "
        f"max_seq_len={max_seq}: prefill would be split across steps and each "
        f"step would overwrite the previous hidden states for the same request"
    )
if engine_kwargs.get("enable_chunked_prefill", True):
    raise ValueError(
        "enable_chunked_prefill must be false for hidden-state extraction: a "
        "chunked prefill writes each chunk under the same Mooncake key, so only "
        "the final chunk's hidden states survive"
    )
if engine_kwargs.get("enable_prefix_caching", True):
    raise ValueError(
        "enable_prefix_caching must be false for hidden-state extraction: a "
        "cached prefix is excluded from the scheduled tokens, so the prompt's "
        "hidden states would be missing entirely from the stored features"
    )

from lumenrl.transfer.eagle_mooncake_store import calculate_eagle3_buffer_size
# Sized for one sequence, not the whole batch: hidden states are put per
# request, so a batch-sized buffer would idle tens of GB of pinned host memory.
# num_aux_layers must be passed explicitly — it defaults to Eagle3's 3, which
# would under-size the buffer for K3's 5-layer contract.
host_buf_size = calculate_eagle3_buffer_size(
    max_seq_len=max_seq, batch_size=1,
    hidden_dim=hidden_dim,
    num_aux_layers=len(aux_layer_ids),
    safety_margin=2.0,
)
mooncake_config = {
    "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
    "metadata_server": os.environ.get("MOONCAKE_METADATA_SERVER", ""),
    "master_server_address": os.environ.get("MOONCAKE_MASTER_SERVER", ""),
    "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
    "device_name": os.environ.get("MOONCAKE_DEVICE_NAME", ""),
    "global_segment_size": int(os.environ.get(
        "MOONCAKE_GLOBAL_SEGMENT_SIZE", str(16 * 1024**3))),
    "local_buffer_size": int(os.environ.get(
        "MOONCAKE_LOCAL_BUFFER_SIZE", str(4 * 1024**3))),
    "enable_gpu_direct": os.environ.get("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1",
    "enable_hard_pin": os.environ.get("MOONCAKE_ENABLE_HARD_PIN", "0") == "1",
    "max_seq_len": max_seq,
    "hidden_dim": hidden_dim,
    "host_buffer_size": host_buf_size,
}
# ATOM captures the post-layer residual stream (hidden_states + residual) via
# forward hooks and has no capture_mode knob — the semantics are fixed and match
# "postnorm". Passing one would be a TypeError.
engine.configure_hidden_states(aux_layer_ids, mooncake_config)
logger.info("Hidden states configured: aux_layers=%s", aux_layer_ids)

# Capture needs no arm/park control message. ATOM keys every Mooncake write on a
# request's external id, and LumenRLModelRunner additionally skips the capture
# forward path for batches where no request carries one. So the sweep selects its
# own behaviour purely by whether it submits data ids:
#   extract sweep  -> generate_hidden_states(rows, data_ids)  -> captures, writes
#   generate sweep -> preprocess() with no request_id         -> stock decode
# Both share this one engine; "mode" is bookkeeping on the client side only.
logger.info("Worker ready (start mode %s, capture is per-request)", start_mode)

# ---- Signal ready ----
resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({
    "status": "ready",
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "aux_layer_indices": aux_layer_ids,
    "mode": start_mode,
}) + "\\n")
resp_f.flush()

cmd_f = open(cmd_fifo, "r")
req_counter = 0

# ---- Command loop ----
for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "extract_hidden":
        input_path = msg["input_path"]
        data = torch.load(input_path, map_location="cpu", weights_only=True)
        input_ids_batch = data["input_ids"]  # [B, T], right-padded
        B, T = input_ids_batch.shape
        req_counter += 1

        # Trim padding per row. On-policy batches are ragged, and prefilling pad
        # tokens would both waste compute and store hidden states for positions
        # the loss masks out anyway.
        lengths = data.get("lengths")
        rows = []
        for i in range(B):
            if lengths is not None:
                n = int(lengths[i])
            else:
                n = T
            n = max(1, min(n, T))
            rows.append(input_ids_batch[i][:n].tolist())

        key_prefix = os.environ.get("LUMENRL_ATOM_KEY_PREFIX", f"atom_{os.getpid()}")
        data_ids = [f"{key_prefix}_x{req_counter}_{i}" for i in range(B)]

        t0 = time.monotonic()
        engine.generate_hidden_states(rows, data_ids)
        elapsed = time.monotonic() - t0
        logger.info(
            "extract_hidden: B=%d, T=%d, tokens=%d, %.2fs",
            B, T, sum(len(r) for r in rows), elapsed,
        )

        resp_f.write(json.dumps({
            "status": "ok", "B": B, "T": T, "D": hidden_dim,
            "mooncake_keys": data_ids,
            "seq_lens": {k: len(r) for k, r in zip(data_ids, rows)},
        }) + "\\n")
        resp_f.flush()
        del data, input_ids_batch, rows

    elif cmd == "generate_tokens":
        input_path = msg["input_path"]
        output_path = msg["output_path"]
        max_tokens = msg.get("max_tokens", 2048)
        temperature = msg.get("temperature", 0.0)
        data = torch.load(input_path, map_location="cpu", weights_only=True)
        prompt_batch = data["input_ids"]  # [B, T_prompt], right-padded
        lengths = data.get("lengths")
        B, T_prompt = prompt_batch.shape
        req_counter += 1

        prompts = []
        for i in range(B):
            n = int(lengths[i]) if lengths is not None else T_prompt
            n = max(1, min(n, T_prompt))
            prompts.append(prompt_batch[i][:n].tolist())

        from atom.sampling_params import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        t0 = time.monotonic()
        engine.core_mgr.reset_dp_router()

        # Submit one sequence at a time instead of through add_request() so the
        # engine's own Sequence objects are in hand: their ids give an exact
        # row mapping. ATOM's generate() instead sorts finished sequences by
        # internal id and zips against input order, which assumes ids sort in
        # submission order — pairing a prompt with another row's response would
        # be silent training-data corruption.
        #
        # Deliberately no request_id: an external id is what tells ATOM where to
        # write hidden states, so withholding it is what keeps this sweep from
        # pushing the prompt prefill to Mooncake (gigabytes per batch that
        # nobody reads).
        io_proc = engine.io_processor
        submitted = [io_proc.preprocess(p, sp) for p in prompts]
        row_of_seq = {seq.id: i for i, seq in enumerate(submitted)}
        if len(row_of_seq) != B:
            raise RuntimeError(
                f"generate_tokens: engine assigned duplicate sequence ids to "
                f"{B} prompts ({len(row_of_seq)} unique); row mapping is unsafe"
            )
        engine.core_mgr.add_request(submitted)

        by_row = {}
        while not engine.is_finished() and (
            engine.core_mgr.is_alive() or engine.core_mgr.is_rest()
        ):
            seqs = engine.step()
            for internal_id, out in io_proc.postprocess(seqs).items():
                row = row_of_seq.get(internal_id)
                if row is None:
                    raise RuntimeError(
                        f"generate_tokens: engine returned sequence {internal_id} "
                        f"that this sweep never submitted"
                    )
                by_row[row] = out
        elapsed = time.monotonic() - t0

        missing = [i for i in range(B) if i not in by_row]
        if missing:
            raise RuntimeError(
                f"generate_tokens: {len(missing)}/{B} requests never returned "
                f"(first missing row: {missing[0]})"
            )

        completions = []
        finish_reasons = []
        for i in range(B):
            out = by_row[i]
            completions.append([int(t) for t in out["token_ids"]])
            finish_reasons.append(str(out.get("finish_reason", "")))
            reported_prompt_len = out.get("num_tokens_input")
            if reported_prompt_len is not None and int(reported_prompt_len) != len(prompts[i]):
                raise RuntimeError(
                    f"generate_tokens: row {i} prompt length mismatch "
                    f"(sent {len(prompts[i])}, engine saw {reported_prompt_len}); "
                    f"prompt/response pairing cannot be trusted"
                )

        gen_lens = [len(c) for c in completions]
        logger.info(
            "generate_tokens: B=%d, max_tokens=%d, gen_len min/mean/max=%d/%.1f/%d, %.2fs",
            B, max_tokens, min(gen_lens), sum(gen_lens) / len(gen_lens),
            max(gen_lens), elapsed,
        )

        # Token ids go through a file rather than the FIFO: B=64 x 512 tokens of
        # JSON would exceed the pipe buffer and deadlock against a reader that is
        # waiting for one line.
        torch.save(
            {
                "prompt_lens": [len(p) for p in prompts],
                "completions": completions,
                "finish_reasons": finish_reasons,
            },
            output_path,
        )

        resp_f.write(json.dumps({
            "status": "ok", "B": B, "output_path": output_path,
        }) + "\\n")
        resp_f.flush()
        del data, prompt_batch, prompts, completions

    elif cmd == "shutdown":
        try:
            engine.shutdown()
        except Exception:
            pass
        gc.collect()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class AtomTeacherEngine:
    """ATOM-based teacher inference for Eagle3 speculative distillation.

    Runs the teacher model in a separate subprocess on dedicated GPUs,
    with tensor parallelism and MXFP4/FP8 quantization.  Hidden states
    (3 aux layers + last hidden) are transferred via Mooncake TCP.

    Communication protocol:
    - Named FIFOs for JSON commands (small control messages)
    - Mooncake TCP ``EagleMooncakeStore`` for hidden state transfer
    - ``/dev/shm`` files for ``input_ids`` (a few KB, not worth Mooncake)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        gpu_ids: list[int] | None = None,
        *,
        mooncake_config: Any = None,
        transport: str = "mooncake",
        quantization: str = "",
        atom_config: dict[str, Any] | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        local_device: torch.device | None = None,
        capture_mode: str = "postnorm",
        aux_layer_ids: list[int] | None = None,
        key_prefix: str | None = None,
        consume_hidden_states: bool = True,
    ) -> None:
        self._model_name = model_name
        self._tp_size = tensor_parallel_size
        self._gpu_ids = gpu_ids or list(range(tensor_parallel_size))
        self._hidden_dir = _HIDDEN_XFER_DIR
        self._start_seq = 0
        # Distinguishes one trainer process's worker logs from the next one's. The
        # pid cannot do it: inside the container the worker's pid is deterministic
        # (418 every time), so after a container restart the new process reuses
        # both the pid and the sequence counter and overwrites the logs of the run
        # that crashed -- which is exactly the evidence needed at that point.
        self._run_stamp = time.strftime("%m%d-%H%M%S")
        self._worker_log_path: str | None = None
        self._mooncake_config = mooncake_config
        self._transport = transport
        self._quantization = quantization
        self._atom_config = atom_config or {}
        self._max_batch = max_batch_size
        self._max_seq = max_seq_len
        self._local_device = local_device or torch.device("cuda:0")
        self._capture_mode = capture_mode
        self._configured_aux_layer_ids = list(aux_layer_ids or [])
        self._key_prefix = key_prefix
        self._consume_hidden_states = consume_hidden_states
        self._mode = "extract"

        self._proc: subprocess.Popen | None = None
        self._fifo_dir: str | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        self._req_counter = 0

        self._hidden_dim: int = 0
        self._num_layers: int = 0
        self._cmd_lock = threading.Lock()
        self._aux_layer_indices: list[int] = []
        self._mooncake_store: Any = None

    @property
    def is_alive(self) -> bool:
        return (
            self._initialized
            and self._proc is not None
            and self._proc.poll() is None
        )

    @property
    def _num_aux_layers(self) -> int:
        """How many aux layers the transfer buffers must be sized for.

        Buffer sizing helpers default to Eagle3's 3 layers; K3's contract is 5, so
        every caller has to be explicit or the host buffer comes out too small to
        hold a sequence.
        """
        if self._aux_layer_indices:
            return len(self._aux_layer_indices)
        if self._configured_aux_layer_ids:
            return len(self._configured_aux_layer_ids)
        return 3

    @property
    def mode(self) -> str:
        """Which sweep the engine is set up for: ``"generate"`` or ``"extract"``."""
        return self._mode

    def switch_mode(self, mode: str) -> None:
        """Retarget the engine between sweeps. Free: nothing has to be told.

        ATOM decides per request whether to capture, keyed on whether the request
        carries an external id, so the two sweeps differ only in what they submit
        — the generate sweep withholds the id, the extract sweep supplies it. One
        loaded model serves both, and this is bookkeeping so callers can assert
        which sweep they are in. That is what makes on-policy rounds affordable:
        the vLLM path tore the engine down and reloaded K3's 1.5 TB of weights on
        every switch.
        """
        if mode not in ("generate", "extract"):
            raise ValueError(f"unknown teacher mode: {mode!r}")
        if self._mode != mode:
            logger.info("AtomTeacherEngine: now in %s mode", mode)
        self._mode = mode

    def _describe_worker_exit(self, exit_code: int) -> str:
        if exit_code < 0:
            sig = -exit_code
            signal_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"SIG{sig}"
            return f"exited by signal {signal_name} ({sig})"
        return f"exited with code {exit_code}"

    def worker_log_tail(self, num_lines: int = 50) -> str:
        """Last lines the worker wrote, or "" if they are unavailable.

        The worker's stdout and stderr go only to this file, never to the
        trainer log, so a worker that stops responding leaves no trace in the
        place anyone looks first. ``start`` writes a new file per attempt, so the
        crashed worker's log stays on disk after a restart.
        """
        try:
            path = self._worker_log_path
            if not path or not os.path.exists(path):
                return ""
            with open(path) as f:
                return "".join(f.readlines()[-num_lines:])
        except Exception:
            return ""

    def _terminate_worker(self, reason: str) -> None:
        if self._proc is None:
            return

        if self._proc.poll() is None:
            logger.warning(
                "AtomTeacherEngine: terminating worker pid=%d (%s).",
                self._proc.pid,
                reason,
            )
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.debug("Failed to SIGTERM worker group: %s", exc)
            try:
                self._proc.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "AtomTeacherEngine: worker pid=%d did not exit after %.1fs; sending SIGKILL.",
                    self._proc.pid,
                    _WORKER_TERMINATE_GRACE_SECONDS,
                )
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    logger.debug("Failed to SIGKILL worker group: %s", exc)
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass

    def _cleanup_ipc_resources(self) -> None:
        if hasattr(self, "_worker_log_f") and self._worker_log_f:
            try:
                self._worker_log_f.close()
            except Exception:
                pass
            self._worker_log_f = None
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except Exception:
                pass
        self._cmd_f = None
        self._resp_f = None

        for p in [self._cmd_fifo, self._resp_fifo]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
        self._cmd_fifo = None
        self._resp_fifo = None

        try:
            if self._fifo_dir and os.path.isdir(self._fifo_dir):
                os.rmdir(self._fifo_dir)
        except Exception:
            pass
        self._fifo_dir = None

    def _wait_for_ready(self, timeout_s: float) -> dict[str, Any]:
        if self._resp_fifo is None:
            raise RuntimeError("Response FIFO path is missing")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        fd = os.open(self._resp_fifo, os.O_RDONLY | os.O_NONBLOCK)
        deadline = time.monotonic() + timeout_s
        buf = ""

        try:
            while True:
                poll_code = self._proc.poll()
                if poll_code is not None:
                    raise RuntimeError(
                        f"teacher worker {self._describe_worker_exit(poll_code)} before READY",
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"teacher worker did not send READY within {timeout_s:.1f}s",
                    )

                ready, _, _ = select.select([fd], [], [], min(0.5, remaining))
                if not ready:
                    continue

                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    continue

                buf += chunk.decode("utf-8", errors="replace")
                if "\n" not in buf:
                    continue

                line, _, _ = buf.partition("\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"teacher worker sent invalid READY payload: {line[:200]!r}",
                    ) from exc
        finally:
            os.close(fd)

    def _open_cmd_writer(self, timeout_s: float) -> Any:
        if self._cmd_fifo is None:
            raise RuntimeError("Command FIFO path is missing")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        deadline = time.monotonic() + timeout_s
        while True:
            poll_code = self._proc.poll()
            if poll_code is not None:
                raise RuntimeError(
                    f"teacher worker {self._describe_worker_exit(poll_code)} before command channel setup",
                )
            try:
                fd = os.open(self._cmd_fifo, os.O_WRONLY | os.O_NONBLOCK)
                return os.fdopen(fd, "w")
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"teacher worker did not open command FIFO within {timeout_s:.1f}s",
                    ) from exc
                time.sleep(0.1)

    def _read_response_line(self, *, timeout_s: float, context: str) -> str:
        if self._resp_f is None:
            raise RuntimeError("Response FIFO is not open")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        deadline = time.monotonic() + timeout_s
        resp_fd = self._resp_f.fileno()
        while True:
            poll_code = self._proc.poll()
            if poll_code is not None:
                raise RuntimeError(
                    f"teacher worker {self._describe_worker_exit(poll_code)} while waiting for {context}",
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {context} after {timeout_s:.1f}s",
                )

            ready, _, _ = select.select([resp_fd], [], [], min(0.5, remaining))
            if not ready:
                continue

            resp_line = self._resp_f.readline()
            if not resp_line:
                raise RuntimeError(
                    f"response FIFO closed while waiting for {context}; worker likely crashed",
                )
            return resp_line

    def start(self, mode: str = "extract") -> None:
        """Start the teacher worker subprocess and set up Mooncake store.

        ``mode`` only selects which sweep the engine comes up ready for; both are
        served by the same process, so callers switch with ``switch_mode()``
        rather than restarting.
        """
        if mode not in ("generate", "extract"):
            raise ValueError(f"unknown teacher mode: {mode!r}")
        if self.is_alive:
            self.switch_mode(mode)
            return
        self._mode = mode

        os.makedirs(self._hidden_dir, exist_ok=True)

        self._fifo_dir = tempfile.mkdtemp(prefix="lumenrl_teacher_fifo_")
        self._cmd_fifo = os.path.join(self._fifo_dir, "cmd")
        self._resp_fifo = os.path.join(self._fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        # Build environment: isolate from torchrun, set teacher GPUs
        env = os.environ.copy()
        gpu_str = ",".join(str(g) for g in self._gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        env["HIP_VISIBLE_DEVICES"] = gpu_str
        env.pop("ROCR_VISIBLE_DEVICES", None)
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

        env["GLOG_minloglevel"] = "3"
        env["GLOG_v"] = "0"
        if self._key_prefix:
            env["LUMENRL_ATOM_KEY_PREFIX"] = self._key_prefix
        env["MOONCAKE_LOG_LEVEL"] = "FATAL"
        env["AITER_LOG_LEVEL"] = "WARNING"
        if "AITER_CONFIG_GEMM_BF16" not in env:
            _bf16_cfg = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "third_party", "aiter", "aiter",
                "configs", "model_configs", "kimik2_bf16_tuned_gemm.csv",
            )
            if not os.path.exists(_bf16_cfg):
                _bf16_cfg = "/root/aiter/aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv"
            env["AITER_CONFIG_GEMM_BF16"] = _bf16_cfg

        attn_backend = os.environ.get("VLLM_ROCM_ATTN_BACKEND")
        if attn_backend:
            env["VLLM_ROCM_ATTN_BACKEND"] = attn_backend

        env["LUMENRL_CAPTURE_MODE"] = self._capture_mode

        # Set Mooncake env vars for worker subprocess
        if self._transport == "mooncake" and self._mooncake_config is not None:
            mc = self._mooncake_config
            master_addr = getattr(mc, "master_server_address", "") or ""
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                master_host = master_addr.rsplit(":", 1)[0]
                s.connect((master_host, 1))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = socket.gethostbyname(socket.gethostname())

            metadata_server = getattr(mc, "metadata_server", "") or ""
            protocol = getattr(mc, "protocol", "tcp") or "tcp"
            device_name = getattr(mc, "device_name", "") or ""
            if protocol == "rdma" and not device_name:
                raise ValueError(
                    "Mooncake RDMA requires a non-empty device_name"
                )

            env["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
            env["MOONCAKE_MASTER_SERVER"] = master_addr
            env["MOONCAKE_METADATA_SERVER"] = metadata_server
            env["MOONCAKE_PROTOCOL"] = protocol
            env["MOONCAKE_DEVICE_NAME"] = device_name
            from lumenrl.transfer.mooncake_config import MooncakeConfig

            global_segment_size = getattr(
                mc, "global_segment_size", 16 * 1024**3
            )
            local_buffer_size = getattr(
                mc, "local_buffer_size", 4 * 1024**3
            )
            env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(
                MooncakeConfig.parse_size(global_segment_size)
                if isinstance(global_segment_size, str)
                else int(global_segment_size)
            )
            env["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(
                MooncakeConfig.parse_size(local_buffer_size)
                if isinstance(local_buffer_size, str)
                else int(local_buffer_size)
            )
            env["MOONCAKE_ENABLE_GPU_DIRECT"] = "0"
            env["MOONCAKE_ENABLE_HARD_PIN"] = (
                "1" if getattr(mc, "enable_hard_pin", False) else "0"
            )

            try:
                from transformers import AutoConfig as _AC
                _hf = _AC.from_pretrained(self._model_name, trust_remote_code=True)
                _hf_text = getattr(_hf, "text_config", _hf)
                _hdim = getattr(_hf_text, "hidden_size", 4096)
            except Exception:
                _hdim = 4096
            from lumenrl.transfer.eagle_mooncake_store import calculate_eagle3_buffer_size
            worker_host_buf = calculate_eagle3_buffer_size(
                max_seq_len=self._max_seq, batch_size=1,
                hidden_dim=_hdim,
                num_aux_layers=self._num_aux_layers,
                safety_margin=2.0,
            )
            env["MOONCAKE_HOST_BUFFER_SIZE"] = str(worker_host_buf)

            logger.info(
                "Mooncake env vars for ATOM worker: local=%s, master=%s, "
                "metadata=%s, protocol=%s",
                local_ip, master_addr, metadata_server, protocol,
            )

        logger.info(
            "AtomTeacherEngine: starting teacher worker for %s "
            "(tp=%d, gpus=%s, transport=%s, quant=%s)",
            self._model_name, self._tp_size, self._gpu_ids,
            self._transport, self._quantization or "none",
        )

        atom_args_json = json.dumps(self._atom_config)

        # One file per start, not one per run. The teacher is restarted every
        # round, and a single reused path meant that a crash was overwritten by
        # the next attempt's log before anyone could read it — including by the
        # automatic container restart, which is how the round-1 startup failure
        # lost its own traceback. The tail dumped on failure is only 50 lines and
        # is usually all NCCL teardown noise.
        # /!\ The counter alone is not enough: the container restarts on failure and
        # the new process starts counting at 1 again. Neither is the pid, which is
        # deterministic inside the container -- see _run_stamp, which is what
        # actually separates one process's logs from its successor's.
        self._start_seq += 1
        worker_log_path = os.path.join(
            self._hidden_dir,
            f"atom_teacher_worker.{self._run_stamp}.{os.getpid()}"
            f".{self._start_seq:03d}.{self._mode}.log",
        )
        self._worker_log_path = worker_log_path
        self._worker_log_f = open(worker_log_path, "w")
        logger.info("AtomTeacherEngine: worker log -> %s", worker_log_path)

        self._proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", _TEACHER_WORKER_SCRIPT,
                self._cmd_fifo, self._resp_fifo,
                self._model_name, str(self._tp_size),
                self._hidden_dir,
                str(self._max_batch), str(self._max_seq),
                atom_args_json,
                json.dumps(self._configured_aux_layer_ids),
                self._mode,
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._worker_log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            resp = self._wait_for_ready(timeout_s=_READY_TIMEOUT_SECONDS)
            if resp.get("status") != "ready":
                raise RuntimeError(f"worker reported non-ready startup status: {resp}")
            self._cmd_f = self._open_cmd_writer(
                timeout_s=_CMD_FIFO_OPEN_TIMEOUT_SECONDS,
            )
            self._resp_f = open(self._resp_fifo, "r")
        except Exception as exc:
            worker_log_tail = self.worker_log_tail()
            logger.error(
                "AtomTeacherEngine: startup failed before READY or channel setup (%s). "
                "Root-cause hint: worker crash, missing READY, or startup timeout.\n"
                "--- worker log (last 50 lines) ---\n%s\n--- end worker log ---",
                exc, worker_log_tail,
            )
            self._terminate_worker("startup failure")
            self._cleanup_ipc_resources()
            self._proc = None
            self._initialized = False
            raise RuntimeError(
                f"Failed to start AtomTeacherEngine worker: {exc}",
            ) from exc

        self._hidden_dim = resp["hidden_dim"]
        self._num_layers = resp.get("num_layers", 0)
        self._aux_layer_indices = resp.get("aux_layer_indices", [])

        # Set up training-side Mooncake store
        if (
            self._consume_hidden_states
            and self._transport == "mooncake"
            and self._mooncake_config is not None
        ):
            try:
                from lumenrl.transfer.eagle_mooncake_store import EagleMooncakeStore
                from lumenrl.transfer.mooncake_config import MooncakeConfig

                mc = self._mooncake_config
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    master_host = (
                        getattr(mc, "master_server_address", "") or ""
                    ).rsplit(":", 1)[0]
                    s.connect((master_host, 1))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = socket.gethostbyname(socket.gethostname())

                from lumenrl.transfer.eagle_mooncake_store import (
                    calculate_eagle3_buffer_size,
                )
                # Reads happen one key at a time, so the buffer needs to hold a
                # single sequence's aux stack — and it must be sized for the real
                # aux-layer count, not the helper's 3-layer default.
                recv_host_buf = calculate_eagle3_buffer_size(
                    max_seq_len=self._max_seq, batch_size=1,
                    hidden_dim=self._hidden_dim,
                    num_aux_layers=self._num_aux_layers,
                    safety_margin=2.0,
                )

                mc_cfg = MooncakeConfig(
                    master_server_address=getattr(mc, "master_server_address", ""),
                    metadata_server=getattr(mc, "metadata_server", ""),
                    local_hostname=local_ip,
                    protocol=getattr(mc, "protocol", "tcp"),
                    device_name=getattr(mc, "device_name", ""),
                    global_segment_size=getattr(mc, "global_segment_size", "16GB"),
                    local_buffer_size=getattr(mc, "local_buffer_size", "4GB"),
                    host_buffer_size=recv_host_buf,
                    max_seq_len=self._max_seq,
                    hidden_dim=self._hidden_dim,
                    get_retry_wait_seconds=getattr(mc, "get_retry_wait_seconds", 1.0),
                    get_retry_max_wait_seconds=getattr(mc, "get_retry_max_wait_seconds", 90.0),
                )
                self._mooncake_store = EagleMooncakeStore(mc_cfg)
                self._mooncake_store.setup(self._local_device)
                logger.info("Training-side EagleMooncakeStore initialized (ATOM)")
            except Exception as e:
                logger.error("Failed to init training-side Mooncake: %s", e)
                self._terminate_worker("mooncake initialization failure")
                self._cleanup_ipc_resources()
                self._proc = None
                self._initialized = False
                raise

        self._initialized = True
        logger.info(
            "AtomTeacherEngine: teacher worker ready (pid=%d, hidden_dim=%d, "
            "aux_layers=%s, transport=%s).",
            self._proc.pid, self._hidden_dim,
            self._aux_layer_indices, self._transport,
        )

    def _send_cmd_unlocked(
        self,
        cmd: dict[str, Any],
        *,
        timeout_s: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict:
        """Send a JSON command and read the response (caller holds _cmd_lock)."""
        try:
            if not self.is_alive:
                raise RuntimeError("Teacher worker is not running")
            self._cmd_f.write(json.dumps(cmd) + "\n")
            self._cmd_f.flush()
            resp_line = self._read_response_line(
                timeout_s=timeout_s,
                context=f"response to cmd={cmd.get('cmd', 'unknown')}",
            )
            return json.loads(resp_line)
        except Exception as exc:
            worker_log_tail = self.worker_log_tail()
            raise RuntimeError(
                "ATOM teacher command "
                f"{cmd.get('cmd', 'unknown')!r} failed: {exc}\n"
                "--- worker log (last 50 lines) ---\n"
                f"{worker_log_tail}"
                "--- end worker log ---"
            ) from exc

    def _send_cmd(
        self,
        cmd: dict[str, Any],
        *,
        timeout_s: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict:
        """Send a JSON command and read the response (thread-safe)."""
        with self._cmd_lock:
            return self._send_cmd_unlocked(cmd, timeout_s=timeout_s)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recv_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run teacher forward pass, return hidden states on training GPU.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` mask (1 = valid, 0 = pad).
            recv_device: Device for received tensors. Defaults to local GPU.

        Returns:
            Dict with ``hidden_states`` ``[B, T, 3*D]`` (3 aux layers),
            ``token_embeds`` ``[B, T, D]``,
            ``last_hidden_states`` ``[B, T, D]``,
            and ``input_ids``.
            All tensors on ``recv_device`` or ``self._local_device``.
        """
        manifest = self.extract_hidden_state_manifest(input_ids, attention_mask)
        T = manifest["sequence_width"]
        D = manifest["hidden_dim"]

        if recv_device is None:
            recv_device = self._local_device

        mooncake_keys = manifest["mooncake_keys"]
        seq_lens = manifest["sequence_lengths"]
        num_aux = len(self._aux_layer_indices)
        training_hidden_size = num_aux * D

        # The teacher prefills each row trimmed to its real length, so entries are
        # ragged; pad back to T so positions still line up with the caller's
        # right-padded batch.
        all_hs = []
        all_ids = []
        all_last_hs = []

        for key, sequence_length in zip(mooncake_keys, seq_lens, strict=True):
            T_i = int(sequence_length)
            shapes = {
                "hidden_states": (T_i, training_hidden_size),
                "input_ids": (T_i,),
                "last_hidden_states": (T_i, D),
            }
            dtypes = {
                "hidden_states": torch.bfloat16,
                "input_ids": torch.int64,
                "last_hidden_states": torch.bfloat16,
            }

            output = self._mooncake_store.get(
                key, shapes, dtypes, device=recv_device,
            )

            all_hs.append(output.hidden_states)
            all_last_hs.append(output.last_hidden_states)
            all_ids.append(output.input_ids)

            self._mooncake_store.remove_eagle3_tensors(
                key, has_last_hidden_states=True, has_target=False,
            )

        hidden_states = _pad_stack(all_hs, T)               # [B, T, num_aux*D]
        last_hidden_states = _pad_stack(all_last_hs, T)     # [B, T, D]
        ret_ids = _pad_stack(all_ids, T)                    # [B, T]

        # First aux layer as embed proxy (same convention as VllmTeacherEngine)
        token_embeds = hidden_states[:, :, :D].clone()

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": ret_ids,
            "last_hidden_states": last_hidden_states,
        }

    def extract_hidden_state_manifest(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Run extraction but leave tensors in Mooncake for a remote consumer.

        The returned object contains only keys and tensor-shape metadata.  This
        is the disaggregated teacher API: Ray carries the small manifest while
        hidden states stay in Mooncake and cross nodes through its data plane.
        """
        if not self.is_alive:
            self.start(mode="extract")
        else:
            self.switch_mode("extract")

        if self._transport != "mooncake":
            raise AssertionError(
                "AtomTeacherEngine only implements the Mooncake transport",
            )

        lengths = attention_mask.sum(dim=1).to(torch.int64).cpu()
        with self._cmd_lock:
            self._req_counter += 1
            tag = f"req_{self._req_counter}"
            input_path = os.path.join(self._hidden_dir, f"{tag}_input.pt")
            torch.save(
                {"input_ids": input_ids.cpu(), "lengths": lengths},
                input_path,
            )
            try:
                resp = self._send_cmd_unlocked({
                    "cmd": "extract_hidden",
                    "input_path": input_path,
                })
            finally:
                try:
                    os.remove(input_path)
                except FileNotFoundError:
                    pass

        if resp.get("status") != "ok":
            raise RuntimeError(f"extract_hidden failed: {resp}")

        keys = list(resp["mooncake_keys"])
        by_key = resp.get("seq_lens", {})
        return {
            "mooncake_keys": keys,
            "sequence_lengths": [int(by_key.get(key, resp["T"])) for key in keys],
            "sequence_width": int(resp["T"]),
            "hidden_dim": int(resp["D"]),
            "num_aux_layers": len(self._aux_layer_indices),
        }

    def generate_tokens(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the teacher's own continuation of each prompt.

        This is the first of the two on-policy sweeps and produces tokens only.
        Hidden states come from a second ``extract_hidden_states()`` pass over the
        finished sequences: ATOM captures during prefill, so a decode sweep sees
        activations one position at a time and cannot fill the batch in one go.

        Args:
            prompt_ids: ``[B, T_prompt]`` right-padded prompt tokens.
            prompt_mask: ``[B, T_prompt]`` mask (1 = real token, 0 = pad).
            max_tokens: Cap on generated tokens per request.
            temperature: Sampling temperature (0.0 = greedy).

        Returns:
            ``(full_ids, seq_lens, prompt_lens)`` where ``full_ids`` is
            ``[B, T_full]`` prompt+response right-padded with zeros, and the two
            length vectors are ``[B]`` on CPU. Pad id 0 is a real token in K3's
            vocabulary, so lengths cannot be recovered from ``full_ids`` and are
            returned explicitly.
        """
        if not self.is_alive:
            self.start(mode="generate")
        else:
            self.switch_mode("generate")

        prompt_lens = prompt_mask.sum(dim=1).to(torch.int64).cpu()

        with self._cmd_lock:
            self._req_counter += 1
            tag = f"req_{self._req_counter}"
            input_path = os.path.join(self._hidden_dir, f"{tag}_prompt.pt")
            output_path = os.path.join(self._hidden_dir, f"{tag}_gen.pt")

            torch.save(
                {"input_ids": prompt_ids.cpu(), "lengths": prompt_lens},
                input_path,
            )

            resp = self._send_cmd_unlocked(
                {
                    "cmd": "generate_tokens",
                    "input_path": input_path,
                    "output_path": output_path,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout_s=_generate_timeout_seconds(
                    prompt_ids.shape[0], max_tokens
                ),
            )

        if resp.get("status") != "ok":
            raise RuntimeError(f"generate_tokens failed: {resp}")

        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        try:
            os.unlink(output_path)
            os.unlink(input_path)
        except OSError:
            pass

        completions = payload["completions"]
        worker_prompt_lens = payload["prompt_lens"]

        B = len(completions)
        if B != prompt_ids.shape[0]:
            raise RuntimeError(
                f"generate_tokens returned {B} rows for a batch of "
                f"{prompt_ids.shape[0]}",
            )

        seq_lens = torch.tensor(
            [int(worker_prompt_lens[i]) + len(completions[i]) for i in range(B)],
            dtype=torch.int64,
        )
        T_full = int(seq_lens.max().item())

        full_ids = torch.zeros(B, T_full, dtype=torch.long)
        for i in range(B):
            p_len = int(worker_prompt_lens[i])
            full_ids[i, :p_len] = prompt_ids[i, :p_len].cpu()
            if completions[i]:
                comp = torch.tensor(completions[i], dtype=torch.long)
                full_ids[i, p_len:p_len + comp.numel()] = comp

        empty = int((seq_lens == prompt_lens).sum().item())
        if empty:
            logger.warning(
                "AtomTeacherEngine: %d/%d prompts produced no tokens; those rows "
                "carry no on-policy signal",
                empty, B,
            )

        return full_ids, seq_lens, prompt_lens

    def get_lm_head_weight(self) -> torch.Tensor:
        """Load the teacher's lm_head.weight from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "lm_head_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=True)
            time.sleep(0.5)
        raise FileNotFoundError(
            f"lm_head_weight.pt not found at {path} after 15s."
        )

    def get_embed_weight(self) -> torch.Tensor:
        """Load the teacher's embed_tokens.weight from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "embed_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=True)
            time.sleep(0.5)
        raise FileNotFoundError(
            f"embed_weight.pt not found at {path} after 15s."
        )

    def get_norm_weight(self) -> tuple[torch.Tensor, float]:
        """Load the teacher's final norm weight and eps from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "norm_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                return data["weight"], data["eps"]
            time.sleep(0.5)
        raise FileNotFoundError(
            f"norm_weight.pt not found at {path} after 15s."
        )

    def shutdown(self) -> None:
        """Terminate the teacher worker subprocess and clean up."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_cmd(
                    {"cmd": "shutdown"},
                    timeout_s=_SHUTDOWN_COMMAND_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning("AtomTeacherEngine: graceful shutdown command failed: %s", exc)
            self._terminate_worker("shutdown requested")

        if self._mooncake_store is not None:
            try:
                if hasattr(self._mooncake_store, "close"):
                    self._mooncake_store.close()
                elif hasattr(self._mooncake_store, "shutdown"):
                    self._mooncake_store.shutdown()
            except Exception as exc:
                logger.warning("AtomTeacherEngine: failed to close Mooncake store: %s", exc)
            finally:
                self._mooncake_store = None

        self._cleanup_ipc_resources()
        self._proc = None
        self._initialized = False
        logger.info("AtomTeacherEngine: shutdown complete.")

    def __del__(self) -> None:
        self.shutdown()


__all__ = ["AtomTeacherEngine"]
