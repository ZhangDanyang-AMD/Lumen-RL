"""SGLang + ATOM teacher engine for speculative distillation.

Launches a teacher model in a subprocess using SGLang Engine with ATOM
plugin (``SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models``).
TorchSpec's SGLang patches enable spec_training mode: the engine captures
hidden states during prefill and stores them to **Mooncake RDMA**.
The training process reads them via ``MooncakeStore.get_tensors()``.

Architecture::

    Training Process                 SGLang Subprocess (ATOM plugin)
    ─────────────────               ──────────────────────────────
    send input_ids via /dev/shm  →  engine.generate(input_ids,
                                        spec_training_data_id=...,
                                        max_new_tokens=0)
                                    → SGLang captures hidden_states
                                    → stores to Mooncake RDMA
    get hidden_states from       ←  returns mooncake keys + shapes
    MooncakeStore

Requires:
    - SGLang with TorchSpec spec_training patches applied
    - ATOM installed (provides optimized Aiter kernels via plugin)
    - Mooncake Transfer Engine (RDMA or TCP)
"""

from __future__ import annotations

import json
import logging
import os
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


_WORKER_SCRIPT = textwrap.dedent("""\
import json
import os
import sys
import time
import logging

# ---- Isolate from torchrun environment ----
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
logger = logging.getLogger("sglang_atom_teacher")

import torch
import sglang as sgl
from transformers import AutoConfig

cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
tp_size = int(sys.argv[4])
gpu_ids_str = sys.argv[5]
atom_plugin = sys.argv[6] == "1"
quantization = sys.argv[7] if len(sys.argv) > 7 else ""

gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
os.environ["HIP_VISIBLE_DEVICES"] = gpu_ids_str
if "ROCR_VISIBLE_DEVICES" in os.environ:
    del os.environ["ROCR_VISIBLE_DEVICES"]
base_gpu_id = 0  # first visible GPU after CUDA_VISIBLE_DEVICES

# Set ATOM plugin if requested
if atom_plugin:
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "atom.plugin.sglang.models"

# Get model hidden size from config
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
hidden_size = getattr(hf_config, "hidden_size", None)
if hidden_size is None:
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
num_layers = getattr(hf_config, "num_hidden_layers", 32)

# Aux layer IDs: early, mid, near-end (Eagle3 convention from TorchSpec)
aux_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
logger.info("Model: %s, hidden=%d, layers=%d, aux_layers=%s",
            model_path, hidden_size, num_layers, aux_layer_ids)

# Create SGLang Engine with spec_training support
logger.info("Creating SGLang Engine (tp=%d, base_gpu=%d, atom_plugin=%s, quant=%s)",
            tp_size, base_gpu_id, atom_plugin, quantization or "none")

engine_kwargs = dict(
    model_path=model_path,
    disable_radix_cache=True,
    enable_return_hidden_states=True,
    enable_aux_hidden_states=True,
    aux_hidden_state_layer_ids=aux_layer_ids,
    enable_spec_training_mooncake=True,
    tp_size=tp_size,
    base_gpu_id=base_gpu_id,
    gpu_id_step=1,
    mem_fraction_static=0.85 if quantization else 0.7,
    trust_remote_code=True,
    chunked_prefill_size=-1,
    disable_cuda_graph=not bool(quantization),
    disable_custom_all_reduce=True,
    log_level="info",
    watchdog_timeout=1200,
)
if quantization in ("mxfp4", "fp4"):
    # Check if model already has a quantization_config (e.g. compressed-tensors).
    # SGLang rejects mismatched quant methods, so skip explicit quantization arg
    # and let ATOM plugin handle online MXFP4 via its own kernels.
    model_quant = getattr(hf_config, "quantization_config", None)
    if model_quant is not None:
        if hasattr(model_quant, "to_dict"):
            mq_method = model_quant.to_dict().get("quant_method", "")
        elif isinstance(model_quant, dict):
            mq_method = model_quant.get("quant_method", "")
        else:
            mq_method = ""
    else:
        mq_method = ""
    if mq_method and mq_method not in ("mxfp4", "fp4"):
        logger.info("Model has existing quantization_config (%s), overriding with json_model_override_args to clear it and force MXFP4", mq_method)
        engine_kwargs["quantization"] = "mxfp4"
        engine_kwargs["json_model_override_args"] = '{"quantization_config": {}}'
        engine_kwargs["kv_cache_dtype"] = "fp8_e4m3"
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["skip_server_warmup"] = True
        engine_kwargs["pre_warm_nccl"] = False
    else:
        engine_kwargs["quantization"] = "mxfp4"
        engine_kwargs["kv_cache_dtype"] = "fp8_e4m3"
        logger.info("MXFP4 quantization enabled (ATOM/AITER ROCm): mxfp4 + fp8_e4m3 KV cache")

engine = sgl.Engine(**engine_kwargs)

logger.info("SGLang Engine created successfully")

# Extract lm_head weight and save
lm_head_path = f"/dev/shm/lumenrl_sglang_lm_head_{os.getpid()}.pt"
try:
    lm_head_w = engine.get_weights_by_name("lm_head.weight")
    if isinstance(lm_head_w, list):
        lm_head_w = lm_head_w[0]
    torch.save(lm_head_w.cpu(), lm_head_path)
    logger.info("lm_head.weight saved: %s", tuple(lm_head_w.shape))
except Exception as e:
    logger.warning("Could not extract lm_head via get_weights_by_name: %s", e)
    lm_head_path = ""

num_aux = len(aux_layer_ids)
req_counter = 0

# Signal ready
resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({
    "status": "ready",
    "hidden_size": hidden_size,
    "num_aux_layers": num_aux,
    "aux_layer_ids": aux_layer_ids,
    "lm_head_path": lm_head_path,
}) + "\\n")
resp_f.flush()

# Command loop
cmd_f = open(cmd_fifo, "r")
for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "extract_hidden":
        input_path = msg["input_path"]
        data = torch.load(input_path, map_location="cpu", weights_only=True)
        input_ids = data["input_ids"]

        B, T = input_ids.shape
        req_counter += 1

        # Convert to list-of-lists for SGLang
        input_ids_list = [input_ids[i].tolist() for i in range(B)]
        data_ids = [f"lumenrl_{os.getpid()}_{req_counter}_{i}" for i in range(B)]

        results = engine.generate(
            input_ids=input_ids_list,
            spec_training_data_id=data_ids,
            sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
        )

        # Extract mooncake keys from results
        mooncake_keys = []
        seq_lens = []
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            store_keys = meta.get("spec_training_mooncake_store_keys", [])
            prompt_tokens = meta.get("prompt_tokens", T)
            for key in store_keys:
                mooncake_keys.append(key)
                seq_lens.append(prompt_tokens)

        resp_f.write(json.dumps({
            "status": "ok",
            "B": B,
            "T": T,
            "hidden_size": hidden_size,
            "num_aux_layers": num_aux,
            "mooncake_keys": mooncake_keys,
            "seq_lens": seq_lens,
        }) + "\\n")
        resp_f.flush()

        try:
            os.unlink(input_path)
        except OSError:
            pass

    elif cmd == "shutdown":
        engine.shutdown()
        if lm_head_path and os.path.exists(lm_head_path):
            os.unlink(lm_head_path)
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class SglangTeacherEngine:
    """Teacher engine: SGLang + ATOM plugin + Mooncake RDMA.

    Launches SGLang Engine in a subprocess on dedicated inference GPUs.
    ATOM provides optimized Aiter kernels via its SGLang plugin.
    TorchSpec patches enable spec_training_mooncake: hidden states are
    stored directly to Mooncake RDMA by SGLang, and the training process
    fetches them via ``MooncakeStore.get_tensors()``.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: list[int],
        mooncake_config: Any = None,
        *,
        tensor_parallel_size: int = 1,
        atom_plugin: bool = True,
        quantization: str = "",
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        local_device: Optional[torch.device] = None,
    ):
        self._model_name = model_name
        self._gpu_ids = gpu_ids
        self._mooncake_config = mooncake_config
        self._tp_size = tensor_parallel_size or len(gpu_ids)
        self._atom_plugin = atom_plugin
        self._quantization = quantization
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._local_device = local_device or torch.device("cuda:0")

        self._proc: Optional[subprocess.Popen] = None
        self._cmd_fifo: Optional[str] = None
        self._resp_fifo: Optional[str] = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False

        self._hidden_size: int = 0
        self._num_aux_layers: int = 0
        self._aux_layer_ids: list[int] = []
        self._lm_head_weight: Optional[torch.Tensor] = None
        self._mooncake_store: Any = None

    @property
    def is_alive(self) -> bool:
        return (
            self._initialized
            and self._proc is not None
            and self._proc.poll() is None
        )

    def start(self) -> None:
        """Start SGLang+ATOM subprocess and set up Mooncake store."""
        if self.is_alive:
            return

        fifo_dir = tempfile.mkdtemp(prefix="lumenrl_sglang_teacher_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

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

        if self._atom_plugin:
            env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "atom.plugin.sglang.models"

        # Set Mooncake env vars for SGLang's MooncakeConfig.from_env()
        if self._mooncake_config is not None:
            mc = self._mooncake_config
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = socket.gethostbyname(socket.gethostname())

            master_addr = getattr(mc, "master_server_address", "") or ""
            metadata_server = getattr(mc, "metadata_server", "") or ""
            protocol = getattr(mc, "protocol", "tcp") or "tcp"
            device_name = getattr(mc, "device_name", "") or ""

            env["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
            env["MOONCAKE_MASTER_SERVER"] = master_addr
            env["MOONCAKE_METADATA_SERVER"] = metadata_server
            env["MOONCAKE_PROTOCOL"] = protocol
            env["MOONCAKE_DEVICE_NAME"] = device_name
            env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(
                mc.global_segment_size_bytes
                if hasattr(mc, "global_segment_size_bytes")
                else 17179869184
            )
            env["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(
                mc.local_buffer_size_bytes
                if hasattr(mc, "local_buffer_size_bytes")
                else 4294967296
            )
            env["MOONCAKE_ENABLE_GPU_DIRECT"] = "0"
            env["MOONCAKE_ENABLE_HARD_PIN"] = "0"

            logger.info(
                "Mooncake env vars for subprocess: local=%s, master=%s, "
                "metadata=%s, protocol=%s, device=%s",
                local_ip, master_addr, metadata_server, protocol, device_name,
            )

        logger.info(
            "SglangTeacherEngine: starting subprocess for %s "
            "(tp=%d, gpus=%s, atom_plugin=%s)",
            self._model_name, self._tp_size, self._gpu_ids, self._atom_plugin,
        )

        self._proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", _WORKER_SCRIPT,
                self._cmd_fifo, self._resp_fifo,
                self._model_name, str(self._tp_size),
                ",".join(str(g) for g in self._gpu_ids),
                "1" if self._atom_plugin else "0",
                self._quantization or "",
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
                "SGLang teacher subprocess exited before ready. "
                "Check stderr for SGLang/ATOM loading errors."
            )
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"SGLang teacher failed to start: {resp}")

        self._cmd_f = open(self._cmd_fifo, "w")

        self._hidden_size = resp["hidden_size"]
        self._num_aux_layers = resp["num_aux_layers"]
        self._aux_layer_ids = resp.get("aux_layer_ids", [])

        lm_head_path = resp.get("lm_head_path", "")
        if lm_head_path and os.path.exists(lm_head_path):
            self._lm_head_weight = torch.load(
                lm_head_path, map_location="cpu", weights_only=True
            )
            logger.info(
                "lm_head.weight loaded: %s", tuple(self._lm_head_weight.shape)
            )

        # Set up training-side Mooncake store (EagleMooncakeStore for key compatibility)
        if self._mooncake_config is not None:
            try:
                from lumenrl.transfer.eagle_mooncake_store import EagleMooncakeStore
                from lumenrl.transfer.mooncake_config import MooncakeConfig

                mc = self._mooncake_config

                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = socket.gethostbyname(socket.gethostname())

                mc_cfg = MooncakeConfig(
                    master_server_address=getattr(mc, "master_server_address", ""),
                    metadata_server=getattr(mc, "metadata_server", ""),
                    local_hostname=local_ip,
                    protocol=getattr(mc, "protocol", "tcp"),
                    device_name=getattr(mc, "device_name", ""),
                    global_segment_size=getattr(mc, "global_segment_size", "16GB"),
                    local_buffer_size=getattr(mc, "local_buffer_size", "4GB"),
                )
                self._mooncake_store = EagleMooncakeStore(mc_cfg)
                self._mooncake_store.setup(self._local_device)
                logger.info("Training-side EagleMooncakeStore initialized")
            except Exception as e:
                logger.warning("Failed to init training-side Mooncake: %s", e)

        self._initialized = True
        logger.info(
            "SglangTeacherEngine: ready (pid=%d, hidden_size=%d, "
            "aux_layers=%s, mooncake=%s)",
            self._proc.pid, self._hidden_size,
            self._aux_layer_ids, self._mooncake_store is not None,
        )

    def _send_cmd(self, cmd: dict) -> dict:
        if not self.is_alive:
            raise RuntimeError("SGLang teacher worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("SGLang teacher worker closed response FIFO")
        return json.loads(resp_line)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run teacher forward via SGLang and fetch hidden states from Mooncake.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` mask (1 = valid, 0 = pad).

        Returns:
            Dict with ``hidden_states`` ``[B, T, D]``,
            ``token_embeds`` ``[B, T, D]`` (first aux layer),
            and ``input_ids``.
        """
        if not self.is_alive:
            self.start()

        input_path = f"/dev/shm/lumenrl_sglang_input_{os.getpid()}.pt"
        torch.save(
            {"input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu()},
            input_path,
        )

        resp = self._send_cmd({
            "cmd": "extract_hidden",
            "input_path": input_path,
        })

        if resp.get("status") != "ok":
            raise RuntimeError(f"extract_hidden failed: {resp}")

        B = resp["B"]
        T = resp["T"]
        hidden_size = resp["hidden_size"]
        num_aux = resp["num_aux_layers"]
        mooncake_keys = resp["mooncake_keys"]
        seq_lens = resp["seq_lens"]

        concat_hidden_size = num_aux * hidden_size

        all_hidden = []
        all_embeds = []
        all_ids = []

        for i, key in enumerate(mooncake_keys):
            seq_len = seq_lens[i]
            shapes = {
                "hidden_states": (seq_len, concat_hidden_size),
                "input_ids": (seq_len,),
                "last_hidden_states": (seq_len, hidden_size),
            }
            dtypes = {
                "hidden_states": torch.bfloat16,
                "input_ids": torch.int64,
                "last_hidden_states": torch.bfloat16,
            }

            if self._mooncake_store is not None:
                output = self._mooncake_store.get(
                    key, shapes, dtypes, device=self._local_device,
                )
                hs_concat = output.hidden_states
                last_hs = output.last_hidden_states if output.last_hidden_states is not None else hs_concat[:, -hidden_size:]
                first_aux = hs_concat[:, :hidden_size]

                all_hidden.append(last_hs.unsqueeze(0))
                all_embeds.append(first_aux.unsqueeze(0))
                all_ids.append(output.input_ids.unsqueeze(0))

                self._mooncake_store.remove_eagle3_tensors(
                    key, has_last_hidden_states=True, has_target=False,
                )
            else:
                raise RuntimeError(
                    "Mooncake store not available. SGLang spec_training "
                    "requires Mooncake for hidden state transfer."
                )

        hidden_states = torch.cat(all_hidden, dim=0).to(self._local_device)
        token_embeds = torch.cat(all_embeds, dim=0).to(self._local_device)
        ret_ids = torch.cat(all_ids, dim=0).to(self._local_device)

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": ret_ids,
        }

    def get_lm_head_weight(self) -> torch.Tensor:
        if self._lm_head_weight is None:
            logger.info("lm_head not from engine; loading from HF: %s", self._model_name)
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import AutoConfig
            import json as _json

            config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
            cache_dir = None
            try:
                index_path = hf_hub_download(
                    self._model_name, "model.safetensors.index.json", cache_dir=cache_dir
                )
                with open(index_path) as f:
                    index = _json.load(f)
                shard_file = index["weight_map"]["lm_head.weight"]
                shard_path = hf_hub_download(self._model_name, shard_file, cache_dir=cache_dir)
                tensors = load_file(shard_path)
                self._lm_head_weight = tensors["lm_head.weight"].to(torch.bfloat16).cpu()
            except Exception:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
                )
                self._lm_head_weight = model.lm_head.weight.detach().cpu()
                del model
            logger.info("lm_head.weight loaded: %s", tuple(self._lm_head_weight.shape))
        return self._lm_head_weight

    def shutdown(self) -> None:
        """Terminate the SGLang subprocess and clean up."""
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

        if self._mooncake_store is not None:
            self._mooncake_store.close()
            self._mooncake_store = None

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
        logger.info("SglangTeacherEngine: shutdown complete.")

    def __del__(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()


__all__ = ["SglangTeacherEngine"]