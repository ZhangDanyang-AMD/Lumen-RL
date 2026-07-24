"""vLLM worker extension for colocated ZMQ CUDA-IPC weight sync.

Vendored + trimmed from verl's ``vLLMColocateWorkerExtension``
(``verl/workers/rollout/vllm_rollout/utils.py``). vLLM instantiates the worker
with this class mixed in via ``worker_extension_cls`` so that, no matter the
underlying worker class (V0/V1), the rollout worker exposes
``update_weights_from_ipc``. LoRA / FP8 / QAT branches are dropped; LumenRL's
verl-aligned BF16 path only needs standard ``model.load_weights``.

The ZMQ socket path is keyed by ``{ray_job_id, replica_rank, local_rank}`` so it
matches the training-side ``BucketedWeightSender`` regardless of per-worker
``CUDA_VISIBLE_DEVICES`` differences and is unique across replicas / jobs.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import signal
from types import MethodType

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "WARN"))


def _monkey_patch_compute_logits(model, vocab_size: int) -> None:
    """Mask out-of-vocab (padded) logits to -inf (verl monkey_patch_compute_logits).

    vLLM pads the vocab to a multiple of TP/64; without masking, the padded
    columns can be sampled — and after an online-FP8 weight reload the requantized
    lm_head can make those padded logits large, so the rollout emits garbage /
    never-EOS sequences that collapse training. verl applies this at engine init.
    """
    original_compute_logits = model.compute_logits

    def compute_logits(self, *args, **kwargs):  # noqa: ANN001
        logits = original_compute_logits(*args, **kwargs)
        if logits is not None:
            logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def set_death_signal() -> None:
    """Kill this process when its parent (the server actor) dies (Linux only)."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception:
        pass


class vLLMColocateWorkerExtension:
    """Mixed into the vLLM worker; receives IPC weight buckets and loads them."""

    def __new__(cls, **kwargs):
        set_death_signal()
        # Online per-block FP8 (verl-aligned `fp8_per_block`): the BF16 actor
        # weights are re-quantized to per-128-block FP8 inside vLLM. On ROCm the
        # per-block weight post-processing needs a guard patch, applied here in
        # the worker subprocess before the model is built.
        try:
            vllm_config = kwargs.get("vllm_config")
            want_patch = os.environ.get("LUMENRL_FP8_PER_BLOCK", "0") == "1"
            if vllm_config is not None:
                from lumenrl.engine.inference.vllm_fp8_utils import is_online_quant_model
                want_patch = want_patch or is_online_quant_model(vllm_config)
            if want_patch:
                from lumenrl.engine.inference.vllm_fp8_utils import (
                    apply_vllm_fp8_per_block_patches,
                )
                apply_vllm_fp8_per_block_patches()
        except Exception as exc:  # pragma: no cover - best effort, never block worker
            logger.warning("fp8_per_block patch at worker __new__ failed: %s", exc)
        return super().__new__(cls)

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"

    def monkey_patch_model(self, vocab_size: int) -> None:
        """verl-aligned engine-init patch: mask OOV/padded logits to -inf."""
        _monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        logger.info("monkey_patch_model: compute_logits OOV mask (vocab=%d)", vocab_size)

    def update_weights_from_ipc(self, use_shm: bool = False) -> None:
        """Receive bucketed weights over ZMQ IPC and load them into the model."""
        from vllm.platforms import current_platform

        from lumenrl.engine.inference.bucketed_weight_transfer import BucketedWeightReceiver

        if getattr(self, "device", None) is None:
            dev_type = current_platform.device_type
            self.device = torch.device(f"{dev_type}:{self.local_rank}")

        model = self.model_runner.model
        model_config = self.model_runner.vllm_config.model_config

        # Online per-block FP8 needs vLLM's layerwise-reload lifecycle around the
        # weight load so the per-block FP8 scales are rebuilt from the fresh BF16
        # weights (verl `is_online_quant_model` branch). Standard BF16 rollout
        # just loads + runs the non-idempotent post-load transforms once.
        is_online_quant = False
        try:
            from lumenrl.engine.inference.vllm_fp8_utils import is_online_quant_model

            is_online_quant = is_online_quant_model(self.model_runner.vllm_config)
        except Exception as exc:  # pragma: no cover
            logger.warning("online-quant detection failed: %s", exc)

        if is_online_quant:
            # verl-exact online per-block reload: initialize layerwise reload BEFORE
            # the first bucket, load each bucket as it arrives (the wrapped loaders
            # track per-layer completion across buckets), finalize AFTER the last.
            from lumenrl.engine.inference.vllm_fp8_utils import (
                finalize_online_quantized_weights_loading,
                prepare_online_quantized_weights_for_loading,
            )

            prepare_online_quantized_weights_for_loading(model)
            receiver = BucketedWeightReceiver(
                zmq_handle=self._get_zmq_handle(),
                device=self.device,
                use_shm=use_shm,
            )
            _stats = {"buckets": 0, "weights": 0}

            def _load_online(weights):
                # CRITICAL: the receiver hands out tensors that are VIEWS into the
                # shared IPC buffer, which the sender overwrites on the next bucket
                # (and _cleanup frees entirely before finalize runs). The online
                # layerwise reload DEFERS per-layer FP8 requant until a layer is
                # complete / until finalize, holding these views -> by then they
                # point at freed/overwritten memory -> corrupted weights (uniform
                # lm_head -> policy collapse after the first update). Clone so the
                # deferred reload owns valid storage. (The standard BF16 path copies
                # into params during the call, so it does not need this.)
                cloned = [(n, t.clone()) for (n, t) in weights]
                _stats["buckets"] += 1
                _stats["weights"] += len(cloned)
                model.load_weights(cloned)

            receiver.receive_weights(on_bucket_received=_load_online)
            finalize_online_quantized_weights_loading(model, model_config)
            logger.info(
                "online fp8 reload: buckets=%d weights=%d",
                _stats["buckets"], _stats["weights"],
            )
            return

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: model.load_weights(weights)
        )

        # Some post-load transforms are non-idempotent; run once after all buckets.
        try:
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            process_weights_after_loading(model, model_config, self.device)
        except Exception as exc:  # pragma: no cover - best effort parity with verl
            logger.warning("process_weights_after_loading skipped: %s", exc)

    def reload_weights_from_safetensors(self, weight_dir: str) -> None:
        """Load weights from safetensors on shared storage (separation mode)."""
        import json
        import os

        from safetensors.torch import load_file
        from vllm.platforms import current_platform

        if getattr(self, "device", None) is None:
            dev_type = current_platform.device_type
            self.device = torch.device(f"{dev_type}:{self.local_rank}")

        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            files = sorted(
                f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
            )

        model = self.model_runner.model
        for fname in files:
            sd = load_file(os.path.join(weight_dir, fname))
            model.load_weights(list(sd.items()))

        # The model was already transformed/compiled during engine startup.
        # process_weights_after_loading() is not idempotent for all vLLM model
        # implementations; running it after every reload can re-wrap modules
        # and trigger a second multi-worker torch.compile.  Reload only replaces
        # parameter data, then wait for all H2D copies before acknowledging RPC.
        torch.cuda.synchronize(self.device)
        logger.info("reload_weights_from_safetensors: loaded from %s", weight_dir)
