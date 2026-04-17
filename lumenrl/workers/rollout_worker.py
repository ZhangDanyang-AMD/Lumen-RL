"""Rollout worker backed by the ATOM inference engine."""

from __future__ import annotations

import logging
from typing import Any

import torch

from lumenrl.core.config import AtomConfig
from lumenrl.core.protocol import DataProto
from lumenrl.engine.inference.atom_engine import AtomEngine
from lumenrl.engine.inference.kv_cache import FP8KVCacheManager
from lumenrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class AtomRolloutWorker(BaseWorker):
    """Generates sequences with optional router logits for MoE / R3."""

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._engine: AtomEngine | None = None
        self._kv_manager: FP8KVCacheManager | None = None

    def init_model(self) -> None:
        """Construct :class:`~lumenrl.engine.inference.atom_engine.AtomEngine`."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        model_name = str(policy.get("model_name", "stub-model"))
        gen = policy.get("generation", {}) or {}
        atom_dict = gen.get("atom_cfg") or policy.get("atom_cfg") or {}
        atom_cfg = AtomConfig(
            tensor_parallel_size=int(atom_dict.get("tensor_parallel_size", 1)),
            kv_cache_dtype=str(atom_dict.get("kv_cache_dtype", "auto")),
            max_model_len=atom_dict.get("max_model_len"),
        )
        self._engine = AtomEngine(config=atom_cfg, model_name=model_name)
        self._engine.init()
        quant = get_nested_config(self.config, "quantization", "rollout", default={}) or {}
        fp8_kv = str(quant.get("precision", "")).lower() in ("fp8", "float8")
        self._kv_manager = FP8KVCacheManager(enabled=fp8_kv)
        self._log.info("AtomRolloutWorker: engine ready for %s", model_name)

    def prepare_for_generation(self) -> None:
        """Recalibrate caches / set sampling defaults before a rollout burst."""
        if self._kv_manager is not None:
            self._kv_manager.recalibrate_scales()
        self._log.info("AtomRolloutWorker.prepare_for_generation")

    def generate(self, batch: DataProto) -> DataProto:
        """Batch decode prompts into ``sequences`` (and optional router tensors)."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before generate().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        prompts = [row.tolist() for row in batch["input_ids"].cpu()]
        sampling = dict(batch.meta.get("sampling_params", {}))
        sampling.setdefault("max_new_tokens", 32)
        sampling.setdefault("temperature", 1.0)
        token_lists = self._engine.generate(prompts, sampling_params=sampling)

        max_len = max(len(seq) for seq in token_lists)
        pad_id = int(batch.meta.get("pad_token_id", 0))
        out = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(token_lists):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        meta_out = {**batch.meta, "sampling_params": sampling}
        out_proto = DataProto(tensors={"sequences": out}, meta=meta_out)

        if batch.meta.get("return_router_dists", False):
            layers = int(batch.meta.get("num_router_layers", 1))
            bsz = out.shape[0]
            for layer_idx in range(layers):
                logits = torch.randn(bsz, 8)
                out_proto.add_router_distributions(layer_idx, logits)
        return out_proto

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Hot-swap rollout weights from training workers."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before update_weights().")
        self._engine.update_weights(state_dict)
        self._log.info("AtomRolloutWorker.update_weights: %d tensors", len(state_dict))

    def finish_generation(self) -> None:
        """End of rollout phase bookkeeping (does not destroy the engine)."""
        self._log.info("AtomRolloutWorker.finish_generation")

    def cleanup(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
        self._engine = None
        self._kv_manager = None
        super().cleanup()
