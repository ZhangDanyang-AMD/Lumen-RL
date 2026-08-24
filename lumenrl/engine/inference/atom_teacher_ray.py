"""Ray facade for disaggregated ATOM speculative-distillation teachers."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.engine.inference.atom_teacher_engine import AtomTeacherEngine


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        result = dict(value)
    else:
        result = {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }
    extra = result.pop("extra_args", None)
    if isinstance(extra, dict):
        result.update(extra)
    return result


class AtomTeacherRayActor:
    """Own one TP ATOM engine and publish hidden states to Mooncake.

    The actor deliberately returns only tokens, masks, and Mooncake keys.  It
    never returns hidden-state tensors through Ray's object store.
    """

    def __init__(
        self,
        config_path: str,
        overrides: list[str],
        replica_index: int,
    ) -> None:
        self.config = LumenRLConfig.from_yaml(config_path, overrides=overrides)
        self.replica_index = replica_index
        teacher_cfg = self.config.algorithm.teacher
        spec_cfg = self.config.algorithm.spec_distill

        if teacher_cfg.inference_backend != "atom":
            raise ValueError("AtomTeacherRayActor requires inference_backend=atom")
        self._generate_mode = getattr(teacher_cfg, "generate_mode", "prefill")
        if self._generate_mode not in ("generate", "prefill"):
            raise ValueError(
                "disaggregated K3 actor requires generate_mode=generate or prefill"
            )
        if getattr(teacher_cfg, "transport", "mooncake") != "mooncake":
            raise ValueError("disaggregated K3 actor requires Mooncake transport")
        if not self.config.mooncake.master_server_address:
            raise ValueError("Mooncake master_server_address must be set before actor startup")

        try:
            import ray

            assigned_gpu_ids = [int(gpu) for gpu in ray.get_gpu_ids()]
        except Exception:
            assigned_gpu_ids = []
        gpu_ids = assigned_gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        if len(gpu_ids) != teacher_cfg.tensor_parallel_size:
            raise RuntimeError(
                f"replica {replica_index} received {len(gpu_ids)} GPUs, "
                f"expected TP={teacher_cfg.tensor_parallel_size}: {gpu_ids}"
            )

        self._engine = AtomTeacherEngine(
            model_name=teacher_cfg.model_name or self.config.policy.model_name,
            tensor_parallel_size=teacher_cfg.tensor_parallel_size,
            gpu_ids=gpu_ids,
            mooncake_config=self.config.mooncake,
            transport="mooncake",
            quantization=getattr(teacher_cfg, "quantization", ""),
            atom_config=_object_to_dict(getattr(teacher_cfg, "atom", None)),
            max_batch_size=max(1, int(self.config.policy.train_global_batch_size)),
            max_seq_len=int(self.config.policy.max_total_sequence_length),
            local_device=torch.device("cpu"),
            capture_mode=getattr(spec_cfg, "capture_mode", "postnorm"),
            aux_layer_ids=list(spec_cfg.aux_hidden_state_layer_ids or []),
            key_prefix=f"atom_ray_t{replica_index}",
            consume_hidden_states=False,
        )
        self._engine.start(
            mode="generate" if self._generate_mode == "generate" else "extract"
        )

    @staticmethod
    def _build_generated_masks(
        seq_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(total_len, dtype=torch.long).unsqueeze(0)
        lens = seq_lens.view(-1, 1)
        starts = prompt_lens.view(-1, 1).clamp(max=total_len)
        return (positions < lens).long(), (
            (positions >= starts) & (positions < lens)
        ).long()

    def process_on_policy_batch(
        self,
        batch_id: int,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Run A1 decode then A2 extraction for one whole batch."""
        if self._generate_mode != "generate":
            raise RuntimeError(
                "process_on_policy_batch requires teacher.generate_mode=generate"
            )
        teacher_cfg = self.config.algorithm.teacher
        full_ids, seq_lens, prompt_lens = self._engine.generate_tokens(
            prompt_ids,
            prompt_mask,
            max_tokens=getattr(teacher_cfg, "generate_max_tokens", 2048),
            temperature=getattr(teacher_cfg, "generate_temperature", 0.0),
        )
        total_len = int(seq_lens.max().item())
        full_ids = full_ids[:, :total_len].contiguous()
        attention_mask, loss_mask = self._build_generated_masks(
            seq_lens, prompt_lens, total_len,
        )
        manifest = self._engine.extract_hidden_state_manifest(
            full_ids, attention_mask,
        )
        manifest.update(
            {
                "batch_id": int(batch_id),
                "input_ids": full_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "replica_index": self.replica_index,
            }
        )
        return manifest

    def process_prefill_batch(
        self,
        batch_id: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Extract teacher states for a response already stored in the dataset."""
        if self._generate_mode != "prefill":
            raise RuntimeError(
                "process_prefill_batch requires teacher.generate_mode=prefill"
            )
        if (
            input_ids.shape != attention_mask.shape
            or input_ids.shape != loss_mask.shape
        ):
            raise ValueError(
                "input_ids, attention_mask, and loss_mask must have identical shapes"
            )
        manifest = self._engine.extract_hidden_state_manifest(
            input_ids.contiguous(),
            attention_mask.contiguous(),
        )
        manifest.update(
            {
                "batch_id": int(batch_id),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "replica_index": self.replica_index,
            }
        )
        return manifest

    def export_static_weights(self, output_path: str) -> str:
        """Write one shared copy for the draft ranks; avoid Ray tensor transfer."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
        norm_weight, norm_eps = self._engine.get_norm_weight()
        torch.save(
            {
                "lm_head_weight": self._engine.get_lm_head_weight(),
                "embed_weight": self._engine.get_embed_weight(),
                "norm_weight": norm_weight,
                "norm_eps": norm_eps,
            },
            tmp,
        )
        os.replace(tmp, path)
        return str(path)

    def identity(self) -> dict[str, Any]:
        return {
            "replica_index": self.replica_index,
            "hostname": socket.gethostname(),
            "gpu_ids": list(self._engine._gpu_ids),
        }

    def shutdown(self) -> None:
        self._engine.shutdown()


__all__ = ["AtomTeacherRayActor"]
