"""Synchronize weights between training and rollout workers."""

from __future__ import annotations

import logging
from typing import Any

import ray
import torch

from lumenrl.engine.inference.weight_loader import DynamicWeightLoader

logger = logging.getLogger(__name__)


def _is_ray_actor(obj: Any) -> bool:
    return hasattr(obj, "remote") and callable(getattr(obj, "remote"))


def _gather_state_dicts_from_ray(
    src_workers: list[Any],
    aggregate: str = "rank0",
) -> dict[str, torch.Tensor]:
    """Pull state dicts from Ray actors and optionally aggregate."""
    refs = []
    for w in src_workers:
        if hasattr(w, "get_state_dict"):
            refs.append(w.get_state_dict.remote())  # type: ignore[union-attr]
        else:
            raise TypeError("Ray source worker must expose get_state_dict().")
    payloads = ray.get(refs)
    if aggregate == "mean" and len(payloads) > 1:
        keys = payloads[0].keys()
        merged: dict[str, torch.Tensor] = {}
        for k in keys:
            merged[k] = torch.stack([p[k].float() for p in payloads]).mean(dim=0).to(
                payloads[0][k].dtype
            )
        return merged
    return payloads[0]


def _push_state_dict_ray(dst_workers: list[Any], state_dict: dict[str, torch.Tensor]) -> None:
    refs = []
    for w in dst_workers:
        if hasattr(w, "update_weights"):
            refs.append(w.update_weights.remote(state_dict))  # type: ignore[union-attr]
        else:
            raise TypeError("Ray destination worker must expose update_weights().")
    ray.get(refs)


class WeightSyncManager:
    """Transfer weights between source and destination worker groups.

    Supports:

    * ``ray_object_store`` — serialize via ``get_state_dict`` / ``update_weights`` Ray calls.
    * ``nccl`` — in-process ``torch.distributed`` broadcast when all ranks share a process
      group (co-located hybrid training).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def transfer(self, src_workers: list[Any], dst_workers: list[Any]) -> None:
        """Move weights from ``src_workers`` to ``dst_workers``."""
        if not src_workers or not dst_workers:
            logger.warning("WeightSyncManager.transfer: empty src or dst; nothing to do.")
            return

        mode = str(self._config.get("mode", "ray_object_store")).lower()
        fp8_cfg = self._config.get("fp8_on_the_fly")
        aggregate = str(self._config.get("aggregate", "rank0"))

        if mode == "nccl":
            self._transfer_nccl(src_workers, dst_workers, fp8_cfg)
            return

        if mode != "ray_object_store":
            logger.warning("Unknown weight sync mode %s; falling back to ray_object_store.", mode)

        if _is_ray_actor(src_workers[0]):
            state_dict = _gather_state_dicts_from_ray(src_workers, aggregate=aggregate)
            cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
            if fp8_cfg:
                # Apply FP8 transform once on controller before fan-out saves bandwidth
                class _Dummy:
                    def update_weights(self, sd: dict[str, torch.Tensor]) -> None:
                        self.sd = sd

                dummy = _Dummy()
                DynamicWeightLoader.load_weights(dummy, cpu_state, fp8_config=fp8_cfg)
                cpu_state = dummy.sd  # type: ignore[attr-defined]
            _push_state_dict_ray(dst_workers, cpu_state)
            logger.info(
                "WeightSyncManager: Ray transfer complete (%d src -> %d dst, %d tensors).",
                len(src_workers),
                len(dst_workers),
                len(cpu_state),
            )
            return

        # In-process path (co-located Python objects)
        src_obj = src_workers[0]
        if not hasattr(src_obj, "get_state_dict"):
            raise TypeError("In-process src worker must implement get_state_dict().")
        state_dict = src_obj.get_state_dict()  # type: ignore[union-attr]
        if fp8_cfg:

            class _Dummy:
                def update_weights(self, sd: dict[str, torch.Tensor]) -> None:
                    self.sd = sd

            dummy = _Dummy()
            DynamicWeightLoader.load_weights(dummy, state_dict, fp8_config=fp8_cfg)
            state_dict = dummy.sd  # type: ignore[attr-defined]

        for dst in dst_workers:
            if not hasattr(dst, "update_weights"):
                raise TypeError("In-process dst worker must implement update_weights().")
            dst.update_weights(state_dict)  # type: ignore[union-attr]
        logger.info("WeightSyncManager: in-process transfer to %d workers.", len(dst_workers))

    def _transfer_nccl(
        self,
        src_workers: list[Any],
        dst_workers: list[Any],
        fp8_cfg: dict[str, Any] | None,
    ) -> None:
        """Co-located transfer: reuse tensors in-process; optional NCCL broadcast extension."""
        try:
            import torch.distributed as dist
        except ImportError:
            dist = None  # type: ignore[assignment]

        leader = src_workers[0]
        if not hasattr(leader, "get_state_dict"):
            raise TypeError("NCCL path expects in-process leader with get_state_dict().")
        state_dict = leader.get_state_dict()  # type: ignore[union-attr]
        if fp8_cfg:

            class _Dummy:
                def update_weights(self, sd: dict[str, torch.Tensor]) -> None:
                    self.sd = sd

            dummy = _Dummy()
            DynamicWeightLoader.load_weights(dummy, state_dict, fp8_config=fp8_cfg)
            state_dict = dummy.sd  # type: ignore[attr-defined]

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            logger.info(
                "WeightSyncManager._transfer_nccl: process group size=%d — "
                "performing in-process copy; extend here for tensor-parallel sharding.",
                dist.get_world_size(),
            )

        for dst in dst_workers:
            dst.update_weights(state_dict)  # type: ignore[union-attr]
        logger.info("WeightSyncManager: co-located (NCCL mode) transfer complete.")
