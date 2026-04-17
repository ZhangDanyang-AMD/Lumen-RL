"""DataProto: tensor-dict batch protocol for cross-worker communication."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class DataProto:
    """A tensor-dict batch that flows between controller and workers.

    Inspired by VeRL's DataProto. All cross-worker data exchange uses this
    format. Tensors are stored on CPU for serialization; workers move them
    to GPU as needed.
    """

    def __init__(self, tensors: dict[str, torch.Tensor] | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.tensors: dict[str, torch.Tensor] = tensors or {}
        self.meta: dict[str, Any] = meta or {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.tensors[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def __len__(self) -> int:
        if not self.tensors:
            return 0
        first_key = next(iter(self.tensors))
        return self.tensors[first_key].shape[0]

    @property
    def batch_size(self) -> int:
        return len(self)

    def keys(self) -> list[str]:
        return list(self.tensors.keys())

    def to(self, device: str | torch.device) -> "DataProto":
        """Move all tensors to the given device."""
        return DataProto(
            tensors={k: v.to(device) for k, v in self.tensors.items()},
            meta=self.meta.copy(),
        )

    def cpu(self) -> "DataProto":
        return self.to("cpu")

    def cuda(self, device: int = 0) -> "DataProto":
        return self.to(f"cuda:{device}")

    def split(self, num_chunks: int) -> list["DataProto"]:
        """Split batch into num_chunks equal parts along dim 0."""
        if self.batch_size == 0:
            return [DataProto(meta=self.meta.copy()) for _ in range(num_chunks)]

        result = []
        for i in range(num_chunks):
            start = i * self.batch_size // num_chunks
            end = (i + 1) * self.batch_size // num_chunks
            chunk_tensors = {k: v[start:end] for k, v in self.tensors.items()}
            result.append(DataProto(tensors=chunk_tensors, meta=self.meta.copy()))
        return result

    @staticmethod
    def merge(protos: list["DataProto"]) -> "DataProto":
        """Merge a list of DataProtos by concatenating tensors along dim 0.

        All non-empty chunks must share the same set of tensor keys. A
        ``ValueError`` is raised if keys are inconsistent across chunks.
        """
        if not protos:
            return DataProto()

        non_empty = [p for p in protos if p.batch_size > 0]
        if not non_empty:
            return DataProto(meta=protos[0].meta.copy())

        ref_keys = set(non_empty[0].keys())
        for idx, p in enumerate(non_empty[1:], start=1):
            chunk_keys = set(p.keys())
            if chunk_keys != ref_keys:
                missing = ref_keys - chunk_keys
                extra = chunk_keys - ref_keys
                raise ValueError(
                    f"DataProto.merge: chunk {idx} keys mismatch. "
                    f"missing={missing or '{}'}, extra={extra or '{}'}"
                )

        merged_tensors = {}
        for key in ref_keys:
            merged_tensors[key] = torch.cat([p[key] for p in non_empty], dim=0)

        return DataProto(tensors=merged_tensors, meta=non_empty[0].meta.copy())

    def select(self, keys: list[str]) -> "DataProto":
        """Return a new DataProto with only the specified tensor keys."""
        return DataProto(
            tensors={k: self.tensors[k] for k in keys if k in self.tensors},
            meta=self.meta.copy(),
        )

    def update(self, other: "DataProto") -> None:
        """Update tensors and meta from another DataProto."""
        self.tensors.update(other.tensors)
        self.meta.update(other.meta)

    def mini_batches(self, batch_size: int) -> Iterator["DataProto"]:
        """Yield mini-batches of the given size."""
        for start in range(0, self.batch_size, batch_size):
            end = min(start + batch_size, self.batch_size)
            chunk_tensors = {k: v[start:end] for k, v in self.tensors.items()}
            yield DataProto(tensors=chunk_tensors, meta=self.meta.copy())

    def add_router_distributions(self, layer_idx: int, logits: torch.Tensor) -> None:
        """Add MoE router distribution for a given layer (R3 support)."""
        key = f"router_dist_layer_{layer_idx}"
        self.tensors[key] = logits.cpu()

    def get_router_distributions(self) -> dict[int, torch.Tensor]:
        """Retrieve all stored router distributions keyed by layer index."""
        result: dict[int, torch.Tensor] = {}
        prefix = "router_dist_layer_"
        for key, tensor in self.tensors.items():
            if key.startswith(prefix):
                layer_idx = int(key[len(prefix):])
                result[layer_idx] = tensor
        return result

    def has_router_distributions(self) -> bool:
        return any(k.startswith("router_dist_layer_") for k in self.tensors)
