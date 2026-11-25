"""Owned, versioned weight snapshots for phase-separated rollout transfers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.distributed.tensor import DTensor


def _tensor_stamp(tensor: torch.Tensor) -> tuple[int, int, int]:
    local = getattr(tensor, "_local_tensor", tensor)
    return (int(tensor._version), int(local._version), int(local.data_ptr()))


@dataclass
class PreparedWeightTransfer:
    version: int
    scheduler_epoch: int | None
    weights: list[tuple[str, torch.Tensor]]
    sources: list[tuple[torch.Tensor, tuple[int, int, int]]]
    wire_bytes: int
    estimated_peak_bytes: int
    cpu_only: bool = False

    def stage_on_cpu(self) -> None:
        """Finish host staging before publication to any receiver."""
        if not self.weights:
            raise RuntimeError("Prepared weight transfer has already been released")
        if self.cpu_only:
            return
        # Preparation already completed every producer gather/cast. Blocking
        # copies here do not interleave with those collectives or receiver work.
        host_weights = [(name, tensor.detach().to(device="cpu", copy=True)) for name, tensor in self.weights]
        self.weights = host_weights
        self.cpu_only = True

    def validate(self, version: int, scheduler_epoch: int | None) -> None:
        if version != self.version or scheduler_epoch != self.scheduler_epoch:
            raise RuntimeError("Prepared weight transfer version/optimizer epoch is stale")
        if any(_tensor_stamp(tensor) != stamp for tensor, stamp in self.sources):
            raise RuntimeError("Prepared weight transfer source tensors changed")
        if not self.weights:
            raise RuntimeError("Prepared weight transfer has already been released")
        if self.cpu_only and any(tensor.device.type != "cpu" for _, tensor in self.weights):
            raise RuntimeError("CPU-prepared transfer contains a non-CPU tensor")

    def release(self) -> None:
        """Drop only this snapshot's references; do not purge allocator caches."""
        self.weights.clear()
        self.sources.clear()


@torch.no_grad()
def prepare_weight_transfer(
    params: Iterable[tuple[str, torch.Tensor]],
    *,
    version: int,
    scheduler_epoch: int | None,
    keep_fp32: bool,
    max_bytes: int,
    available_bytes: int | None,
    synchronize: Callable[[], None],
) -> PreparedWeightTransfer:
    """Gather in common order, retain sources through completion, then publish.

    The caller must invoke this on every participating actor before starting
    receivers. ``synchronize`` completes the producer's captured current stream
    after its gather/cast dependencies. No live master or optimizer is mutated.
    """
    if type(version) is not int or version < 0:
        raise ValueError("Weight transfer version must be a nonnegative integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("Prepared weight byte limit must be a positive integer")
    entries = list(params)
    if not entries or len({name for name, _ in entries}) != len(entries):
        raise ValueError("Weight transfer requires nonempty, uniquely named tensors")
    if any(not isinstance(tensor, torch.Tensor) for _, tensor in entries):
        raise TypeError("Weight transfer entries must be tensors")

    # Count full gathered sources, owned wire copies, and one largest gather
    # scratch allowance. Counting aliases separately is deliberately conservative.
    source_bytes = [tensor.numel() * tensor.element_size() for _, tensor in entries]
    wire_bytes = sum(
        tensor.numel() * (torch.bfloat16.itemsize if tensor.dtype == torch.float32 and not keep_fp32 else tensor.element_size())
        for _, tensor in entries
    )
    gathered_bytes = sum(size for (_, tensor), size in zip(entries, source_bytes) if isinstance(tensor, DTensor))
    estimated_peak = gathered_bytes + wire_bytes + max(source_bytes)
    if estimated_peak > max_bytes:
        raise ValueError(f"Prepared weight estimate {estimated_peak} exceeds byte limit {max_bytes}")
    if available_bytes is not None and estimated_peak > available_bytes:
        raise RuntimeError(f"Prepared weight estimate {estimated_peak} exceeds free device bytes {available_bytes}")

    weights: list[tuple[str, torch.Tensor]] = []
    full_sources: list[torch.Tensor] = []
    guards = [(tensor, _tensor_stamp(tensor)) for _, tensor in entries]
    try:
        for name, tensor in entries:
            full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            full_sources.append(full)
            detached = full.detach()
            # Match the existing wire rule: only FP32 is converted to BF16.
            # Other floating types and integer/bool buffers keep their dtype.
            wire = detached.to(torch.bfloat16) if detached.dtype == torch.float32 and not keep_fp32 else detached.clone()
            weights.append((name, wire.contiguous()))
        synchronize()
        snapshot = PreparedWeightTransfer(version, scheduler_epoch, weights, guards, wire_bytes, estimated_peak)
        snapshot.validate(version, scheduler_epoch)
        return snapshot
    except BaseException:
        weights.clear()
        guards.clear()
        raise
    finally:
        # FP32 gather temporaries remain alive until the producer stream has
        # completed on the successful path; wire tensors outlive all sender ACKs.
        full_sources.clear()
