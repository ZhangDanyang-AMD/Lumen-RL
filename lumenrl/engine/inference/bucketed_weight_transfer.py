"""Bucketed weight transfer via ZMQ + CUDA IPC (or shared-memory fallback).

Vendored from verl
(``verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py``) and made
self-contained so LumenRL does not depend on the verl source tree at runtime.
Behaviour is intentionally identical: a training worker (sender) packs weight
tensors into a fixed-size GPU buffer shared via a CUDA IPC handle and streams
them bucket-by-bucket over a ZMQ REQ/REP socket to the colocated vLLM worker
(receiver), which views tensors directly out of the shared buffer and loads
them into the model.

torch.cuda works for both NVIDIA and AMD/ROCm builds, so no device abstraction
layer is required here.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from multiprocessing import shared_memory
from typing import Callable, Iterable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "INFO"))


def _debug_enabled() -> bool:
    return os.getenv("LUMEN_WEIGHT_TRANSFER_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _debug_timeout_ms() -> int:
    return int(os.getenv("LUMEN_WEIGHT_TRANSFER_ZMQ_TIMEOUT_MS", "0") or 0)


def _device_name() -> str:
    return "cuda"


def _device_id() -> int:
    return torch.cuda.current_device()


def _sync() -> None:
    torch.cuda.synchronize()


def _debug_log(role: str, message: str, **kwargs) -> None:
    if not _debug_enabled():
        return
    fields = {"pid": os.getpid(), "device": f"{_device_name()}:{_device_id()}", **kwargs}
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[weight-ipc][%s] %s %s", role, message, extra)


def _configure_socket(socket, role: str, zmq_handle: str) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    timeout_ms = _debug_timeout_ms()
    if timeout_ms > 0:
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    _debug_log(role, "socket configured", zmq_handle=zmq_handle, timeout_ms=timeout_ms)


async def ensure_async_iterator(iterable):
    """Yield from either an async iterator or a plain (sync) iterable."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# Adapted from vllm/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # Patch the device id so it matches the receiver's current device even
        # when sender and receiver have different CUDA_VISIBLE_DEVICES mappings
        # (crucial on ROCm where Ray pins each worker to its own device index).
        list_args[6] = device_id
    return func(*list_args)


def create_shared_memory(size: int, name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)
    return tensor, shm


class BucketedWeightSender:
    """Send model weights via bucketed IPC transfer over ZMQ (REQ side)."""

    def __init__(self, zmq_handle: str, bucket_size_mb: int = 512, use_shm: bool = False):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights: Iterable) -> None:
        """Stream ``(name, tensor)`` pairs to the receiver bucket-by-bucket."""
        try:
            self._init_socket()
            self._init_buffer()

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in ensure_async_iterator(weights):
                weight = weight.contiguous()
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    _sync()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({tuple(weight.shape)}, {weight.dtype}) exceeds the "
                        f"bucket size; increase update_weights_bucket_megabytes "
                        f"({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(
                    weight.view(-1).view(torch.uint8), non_blocking=True
                )
                offset += weight.nbytes

            _sync()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        _configure_socket(self.socket, "sender", self.zmq_handle)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self) -> None:
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(
                self.bucket_size, dtype=torch.uint8, device=f"{_device_name()}:{_device_id()}"
            )
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            shm_name = f"lumen_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            self.socket.send_pyobj({"name": shm_name, "size": self.bucket_size})

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor) -> None:
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {
            name: {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
                "handle": handle,
            }
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()


class BucketedWeightReceiver:
    """Receive model weights via bucketed IPC transfer over ZMQ (REP side)."""

    def __init__(self, zmq_handle: str, device: torch.device, use_shm: bool = False):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: Callable) -> None:
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = (
                        meta["shape"], meta["dtype"], meta["offset"], meta["handle"],
                    )
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                _sync()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        self.socket = self.zmq_context.socket(zmq.REP)
        _configure_socket(self.socket, "receiver", self.zmq_handle)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self) -> None:
        started_at = time.time()
        comm_metadata = self.socket.recv_pyobj()
        _debug_log("receiver", "got initial metadata", elapsed_s=f"{time.time() - started_at:.3f}")
        buffer, shm = None, None
        if not self.use_shm:
            buffer = rebuild_ipc(comm_metadata, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        _sync()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
