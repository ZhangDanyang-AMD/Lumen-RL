"""The RLHFModelRunner every ATOM replica runs, which can disable aiter custom all-reduce.

It also fixes three ways ATOM's weight sync silently or loudly fails for
DeepSeek-V4.1: the factory answers class-level capability questions for the
runner it builds (``_ResolvesToRunner``), ``clear_kv_cache`` tolerates a runner
with no ``kv_cache``, and native group32 FP8 weights are requantized
(``quantize_native_group``). With ``LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS=1`` a sleep
moves the weights to host memory instead of discarding them (``offload_weights``),
for a trainer that does not sync every weight.

When ``LUMENRL_DISABLE_CUSTOM_AR=1`` is set, monkey-patches aiter so that
``CustomAllreduce`` is never instantiated.  This avoids the
``hipIpcGetMemHandle`` / ``exit(0)`` crash on ROCm where
``hipDeviceMallocUncached`` memory does not support IPC handles.

The patch runs inside ``_setup_device_and_distributed``, *after* the normal
import chain has initialized aiter, but *before* ``init_dist_env`` is called
(which would trigger the fatal HIP IPC path).

Usage: pass ``runner_qualname="lumenrl.engine.inference.model_runner_nocustomar.NoCustomARModelRunner"``
to ``AsyncLLMEngine`` (via ``engine_kwargs``).
"""

import logging
import os
import sys

logger = logging.getLogger("atom")

_patched = False


def _patch_disable_custom_ar():
    """Permanently disable aiter custom all-reduce.

    Uses sys.modules to access aiter submodules (already loaded by
    RLHFModelRunner's import chain) instead of ``import aiter.dist.parallel_state``
    which would trigger a circular import through aiter.__init__.
    """
    global _patched
    if _patched:
        return
    _patched = True

    _ps = sys.modules["aiter.dist.parallel_state"]
    _comm = sys.modules["aiter.ops.communication"]

    _ps._ENABLE_CUSTOM_ALL_REDUCE = False
    os.environ["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"

    def _noop(enable: bool) -> None:
        pass

    _ps.set_custom_all_reduce = _noop
    _comm.set_custom_all_reduce = _noop

    import torch
    _GC = _ps.GroupCoordinator

    def _all_gather_nccl(self, input_, dim=0):
        world_size = self.world_size
        input_size = input_.size()
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group,
        )
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1:]
        )

    _GC._all_gather_out_place = _all_gather_nccl
    logger.info("Patched _all_gather_out_place to NCCL fallback")

    def _init_dist_env_no_ca(
        tensor_model_parallel_size,
        rankID,
        backend="cpu:gloo,cuda:nccl",
        distributed_init_method="env://",
        local_rank=-1,
        data_parallel_size=1,
        data_parallel_rank=0,
        **extra_kwargs,
    ):
        pipeline_model_parallel_size = 1
        world_size = pipeline_model_parallel_size * tensor_model_parallel_size
        _ps.init_distributed_environment(
            world_size=world_size,
            rank=rankID,
            distributed_init_method=distributed_init_method,
            backend=backend,
            local_rank=local_rank,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
        )
        _ps.ensure_model_parallel_initialized(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            data_parallel_size=data_parallel_size,
        )
        logger.info(
            "init_dist_env (no custom AR): rank=%d/%d ready",
            rankID, tensor_model_parallel_size,
        )

    _comm.init_dist_env = _init_dist_env_no_ca

    _aiter = sys.modules.get("aiter")
    if _aiter is not None:
        _aiter.init_dist_env = _init_dist_env_no_ca

    logger.info("Patched aiter: custom all-reduce permanently disabled (hipIpcGetMemHandle workaround)")


def quantize_native_group(weight, rows: int, fp8_dtype):
    """``weight`` [N, K] in a native group32 layout: ``fp8_dtype`` with one
    power-of-two scale per ``rows`` x 32 block, the scales E8M0 [N / rows, K / 32]
    (DeepSeek-V4.1 stores its linears so with ``rows`` 32). A scale is
    ``2 ** ceil(log2(amax / fp8_max))``, as the release and ATOM's
    ``quantize_fp8`` round it, so a dequantized release weight comes back as the
    same values."""
    import torch

    n, k = weight.shape
    fp8_max = torch.finfo(fp8_dtype).max
    blocks = weight.float().view(n // rows, rows, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    mantissa, exponent = torch.frexp(amax / fp8_max)  # an exact power of two stays put
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    scale = torch.ldexp(torch.ones_like(amax), exponent)
    q = (blocks / scale[:, None, :, None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q.view(n, k), scale.to(torch.float8_e8m0fnu)


def _sleep_keeps_weights() -> bool:
    return os.environ.get("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS", "0") in ("1", "true", "True")


def host_weight_buffers(runner) -> dict:
    """Host tensors shaped like every parameter of ``runner.model``: views of one
    pinned allocation, made when the runner starts. Pinned per parameter at the
    first sleep instead, ~70 GB a card of V4.1 took past ATOM's timeout for
    `release_memory` on a busy host."""
    import torch

    params = list(runner.model.named_parameters())
    sizes = [-(-p.numel() * p.element_size() // 256) * 256 for _, p in params]  # aligned views
    flat = torch.empty(sum(sizes), dtype=torch.uint8, pin_memory=torch.cuda.is_available())
    views, offset = {}, 0
    for (name, p), size in zip(params, sizes):
        nbytes = p.numel() * p.element_size()
        views[name] = flat[offset : offset + nbytes].view(p.dtype).view(p.shape)
        offset += size
    return views


def offload_weights(runner) -> None:
    """Every parameter of ``runner.model`` to host memory, the device copy freed.

    ATOM's sleep discards the weights and counts on the next sync to rewrite
    every one; a trainer that syncs only what it trains (DeepSeek-V4.1 keeps its
    routed experts and indexers frozen) would wake to empty ones."""
    import torch
    from atom.rollout.memory_manager import release_cudagraphs

    release_cudagraphs(runner)
    host = getattr(runner, "_host_weights", None)
    if host is None:
        host = runner._host_weights = host_weight_buffers(runner)
    for name, p in runner.model.named_parameters():
        host[name].copy_(p.data, non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for p in runner.model.parameters():
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    for module in runner._weight_memory_modules():
        module.release_weight_memory()
    runner._weights_offloaded = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def restore_weights(runner) -> None:
    """The parameters ``offload_weights`` moved, back on the device as they were."""
    import torch

    for name, p in runner.model.named_parameters():
        p.data = runner._host_weights[name].to(runner.device, non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runner._weights_offloaded = False
    for module in runner._weight_memory_modules():
        module.resume_weight_memory()


def rank_shard(tensor, shape: tuple, rank: int, world: int):
    """This TP rank's part of a full ``tensor`` for a parameter of ``shape``."""
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    for dim, (full, part) in enumerate(zip(tensor.shape, shape)):
        if full == part * world and tensor.dim() == len(shape):
            return tensor.narrow(dim, rank * part, part)
    raise ValueError(f"a {tuple(tensor.shape)} weight does not shard onto {tuple(shape)} over TP {world}")


class _ResolvesToRunner(type):
    """Class-level lookups on the factory answer for the runner it builds.

    ATOM decides what a deployment may do from the runner CLASS before any
    runner exists: ``atom.mono.runtime.deployment._takes_weight_updates`` asks
    ``hasattr(cls, "update_weights")``. Asked of a factory with no such
    attribute, the answer was no, so DeepSeek-V4.1 decoded through "mono" --
    whose weight copies no sync reaches, under parameter names no sync matches
    -- and every weight update applied nothing.
    """

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(cls.runner_class(), name)


class NoCustomARModelRunner(metaclass=_ResolvesToRunner):
    """Lazy-import wrapper that creates a patched RLHFModelRunner subclass.

    The real base class is resolved inside the spawned subprocess (avoids
    importing ATOM/aiter in the driver).  The subclass applies the aiter
    patch inside ``_setup_device_and_distributed`` — after aiter is fully
    imported but before ``init_dist_env`` triggers the HIP IPC path.
    """

    _real_cls = None

    @classmethod
    def runner_class(cls):
        if cls._real_cls is not None:
            return cls._real_cls
        from atom.rollout.model_runner_ext import RLHFModelRunner

        class _PatchedRunner(RLHFModelRunner):
            def __init__(self, rank, config, *args, **kwargs):
                super().__init__(rank, config, *args, **kwargs)
                if _sleep_keeps_weights():
                    self._host_weights = host_weight_buffers(self)

            def clear_kv_cache(self) -> bool:
                # ATOM sets `kv_cache` only when the shared paged pool is not
                # empty. A model whose attention builders own every pool
                # (DeepSeek-V4.1) never gets it, and the mixin reads it
                # unguarded, so the clear that ends every weight sync raised.
                if not hasattr(self, "kv_cache"):
                    self.kv_cache = None
                return super().clear_kv_cache()

            def _release_weights(self) -> None:
                if _sleep_keeps_weights():
                    return offload_weights(self)
                return super()._release_weights()

            def _resume_weights(self) -> None:
                if getattr(self, "_weights_offloaded", False):
                    return restore_weights(self)
                return super()._resume_weights()

            def _requantize_fp8_weight(self, module, param_name, param, tensor) -> None:
                # ATOM's own knows per_1x128, per-tensor and per-token, and for any
                # other layout -- a native group32 weight, every FP8 linear of
                # DeepSeek-V4.1 -- writes nothing while the update is counted.
                rows = getattr(module, "native_a8_group_rows", None)
                if rows is None:
                    return super()._requantize_fp8_weight(module, param_name, param, tensor)
                tensor = rank_shard(tensor, tuple(param.shape), self.rank, self.world_size)
                q, scale = quantize_native_group(tensor.to(self.device), rows, param.dtype)
                # the scale rides on the weight's fence, as in ATOM's per_1x128 path
                self._copy_into_param(param, q)
                module.weight_scale.data.copy_(scale.view(module.weight_scale.dtype))
                self._post_process_fp8_weight(module, param)

            def _setup_device_and_distributed(self, rank, config):
                _dbg2 = os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True")
                disable = os.environ.get("LUMENRL_DISABLE_CUSTOM_AR", "0") in ("1", "true", "True")
                if _dbg2:
                    logger.info("[DBG] _PatchedRunner._setup_device_and_distributed: rank=%s disable_ca=%s tp=%s",
                                rank, disable, getattr(config, 'tensor_parallel_size', '?'))
                if disable:
                    _patch_disable_custom_ar()
                super()._setup_device_and_distributed(rank, config)

        cls._real_cls = _PatchedRunner
        return _PatchedRunner

    def __new__(cls, rank, *args, **kwargs):
        if os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True"):
            logger.info("[DBG] NoCustomARModelRunner: creating _PatchedRunner(rank=%s)", rank)
        return cls.runner_class()(rank, *args, **kwargs)
