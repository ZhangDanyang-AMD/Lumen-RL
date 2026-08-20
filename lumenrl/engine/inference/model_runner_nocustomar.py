"""RLHFModelRunner variant that disables aiter custom all-reduce.

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


class NoCustomARModelRunner:
    """Lazy-import wrapper that creates a patched RLHFModelRunner subclass.

    The real base class is resolved inside the spawned subprocess (avoids
    importing ATOM/aiter in the driver).  The subclass applies the aiter
    patch inside ``_setup_device_and_distributed`` — after aiter is fully
    imported but before ``init_dist_env`` triggers the HIP IPC path.
    """

    _real_cls = None

    def __new__(cls, rank, *args, **kwargs):
        _dbg = os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True")
        if cls._real_cls is None:
            if _dbg:
                logger.info("[DBG] NoCustomARModelRunner: importing RLHFModelRunner (rank=%s)", rank)
            from atom.rollout.model_runner_ext import RLHFModelRunner

            class _PatchedRunner(RLHFModelRunner):
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

        if _dbg:
            logger.info("[DBG] NoCustomARModelRunner: creating _PatchedRunner(rank=%s)", rank)
        return cls._real_cls(rank, *args, **kwargs)
