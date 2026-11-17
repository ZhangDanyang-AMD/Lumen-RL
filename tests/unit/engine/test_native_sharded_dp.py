"""Sharded ``megatron_cfg.data_parallel`` (``torch_fsdp2``, ``megatron_fsdp``) on the
native engine."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine
from lumenrl.engine.training.model_registry import ModelCaps

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER = HERE / "_native_sharded_dp_worker.py"


def _visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


class _Spec:
    name = "qwen3"
    caps = ModelCaps()


def _engine(**ec) -> MegatronNativeEngine:
    engine = MegatronNativeEngine.__new__(MegatronNativeEngine)
    engine.engine_config = ec
    engine._spec = _Spec()
    engine._pp = int(ec.get("pipeline_model_parallel_size", 1))
    engine._ep = int(ec.get("expert_model_parallel_size", 1))
    engine._overlap_param_gather = bool(ec.get("overlap_param_gather", False))
    return engine


class _Cfg:
    gradient_accumulation_fusion = False
    fp8 = None


MODES = ["torch_fsdp2", "megatron_fsdp"]


@pytest.mark.parametrize("mode", MODES)
def test_supported_layout_passes(mode):
    _engine()._check_sharded_dp(_Cfg(), mode)


def test_megatron_fsdp_allows_expert_parallel_and_overlapped_gathers():
    _engine(expert_model_parallel_size=8, overlap_param_gather=True)._check_sharded_dp(
        _Cfg(), "megatron_fsdp")


@pytest.mark.parametrize("mode, ec, problem", [
    ("torch_fsdp2", {"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=2"),
    ("megatron_fsdp", {"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=2"),
    ("torch_fsdp2", {"expert_model_parallel_size": 8}, "expert_model_parallel_size=8"),
    ("torch_fsdp2", {"overlap_param_gather": True}, "overlap_param_gather"),
    ("torch_fsdp2", {"optimizer_cpu_offload": True}, "optimizer_cpu_offload"),
    ("megatron_fsdp", {"optimizer_cpu_offload": True}, "optimizer_cpu_offload"),
])
def test_unsupported_layouts_are_named(mode, ec, problem):
    with pytest.raises(ValueError, match=f"{mode} does not support: .*{problem}"):
        _engine(**ec)._check_sharded_dp(_Cfg(), mode)


@pytest.mark.parametrize("mode", MODES)
def test_fusion_and_fp8_are_rejected(mode):
    cfg = _Cfg()
    cfg.gradient_accumulation_fusion, cfg.fp8 = True, "e4m3"
    with pytest.raises(ValueError, match="gradient_accumulation_fusion.*FP8"):
        _engine()._check_sharded_dp(cfg, mode)


@pytest.mark.parametrize("flag", ["_fsdp2", "_mfsdp"])
def test_dist_checkpoint_refused_when_sharded(flag):
    engine = _engine()
    setattr(engine, flag, True)
    with pytest.raises(NotImplementedError, match="sharded data_parallel"):
        engine._dist_sharded_state_dict(is_loading=False)


def _run_worker(mode, nproc, tp, strategy, port, tmp_path):
    env = dict(os.environ, LUMENRL_ROOT=str(REPO_ROOT), SHARDED_DP=mode,
               SHARDED_DP_TP=str(tp), SHARDED_DP_STRATEGY=strategy,
               SHARDED_DP_MODEL_DIR=str(tmp_path / "tiny"))
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
         f"--master_port={port}", str(WORKER)],
        capture_output=True, text=True, timeout=1800, env=env, cwd=str(REPO_ROOT),
    )
    assert "SHARDED_DP_OK" in proc.stdout, (
        f"{WORKER.name} failed (rc={proc.returncode}) {mode} {strategy} nproc={nproc} tp={tp}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )


@pytest.mark.multigpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nproc, tp", [(1, 1), (2, 1), (4, 1), (4, 2)])
def test_sharded_dp_matches_ddp(mode, nproc, tp, tmp_path):
    if _visible_gpus() < nproc:
        pytest.skip(f"needs {nproc} GPUs")
    _run_worker(mode, nproc, tp, "optim_grads_params",
                29760 + 4 * nproc + tp + 20 * MODES.index(mode), tmp_path)


@pytest.mark.multigpu
@pytest.mark.parametrize("nproc, tp", [(2, 1), (4, 2)])
def test_megatron_fsdp_optim_grads_matches_ddp(nproc, tp, tmp_path):
    """ZeRO-2: whole weights, sharded gradients and optimizer state."""
    if _visible_gpus() < nproc:
        pytest.skip(f"needs {nproc} GPUs")
    _run_worker("megatron_fsdp", nproc, tp, "optim_grads", 29820 + 4 * nproc + tp, tmp_path)
