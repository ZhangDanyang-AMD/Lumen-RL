"""DSv3 under tensor, expert and pipeline parallelism.

The engine reconstructs full tensors from ``sharded_state_dict()`` metadata and
relabels stage-local layer numbers before the weight bridge ever runs, so the
properties that matter under model parallelism are that MLA describes itself
correctly and that each pipeline stage publishes global layer numbers. Neither
would raise if wrong -- both would hand the rollout a mis-assembled model.

Needs 2 GPUs and a real process group, so these run as torchrun subprocesses and
skip when the hardware is not there.
"""

import os
import pathlib
import subprocess
import sys

import pytest

WORKER = pathlib.Path(__file__).parent / "_dsv3_shard_worker.py"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _run_worker(port: str, **topology: int) -> None:
    env = dict(os.environ)
    env["LUMENRL_ROOT"] = str(REPO_ROOT)
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    env.update({f"DSV3_{k.upper()}": str(v) for k, v in topology.items()})

    proc = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=2", f"--master_port={port}",
            str(WORKER),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert "DSV3_SHARD_OK" in proc.stdout, (
        f"{topology} checks failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-2500:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2500:]}"
    )


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for TP=2/EP=2")
def test_mla_describes_itself_correctly_under_tp2_ep2():
    """MLA global shapes and shard axes must match what the gather assumes.

    Asserted inside the worker on every rank: q/kv down-projections replicated,
    up-projections column-parallel on dim 0, output row-parallel on dim 1, fused
    LoRA norms replicated, experts split by EP, router replicated.
    """
    _run_worker("29607", tp=2, pp=1, ep=2, layers=3)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for PP=2")
def test_pipeline_stages_publish_global_layer_numbers():
    """Each stage must publish its layers under global numbers, dense layer included.

    DSv3 always carries a ``moe_layer_freq`` list, which puts it on
    ``TransformerBlock.sharded_state_dict``'s heterogeneous branch -- the case
    where deriving the stage offset from checkpoint metadata silently returned 0
    and later stages republished under stage 0's layer numbers. 4 layers over 2
    stages with ``first_k_dense_replace=1`` puts the dense layer on stage 0 and
    an all-MoE stage 1, so a collapsed offset shows up as both a coverage gap and
    a dense/MoE mismatch.
    """
    _run_worker("29608", tp=1, pp=2, ep=1, layers=4)
