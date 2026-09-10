"""DSv3 under tensor and expert parallelism.

The engine reconstructs full tensors from ``sharded_state_dict()`` metadata
before the weight bridge ever runs, so the property that matters under TP/EP is
that MLA describes itself correctly there. Wrong metadata would not raise -- it
would hand the rollout a mis-assembled attention block.

Needs 2 GPUs and a real process group, so it runs as a torchrun subprocess and
skips when the hardware is not there.
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


@pytest.mark.gpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for TP=2/EP=2")
def test_mla_describes_itself_correctly_under_tp2_ep2():
    """MLA global shapes and shard axes must match what the gather assumes.

    Asserted inside the worker on every rank: q/kv down-projections replicated,
    up-projections column-parallel on dim 0, output row-parallel on dim 1, fused
    LoRA norms replicated, experts split by EP, router replicated.
    """
    env = dict(os.environ)
    env["LUMENRL_ROOT"] = str(REPO_ROOT)
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")

    proc = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=2", "--master_port=29607",
            str(WORKER),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert "DSV3_SHARD_OK" in proc.stdout, (
        f"sharded checks failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-2500:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2500:]}"
    )
