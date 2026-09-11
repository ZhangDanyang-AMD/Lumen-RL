"""DSv3 bridge against a real DeepSeek-V3 checkpoint.

The unit tests build tiny synthetic configs, which cannot catch a field the
builder never reads: a synthetic config has no ``rope_scaling`` and no
``n_group``, so leaving those at Megatron's defaults looks fine. A real
checkpoint has them, and a logit comparison against the HF reference is the only
check that fails when the resulting model is subtly wrong rather than unbuildable.

``bzantium/tiny-deepseek-v3`` is production DeepSeek-V3 geometry -- hidden 7168,
128 heads, q_lora 1536, kv_lora 512, qk 128+64, v 128, moe_ffn 2048, YaRN with
factor 40 -- trimmed only to 6 layers and 8 experts, so it is 10.7 GB and fits on
one GPU. It also ships ``first_k_dense_replace=3``, exercising a multi-layer
dense prefix the tiny fixtures do not.

Fetch with::

    HF_HOME=/data/rl_data/hf python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('bzantium/tiny-deepseek-v3')"
"""

import glob
import os
import pathlib
import subprocess
import sys

import pytest

WORKER = pathlib.Path(__file__).parent / "_dsv3_parity_worker.py"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_HF_HOME = os.environ.get("HF_HOME", "/data/rl_data/hf")
_GLOB = f"{_HF_HOME}/hub/models--bzantium--tiny-deepseek-v3/snapshots/*/"


def _checkpoint() -> str | None:
    """The checkpoint path, from ``DSV3_CKPT`` or the HF cache."""
    explicit = os.environ.get("DSV3_CKPT")
    if explicit and os.path.isdir(explicit):
        return explicit
    found = sorted(glob.glob(_GLOB))
    return found[0] if found else None


def _visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


@pytest.mark.gpu
@pytest.mark.skipif(_visible_gpus() < 1, reason="needs a GPU")
@pytest.mark.skipif(_checkpoint() is None, reason=f"no DSv3 checkpoint under {_GLOB}")
def test_bridge_reproduces_the_reference_logits_in_fp32():
    """Real weights through the bridge must match the HF reference exactly in fp32.

    Run in a subprocess: it needs a NCCL process group and ~22 GB of HBM for the
    two models, and initialising Megatron's parallel state in-process would leak
    into the rest of the suite.

    This is the test that caught the RoPE and expert-group defaults -- with them
    left at Megatron's values the same comparison gave 11.4% mean relative error
    and 71.9% top-1 agreement. See ``test_dsv3_rope_and_routing.py``.
    """
    env = dict(os.environ)
    env["LUMENRL_ROOT"] = str(REPO_ROOT)
    env["DSV3_CKPT"] = _checkpoint()
    env.setdefault("HIP_VISIBLE_DEVICES", "0")

    proc = subprocess.run(
        [sys.executable, str(WORKER)],
        capture_output=True, text=True, timeout=1800, env=env, cwd=str(REPO_ROOT),
    )
    assert "DSV3_PARITY_OK" in proc.stdout, (
        f"parity failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
