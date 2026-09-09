"""``moe_backend`` / ``linear_backend`` are only forwarded when they name a backend.

vLLM validates these against its own backend list, so forwarding a sentinel
fails the rollout at worker start with

    ValueError: moe_backend='' is not supported for unquantized MoE

which is reported from inside the vLLM worker process and does not mention
LumenRL config at all. These tests pin the two sentinels and the field's
single declaration.
"""

import ast
import collections
import inspect

from lumenrl.core.config import VLLMConfig


def _forwarded(backend: str) -> bool:
    """Mirror of the guard in ``rl_trainer._setup_ray_vllm_rollout``."""
    return str(backend) not in ("auto", "")


def test_empty_backend_is_not_forwarded():
    """The regression: default was '' and the guard only skipped 'auto'."""
    assert _forwarded("") is False


def test_auto_backend_is_not_forwarded():
    assert _forwarded("auto") is False


def test_named_backend_is_forwarded():
    assert _forwarded("triton") is True


def test_default_config_does_not_forward_a_backend():
    """A config that never mentions moe_backend must not send one to vLLM."""
    assert _forwarded(VLLMConfig().moe_backend) is False


def test_moe_backend_is_declared_exactly_once():
    """Two declarations in one dataclass silently shadow: the later one wins.

    ``VLLMConfig`` briefly carried both ``moe_backend = "auto"`` and
    ``moe_backend = ""``. Python kept the second, flipping the default for every
    model and breaking unquantized MoE rollout everywhere.
    """
    src = inspect.getsource(VLLMConfig)
    tree = ast.parse(src.lstrip())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    names = [
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    ]
    dupes = [k for k, v in collections.Counter(names).items() if v > 1]
    assert dupes == [], f"VLLMConfig declares these fields twice: {dupes}"


def test_guard_in_trainer_matches_this_contract():
    """Keep the real guard and this test's mirror from drifting apart."""
    from lumenrl.trainer import rl_trainer

    src = inspect.getsource(rl_trainer)
    assert 'str(vcfg.moe_backend) not in ("auto", "")' in src
