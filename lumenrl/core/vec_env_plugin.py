"""Plugin contract for vectorized-environment PPO (``VecEnvPPOTrainer``).

LumenRL core owns the RL machinery (rollout loop, GAE, PPO loss, optimizer, adaptive learning rate,
checkpoints). A project supplies the simulator and the network through a module that exposes
``VEC_ENV_PLUGIN``, a :class:`VecEnvPlugin`. The module is named in the config as ``vec_env.plugin``;
``vec_env.task`` picks a task when one plugin serves several.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import Tensor


class VecEnv(Protocol):
    """A batch of ``num_envs`` environments that auto-reset, stepped together on one device.

    reset() -> obs
        Observations of every environment: a tensor ``[num_envs, ...]`` or a dict of such tensors.
    step(actions [num_envs, action_dim] float32) -> (obs, reward, terminated, truncated)
        ``reward`` float32 ``[num_envs]``; ``terminated`` / ``truncated`` bool ``[num_envs]``. Environments
        that ended are reset inside ``step``, and ``obs`` is already the first observation of the new episode.
        Action clipping, if the task wants it, happens here.
    """

    num_envs: int
    action_dim: int

    def reset(self) -> Any: ...

    def step(self, actions: Tensor) -> tuple[Any, Tensor, Tensor, Tensor]: ...


@dataclass
class VecEnvPlugin:
    """Factories supplied by a project.

    build_env(task, num_envs=..., device=..., seed=..., **vec_env.env_kwargs) -> VecEnv
    build_policy(task, env, obs, **vec_env.policy_kwargs) -> nn.Module
        ``obs`` is what ``env.reset()`` returned (for input shapes). ``forward(obs)`` returns ``(mu, std, value)``: Gaussian mean and standard deviation ``[num_envs,
        action_dim]`` and the critic value ``[num_envs]`` (normalized when ``vec_env.normalize_value``).
        With ``vec_env.normalize_input`` the module owns ``obs_norm``, a
        ``lumenrl.algorithms.vec_env_ppo.RunningObsNorm`` applied inside ``forward``.
    """

    build_env: Callable[..., VecEnv]
    build_policy: Callable[..., torch.nn.Module]


def load_vec_env_plugin(module_path: str) -> VecEnvPlugin:
    if not module_path:
        raise ValueError("vec_env.plugin must name a module that exposes VEC_ENV_PLUGIN")
    module = importlib.import_module(module_path)
    plugin = getattr(module, "VEC_ENV_PLUGIN", None)
    if not isinstance(plugin, VecEnvPlugin):
        raise TypeError(f"{module_path}.VEC_ENV_PLUGIN must be a VecEnvPlugin")
    return plugin
