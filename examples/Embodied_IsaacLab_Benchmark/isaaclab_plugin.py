"""Lumen-RL vec-env plugin for the Isaac Lab 2.3 performance-benchmark tasks on hipPhysics.

``vec_env.plugin: isaaclab_plugin`` (this directory on PYTHONPATH; the run_bench_*.sh scripts set it up) with
``vec_env.task`` one of :data:`TASKS`. The plugin only supplies environments and networks; rollout, GAE, the PPO
update and the learning-rate schedule are ``lumenrl.trainer.vec_env_ppo_trainer.VecEnvPPOTrainer``. The
hyperparameters are in configs/<task>.yaml.

  cartpole       HipPhysics-Cartpole-Direct-v0 (hipphysics_lab), RL-Games shared [32, 32] ELU MLP
  cartpole_rgb   cartpole_rgb_env.RgbCartpole (env_kwargs.renderer: hipphysics | hiprt), RL-Games camera CNN
  shadow         shadow_env.ShadowHandEnv, RL-Games shared [512, 512, 256, 128] ELU MLP
  g1_rough       g1_env.G1RoughEnv, RSL-RL separate actor / critic [512, 256, 128] ELU

Import torch before hipphysics: both ship LLVM, and the other order aborts with
"Option 'spirv-expand-step' registered more than once".
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lumenrl.algorithms.vec_env_ppo import RunningObsNorm
from lumenrl.core.vec_env_plugin import VecEnvPlugin

SHADOW_TASK = "HipPhysics-Repose-Cube-Shadow-Direct-v0"

# Published Isaac Lab 2.3 numbers (docs.nvidia.com/isaaclab/2.3/overview/performance-benchmarks): reference_rtx4090
# (single GPU), reference_l40 (single GPU) and reference_l40_x4 (single node, 4 x L40, per-env count not stated; the
# scaling implies 4096 envs per GPU). ram_gb / vram_gb: the page's single Memory Consumption table, which does not
# name a GPU.
TASKS: dict[str, dict[str, Any]] = {
    "cartpole": {
        "task": "HipPhysics-Cartpole-Direct-v0",
        "source_task": "Isaac-Cartpole-Direct-v0",
        "hyperparameters": "Isaac Lab v2.3.0 direct/cartpole rl_games_ppo_cfg.yaml, value coefficient 0.5 "
                           "(RL-Games critic_coef 4)",
        "result_name": "cartpole_direct",
        "reference_rtx4090": {"environment_step": 1_100_000, "environment_step_and_inference": 910_000,
                              "environment_step_inference_and_train": 510_000, "ram_gb": 3.7, "vram_gb": 3.3},
        "reference_l40": {"environment_step": 620000,
                              "environment_step_and_inference": 490000,
                              "environment_step_inference_and_train": 260000},
        "reference_l40_x4": {"environment_step": 2700000,
                              "environment_step_and_inference": 2100000,
                              "environment_step_inference_and_train": 950000},
    },
    "cartpole_rgb": {
        "task": "HipPhysics-Cartpole-RGB-Camera (hipPhysics engine, one 100x100 side camera per env)",
        "source_task": "Isaac-Cartpole-RGB-Camera-Direct-v0",
        "hyperparameters": "Isaac Lab v2.3.0 rl_games_camera_ppo_cfg.yaml, value coefficient 0.5 (RL-Games 2)",
        "result_name": "cartpole_rgb_camera",
        "reference_rtx4090": {"environment_step": 50_000, "environment_step_and_inference": 45_000,
                              "environment_step_inference_and_train": 32_000, "ram_gb": 7.5, "vram_gb": 16.7},
        "reference_l40": {"environment_step": 30000,
                              "environment_step_and_inference": 28000,
                              "environment_step_inference_and_train": 21000},
        "reference_l40_x4": {"environment_step": 130000,
                              "environment_step_and_inference": 120000,
                              "environment_step_inference_and_train": 90000},
    },
    "shadow": {
        "task": SHADOW_TASK,
        "source_task": "Isaac-Repose-Cube-Shadow-Direct-v0",
        "hyperparameters": "Isaac Lab v2.3.0 direct/shadow_hand rl_games_ppo_cfg.yaml, value coefficient 0.5 "
                           "(RL-Games critic_coef 4); no time-out bootstrap (the env does not expose the terminal "
                           "observation)",
        "result_name": "shadow_repose",
        "reference_rtx4090": {"environment_step": 200_000, "environment_step_and_inference": 190_000,
                              "environment_step_inference_and_train": 170_000, "ram_gb": 6.7, "vram_gb": 6.4},
        "reference_l40": {"environment_step": 170000,
                              "environment_step_and_inference": 140000,
                              "environment_step_inference_and_train": 120000},
        "reference_l40_x4": {"environment_step": 440000,
                              "environment_step_and_inference": 420000,
                              "environment_step_inference_and_train": 390000},
    },
    "g1_rough": {
        "task": "Isaac-Velocity-Rough-G1-v0 (g1_env.G1RoughEnv)",
        "source_task": "Isaac-Velocity-Rough-G1-v0",
        "hyperparameters": "Isaac Lab v2.3.0 RSL-RL G1RoughPPORunnerCfg, value coefficient 0.5 (RSL-RL 1.0)",
        "result_name": "g1_rough",
        "reference_rtx4090": {"environment_step": 94_000, "environment_step_and_inference": 88_000,
                              "environment_step_inference_and_train": 82_000, "ram_gb": 6.5, "vram_gb": 6.1},
        "reference_l40": {"environment_step": 72000,
                              "environment_step_and_inference": 64000,
                              "environment_step_inference_and_train": 62000},
        "reference_l40_x4": {"environment_step": 290000,
                              "environment_step_and_inference": 270000,
                              "environment_step_inference_and_train": 250000},
    },
}


# ---- environments ---------------------------------------------------------------------------------
class LabEnv:
    """hipphysics_lab DirectRLEnv -> VecEnv. ``obs_key`` selects one observation group (None keeps the dict);
    ``action_clip`` > 0 clamps actions before the env applies its action scale."""

    def __init__(self, env, obs_key: str | None, action_clip: float):
        self.env = env
        self.obs_key = obs_key
        self.action_clip = float(action_clip)
        self.num_envs = int(env.num_envs)
        self.action_dim = int(env.cfg.action_space)
        self._seed = int(env.cfg.seed)

    def _obs(self, obs: dict):
        if self.obs_key is None:
            return obs
        return obs[self.obs_key].to(dtype=torch.float32)

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed)
        return self._obs(obs)

    def step(self, actions: torch.Tensor):
        if self.action_clip > 0:
            actions = actions.clamp(-self.action_clip, self.action_clip)
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        return self._obs(obs), reward.to(torch.float32), terminated, truncated

    def close(self) -> None:
        self.env.close()


def _lab_cfg(cfg, num_envs: int, device: str, seed: int):
    cfg.sim.device = device
    cfg.sim.backend = "hip"
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return cfg


def build_env(task: str, *, num_envs: int, device: str, seed: int, **kwargs):
    if task == "cartpole":
        import hipphysics_lab

        name = TASKS[task]["task"]
        cfg = _lab_cfg(hipphysics_lab.load_cfg_from_registry(name), num_envs, device, seed)
        return LabEnv(hipphysics_lab.make(name, cfg=cfg), "policy", kwargs.get("action_clip", 1.0))
    if task == "shadow":
        import hipphysics_lab
        import shadow_env
        from hipphysics_lab import registry

        shadow_env.enable_fixed_tendons()
        registry.register(SHADOW_TASK, "shadow_env:ShadowHandEnv",
                          {"env_cfg_entry_point": "shadow_env:ShadowHandEnvCfg",
                           "source_id": "Isaac-Repose-Cube-Shadow-Direct-v0"})
        cfg = _lab_cfg(registry.load_cfg_from_registry(SHADOW_TASK), num_envs, device, seed)
        return LabEnv(hipphysics_lab.make(SHADOW_TASK, cfg=cfg), "policy", kwargs.get("action_clip", 1.0))
    if task == "g1_rough":
        import g1_env

        cfg = _lab_cfg(g1_env.G1RoughEnvCfg(), num_envs, device, seed)
        return LabEnv(g1_env.G1RoughEnv(cfg), None, kwargs.get("action_clip", 0.0))
    if task == "cartpole_rgb":
        from cartpole_rgb_env import RgbCartpole

        return RgbCartpole(num_envs, device, seed, kwargs.get("renderer", "hipphysics"))
    raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")


# ---- policies -------------------------------------------------------------------------------------
def _mlp(in_dim: int, units, out_dim: int | None = None) -> nn.Sequential:
    layers, last = [], in_dim
    for width in units:
        layers += [nn.Linear(last, width), nn.ELU()]
        last = width
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SharedMlpActorCritic(nn.Module):
    """RL-Games ``continuous_a2c_logstd`` with ``separate: False``: input normalization, shared ELU trunk, learned
    log-std (``fixed_sigma``, const init 0)."""

    def __init__(self, obs_dim: int, act_dim: int, units):
        super().__init__()
        self.obs_norm = RunningObsNorm(obs_dim)
        self.trunk = _mlp(obs_dim, units)
        last = units[-1]
        self.mu = nn.Linear(last, act_dim)
        self.value = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        hidden = self.trunk(self.obs_norm(obs))
        mu = self.mu(hidden)
        return mu, self.log_std.exp().clamp(min=1e-6).expand_as(mu), self.value(hidden).squeeze(-1)


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float, image / 255 minus the per-image channel mean (Isaac's rule)."""
    x = frames.float().div_(255.0)
    x = x - x.mean(dim=(1, 2), keepdim=True)
    return x.permute(0, 3, 1, 2).contiguous()


class CnnActorCritic(nn.Module):
    """Isaac's rl_games_camera_ppo_cfg.yaml network: conv 32x8s4, 64x4s2, 64x3s1 (ReLU), MLP [512] (ELU), shared
    trunk, learned log-std. Input: uint8 frames (N, 100, 100, 3)."""

    def __init__(self, act_dim: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(nn.Linear(64 * 9 * 9, 512), nn.ELU())  # 100 -> 24 -> 11 -> 9
        self.mu = nn.Linear(512, act_dim)
        self.value = nn.Linear(512, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, frames: torch.Tensor):
        h = self.mlp(self.conv(preprocess_frames(frames)))
        mu = self.mu(h)
        return mu, self.log_std.exp().clamp(min=1e-6).expand_as(mu), self.value(h).squeeze(-1)


class SeparateActorCritic(nn.Module):
    """RSL-RL ActorCritic: separate MLPs on the ``policy`` and ``critic`` observation groups, learned std vector."""

    def __init__(self, obs_dim: int, critic_dim: int, act_dim: int, units, init_std: float):
        super().__init__()
        self.actor = _mlp(obs_dim, units, act_dim)
        self.critic = _mlp(critic_dim, units, 1)
        self.std = nn.Parameter(init_std * torch.ones(act_dim))

    def forward(self, obs: dict):
        mu = self.actor(obs["policy"])
        return mu, self.std.expand_as(mu), self.critic(obs["critic"]).squeeze(-1)


def build_policy(task: str, env, obs, **kwargs) -> nn.Module:
    if task == "cartpole":
        return SharedMlpActorCritic(obs.shape[-1], env.action_dim, kwargs.get("units", (32, 32)))
    if task == "shadow":
        return SharedMlpActorCritic(obs.shape[-1], env.action_dim, kwargs.get("units", (512, 512, 256, 128)))
    if task == "g1_rough":
        return SeparateActorCritic(obs["policy"].shape[-1], obs["critic"].shape[-1], env.action_dim,
                                   kwargs.get("units", (512, 256, 128)), kwargs.get("init_std", 1.0))
    if task == "cartpole_rgb":
        return CnnActorCritic(env.action_dim)
    raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")


VEC_ENV_PLUGIN = VecEnvPlugin(build_env=build_env, build_policy=build_policy)
