"""Isaac-Repose-Cube-Shadow-Direct-v0 on hipPhysics (benchmark env; not a registered HipPhysics task).

Isaac Lab v2.3.0 source: direct/inhand_manipulation/inhand_manipulation_env.py (InHandManipulationEnv) with
direct/shadow_hand/shadow_hand_env_cfg.py (ShadowHandEnvCfg, obs_type "full", asymmetric_obs False). The hipPhysics
in-hand machinery (DirectRLEnv hooks, quaternion helpers, reorient_reward, randomize_rotation) is inherited from the
Allegro reorientation port in hipphysics_lab.tasks.reorient.reorient_direct_env, which implements the same Isaac
in-hand reward and observation for the Allegro hand. This file overrides what depends on the hand:

  * 24 hand DOFs with 20 actuated actions (the Menagerie tendon for the FF/MF/RF/LF PIP-DIP pair).
  * the Shadow model (shadow_model.py) instead of the Allegro model.
  * Isaac's reset: dof_pos = default + reset_dof_pos_noise * (delta_min + (delta_max - delta_min) * 0.5 * noise).
  * the 5 fingertips of ShadowHandEnvCfg.fingertip_body_names.

Observation dim 157 (Isaac's "full" obs), action dim 20, episode 10 s at 60 Hz (dt 1/120, decimation 2).
"""

from __future__ import annotations

import torch

from hipphysics_lab.direct import InteractiveSceneCfg, SimulationCfg
from hipphysics_lab.direct.articulation import Articulation
from hipphysics_lab.direct.math import sample_uniform, saturate
from hipphysics_lab.tasks.reorient.reorient_direct_env import AllegroHandEnv, AllegroHandEnvCfg

import shadow_model as sm


def enable_fixed_tendons() -> None:
    """Let the engine model the Shadow fixed tendons (PIP + DIP coupling and their tendon actuators).

    hipPhysics's mjmodel_loader refuses tendons unless create_engine(tendons=True) (the fixed-tendon add-on, tendon.h,
    no limit rows). The hip backend does not pass that flag, so this wraps create_engine in this process only; the
    HipPhysics tree is unchanged. Call once before the env is built."""
    from hipphysics_molmospaces import mjmodel_loader as ml

    if getattr(ml.create_engine, "_shadow_tendons", False):
        return
    original = ml.create_engine

    def create_engine(mj, *args, **kwargs):
        kwargs.setdefault("tendons", True)
        return original(mj, *args, **kwargs)

    create_engine._shadow_tendons = True
    ml.create_engine = create_engine


class ShadowHandEnvCfg(AllegroHandEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    # ShadowHandEnvCfg.sim: dt 1/120, one physics step per Isaac physics step (physics_substeps 1).
    sim = SimulationCfg(dt=1 / 120, physics_substeps=1)
    scene = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75)
    joint_names = sm.JOINT_NAMES
    fingertip_body_names = sm.FINGERTIP_BODIES
    default_joint_pos = {}  # SHADOW_HAND_CFG.init_state.joint_pos: every joint 0.0
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    in_hand_pos_offset = (0.0, 0.0, -0.04)
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


class ShadowHandEnv(AllegroHandEnv):
    _setup_scene_needs_model = True

    def _build_model(self):
        return sm.compile_model()

    def _setup_scene(self):
        cfg, mj, dev, N = self.cfg, self.mj_model, self.device, self.num_envs
        self.hand = Articulation(self, joint_names=cfg.joint_names, root_body="hand_root",
                                 body_names=list(cfg.fingertip_body_names),
                                 default_joint_pos=[0.0] * len(cfg.joint_names))
        self.object = Articulation(self, root_body="object", default_root_pos=sm.CUBE_DEFAULT_POS,
                                   default_root_quat=(0.0, 0.0, 0.0, 1.0))
        self.num_hand_dofs = self.hand.num_joints
        self.num_fingertips = len(cfg.fingertip_body_names)
        # actuator index per action (ShadowHand actuators in ACTUATORS order), and joint -> ctrl map
        import mujoco

        self._act = torch.tensor(
            [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, f"A_{row[0]}") for row in sm.ACTUATORS],
            device=dev,
        )
        self._dof_to_ctrl = torch.tensor(sm.dof_to_ctrl_matrix(), dtype=torch.float32, device=dev)
        # action range = each actuator's ctrl range (joint limits, or the tendon length range)
        act_range = torch.tensor(sm.action_ranges(), dtype=torch.float32, device=dev)
        self.action_lower_limits = act_range[:, 0]
        self.action_upper_limits = act_range[:, 1]
        lim = self.hand.data.joint_limits
        self.hand_dof_lower_limits = lim[..., 0]
        self.hand_dof_upper_limits = lim[..., 1]
        f = dict(dtype=torch.float32, device=dev)
        self.prev_targets = torch.zeros((N, len(sm.ACTUATORS)), **f)
        self.cur_targets = torch.zeros((N, len(sm.ACTUATORS)), **f)
        self.in_hand_pos = self.object.data.default_root_pose[:, 0:3].clone()
        self.in_hand_pos += torch.tensor(cfg.in_hand_pos_offset, **f)
        self.goal_rot = torch.zeros((N, 4), **f)
        self.goal_rot[:, 3] = 1.0
        self.reset_goal_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self.successes = torch.zeros(N, **f)
        self.consecutive_successes = torch.zeros(1, **f)
        self._success_flags = torch.zeros(N, dtype=torch.bool, device=dev)
        self._orientation_error_buf = torch.full((N,), torch.inf, device=dev)
        self.x_unit_tensor = torch.tensor([1, 0, 0], **f).repeat((N, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], **f).repeat((N, 1))
        self.actions = torch.zeros(N, cfg.action_space, **f)
        self._compute_intermediate_values()

    def _apply_action(self):
        lo, hi = self.action_lower_limits, self.action_upper_limits
        # Isaac scale(): [-1, 1] -> [lower, upper]
        targets = 0.5 * (self.actions + 1.0) * (hi - lo) + lo
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        targets = saturate(targets, lo, hi)
        self.cur_targets[:] = targets
        self.prev_targets[:] = targets
        self.ctrl[:, self._act] = targets.to(self.ctrl.dtype)

    def _reset_idx(self, env_ids):
        goals = self.successes[env_ids]
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (goals / (goals + 1.0)).mean()
        super(AllegroHandEnv, self)._reset_idx(env_ids)
        self._reset_target_pose(env_ids)
        n, g, dev = len(env_ids), self.rng, self.device
        obj_pose = self.object.data.default_root_pose[env_ids].clone()
        pos_noise = sample_uniform(-1.0, 1.0, (n, 3), dev, g)
        obj_pose[:, 0:3] = obj_pose[:, 0:3] + self.cfg.reset_position_noise * pos_noise
        rot_noise = sample_uniform(-1.0, 1.0, (n, 2), g.device, g)
        obj_pose[:, 3:7] = self._random_rotation(rot_noise, env_ids)
        obj_vel = torch.zeros(n, 6, device=dev)
        default_dof = self.hand.data.default_joint_pos[env_ids]
        delta_max = self.hand.data.joint_limits[env_ids][..., 1] - default_dof
        delta_min = self.hand.data.joint_limits[env_ids][..., 0] - default_dof
        dof_noise = sample_uniform(-1.0, 1.0, (n, self.num_hand_dofs), dev, g)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_noise
        dof_pos = default_dof + self.cfg.reset_dof_pos_noise * rand_delta
        dof_vel_noise = sample_uniform(-1.0, 1.0, (n, self.num_hand_dofs), dev, g)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        ctrl = dof_pos @ self._dof_to_ctrl.T
        self.prev_targets[env_ids] = ctrl
        self.cur_targets[env_ids] = ctrl
        qpos, qvel = self.default_state_rows(env_ids)
        self.object.write_root_pose(qpos, obj_pose)
        self.object.write_root_velocity(qpos, qvel, obj_vel)
        self.hand.write_joint_state(qpos, qvel, dof_pos, dof_vel)
        self._write_rows(env_ids, qpos, qvel)
        self.ctrl[env_ids.unsqueeze(-1), self._act] = ctrl.to(self.ctrl.dtype)  # set_position_index(ctrl)
        self.successes[env_ids] = 0
        self._compute_intermediate_values()
