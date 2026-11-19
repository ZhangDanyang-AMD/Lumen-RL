"""Isaac-Velocity-Rough-G1-v0 on hipPhysics (benchmark env; not a registered HipPhysics task).

Isaac Lab v2.3.0 source: tasks/manager_based/locomotion/velocity/velocity_env_cfg.py (the MDP terms) and
config/g1/rough_env_cfg.py (G1RoughEnvCfg), robot G1_MINIMAL_CFG (isaaclab_assets/robots/unitree.py). The robot is
g1_model.py's G1 (Menagerie unitree_g1, boxes for the collision meshes); the terrain is g1_terrain.py's Isaac
ROUGH_TERRAINS_CFG on one heightfield. Settings and terms:

  * sim dt 0.005 s, decimation 4 (policy 50 Hz), episode 20 s; actions = JointPositionAction, scale 0.5, default
    offset, 29 joints; the actuators are the implicit PD of g1_model.py.
  * policy observations (286): base lin vel (noise U(-0.5, 0.5)), base ang vel (U(-0.2, 0.2)), projected gravity
    (U(-0.05, 0.05)), velocity command (3), joint pos relative (U(-0.01, 0.01)), joint vel relative (U(-1.5, 1.5)),
    last action, height scan (187 points, offset 0.5, clipped to [-1, 1], noise U(-0.1, 0.1)).
    The critic sees the same 286 values without noise.
  * commands: lin vel x (0, 1), y (-0.5, 0.5), ang vel z (-1, 1), resampled every 10 s; 20 % standing envs; heading
    control (ang vel z = clip(0.5 * heading error, -1, 1)).
  * events: reset base pose xy U(-0.5, 0.5), yaw U(-pi, pi), velocity U(-0.5, 0.5) (6 values); reset joints
    default x U(0.5, 1.5); push every 10-15 s with xy velocity U(-0.5, 0.5).
  * terminations: time out (20 s) and base contact (torso_link).
  * rewards (weights of the rough G1 config): track lin vel xy exp (1.0, std 0.5), track ang vel z exp (1.0, std 0.5),
    lin vel z l2 (-0.2), ang vel xy l2 (-0.05), dof torques l2 (-1.5e-7), dof acc l2 (-1.25e-7), action rate l2
    (-0.005), feet air time (0.25, threshold 0.4, only with command > 0.1), feet slide (-0.1), undesired contacts
    (-1.0), flat orientation l2 (-1.0), joint pos limits (-1.0, soft limits at 0.9), joint deviation hip / arms /
    torso (-0.1 each).
  * curriculum: terrain level up when the robot walked more than half a tile, down when it walked less than
    half the commanded distance; start levels 0..5 (Isaac's max_init_terrain_level).

Approximations (this direct-workflow env does not bind the lab's ContactSensor; the numbers are a throughput
benchmark, not an Isaac reproduction):
  * the standing height is 0.79 m (Menagerie's stand keyframe) instead of G1_CFG's 0.74 m: with this model's
    collision boxes and the default pose, 0.74 m starts the soles 4.6 cm into the floor;
  * contacts are geometric: a collision geom point (box corner or sphere axis point) within `contact_threshold`
    of the terrain counts as contact, instead of Isaac's contact-force threshold (1 N);
  * dof torques are the PD law's torque clipped to the effort limit (the implicit drive's force);
  * the height scan is bilinear on the same heightfield the physics uses, at the 187 grid points;
  * the robot's collision shapes are oriented boxes of its meshes (g1_model.py), not the meshes;
  * no per-env friction / mass randomization (Isaac's physics_material and add_base_mass events).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from hipphysics_lab.direct import DirectRLEnv, DirectRLEnvCfg, InteractiveSceneCfg, SimulationCfg
from hipphysics_lab.direct.articulation import Articulation
from hipphysics_lab.direct.math import euler_xyz_from_quat, sample_uniform, wrap_to_pi

import g1_model
import g1_terrain as gt

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO = "torso_link"
BASE = "pelvis"
ROOT_HEIGHT = 0.79  # standing height of this model (Menagerie keyframe "stand"); G1_CFG 0.74 leaves the soles 4.6 cm in the floor
SCAN_X = tuple(round(-0.8 + 0.1 * i, 6) for i in range(17))  # GridPatternCfg size (1.6, 1.0), resolution 0.1
SCAN_Y = tuple(round(-0.5 + 0.1 * i, 6) for i in range(11))
SCAN_SENSOR_HEIGHT = 20.0  # RayCasterCfg offset z
ENV_LEVEL_MAX_INIT = 5  # Isaac's max_init_terrain_level for the rough G1 task


def build_model_xml(terrain: gt.Terrain) -> str:
    """g1_boxes.xml plus the terrain heightfield (asset + worldbody geom)."""
    root = ET.parse(g1_model.OUT).getroot()
    sx, sy = terrain.half_size
    asset = ET.Element("asset")
    root.insert(0, asset)
    elev = " ".join(f"{v:.4f}" for v in terrain.elevation_rows().ravel())
    ET.SubElement(asset, "hfield", name="terrain", nrow=str(gt.NY), ncol=str(gt.NX),
                  size=f"{sx:.6f} {sy:.6f} {terrain.zmax:.6f} 0.5", elevation=elev)
    ET.SubElement(root.find("worldbody"), "geom", name="terrain", type="hfield", hfield="terrain",
                  pos=f"{sx:.6f} {sy:.6f} {terrain.hmin:.6f}", friction="1.0 0.005 0.0001")
    return ET.tostring(root, encoding="unicode")


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by unit quaternions q (wxyz); broadcast over leading dims."""
    v = v.expand(*q.shape[:-1], 3)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


class G1RoughEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_space = 29
    observation_space = 286  # 3 + 3 + 3 + 3 + 29 + 29 + 29 + 187
    state_space = 0
    # G1RoughEnvCfg.sim: dt 0.005, one physics step per Isaac physics step.
    sim = SimulationCfg(dt=0.005, physics_substeps=1)
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)
    terrain_seed = 0
    action_scale = 0.5
    # commands (UniformVelocityCommandCfg, rough G1)
    resample_s = 10.0
    rel_standing_envs = 0.2
    lin_vel_x = (0.0, 1.0)
    lin_vel_y = (-0.5, 0.5)
    ang_vel_z = (-1.0, 1.0)
    heading_control_stiffness = 0.5
    # events
    reset_base_xy = 0.5
    reset_base_vel = 0.5
    reset_joint_scale = (0.5, 1.5)
    push_interval_s = (10.0, 15.0)
    push_vel = 0.5
    # rewards (weights; std of the exp terms)
    track_std = 0.5
    rew_lin_vel_z = -0.2
    rew_ang_vel_xy = -0.05
    rew_dof_torques = -1.5e-7
    rew_dof_acc = -1.25e-7
    rew_action_rate = -0.005
    rew_feet_air_time = 0.25
    feet_air_threshold = 0.4
    rew_feet_slide = -0.1
    rew_undesired = -1.0
    rew_flat_orientation = -1.0
    rew_joint_limits = -1.0
    joint_limit_factor = 0.9
    rew_joint_deviation = -0.1
    # contact proxy: a collision point this close to the terrain (m) counts as contact
    contact_threshold = 0.02
    # observation noise (Isaac's Unoise ranges)
    noise_lin_vel = 0.5
    noise_ang_vel = 0.2
    noise_gravity = 0.05
    noise_joint_pos = 0.01
    noise_joint_vel = 1.5
    noise_height_scan = 0.1


class G1RoughEnv(DirectRLEnv):
    _setup_scene_needs_model = True

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.terrain = gt.build(seed=cfg.terrain_seed)
        super().__init__(cfg, render_mode, **kwargs)

    def _build_model(self):
        return mujoco.MjModel.from_xml_string(build_model_xml(self.terrain))

    def _setup_scene(self):
        cfg, mj, dev, N = self.cfg, self.mj_model, self.device, self.num_envs
        f = dict(dtype=torch.float32, device=dev)
        joints = g1_model.ACTION_JOINTS
        self.robot = Articulation(
            self, joint_names=joints, root_body=BASE, body_names=list(FEET),
            default_joint_pos=[g1_model.DEFAULT_POS.get(j, 0.0) for j in joints],
            default_root_pos=(0.0, 0.0, ROOT_HEIGHT), default_root_quat=(0.0, 0.0, 0.0, 1.0),
            soft_joint_pos_limit_factor=cfg.joint_limit_factor,
        )
        self.num_joints = len(joints)
        self._act = torch.tensor(
            [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, f"A_{j[:-len('_joint')]}") for j in joints],
            device=dev)
        self._kp = torch.tensor([g1_model.JOINT_PARAMS[j][0] for j in joints], **f)
        self._kv = torch.tensor([g1_model.JOINT_PARAMS[j][1] for j in joints], **f)
        self._effort = torch.tensor([g1_model.JOINT_PARAMS[j][2] for j in joints], **f)
        self.default_pos = torch.tensor([g1_model.DEFAULT_POS.get(j, 0.0) for j in joints], **f).expand(N, -1).clone()
        self._hip_idx = torch.tensor([i for i, j in enumerate(joints) if "hip_roll" in j or "hip_yaw" in j], device=dev)
        self._arm_idx = torch.tensor([i for i, j in enumerate(joints) if "shoulder" in j or "elbow" in j
                                      or "wrist" in j], device=dev)
        self._torso_idx = torch.tensor([i for i, j in enumerate(joints) if "waist" in j], device=dev)
        self.ground = gt.Ground(self.terrain.hf, dev)
        self._build_contact_points(mj)

        # origins of (difficulty row, column) tiles; env columns fixed, levels move with the curriculum
        self._origin_table = torch.tensor(
            [[self.terrain.origin(r, c) for c in range(gt.NUM_COLS)] for r in range(gt.NUM_ROWS)], **f)
        self.env_cols = torch.randint(0, gt.NUM_COLS, (N,), device=dev)
        self.env_levels = torch.randint(0, ENV_LEVEL_MAX_INIT + 1, (N,), device=dev)
        self.origin_xy = torch.zeros(N, 2, **f)
        self.origin_z = torch.zeros(N, **f)
        self._refresh_origins(torch.arange(N, device=dev))

        self.actions = torch.zeros(N, cfg.action_space, **f)
        self.prev_actions = torch.zeros_like(self.actions)
        self.targets = self.default_pos.clone()
        self.prev_joint_vel = torch.zeros(N, self.num_joints, **f)
        self.commands = torch.zeros(N, 3, **f)
        self.heading_target = torch.zeros(N, **f)
        self.cmd_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.push_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.air_time = torch.zeros(N, 2, **f)
        self.last_air_time = torch.zeros(N, 2, **f)
        self.foot_contact = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.torso_contact = torch.zeros(N, dtype=torch.bool, device=dev)
        self.undesired_count = torch.zeros(N, **f)
        self.noise_amp = self._noise_amplitudes()
        self._resample_steps = int(round(cfg.resample_s / self.step_dt))
        lo, hi = cfg.push_interval_s
        self._push_steps = (int(round(lo / self.step_dt)), int(round(hi / self.step_dt)))
        self.torque = torch.zeros(N, self.num_joints, **f)
        self._standing = torch.zeros(N, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ geometry
    def _build_contact_points(self, mj):
        """Body-frame points of every robot collision geom: box corners and sphere axis points."""
        feet = {mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n): i for i, n in enumerate(FEET)}
        torso = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, TORSO)
        # undesired contacts exclude every ankle body (Isaac: body names "^(?!.*ankle).*")
        ankles = {b for b in range(mj.nbody) if "ankle" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or "")}
        pts, bodies = [], []
        for g in range(mj.ngeom):
            gtype = int(mj.geom_type[g])
            if gtype not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE):
                continue
            b = int(mj.geom_bodyid[g])
            if b == 0:
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, mj.geom_quat[g])
            rot = rot.reshape(3, 3)
            size = mj.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size[:3]
            else:
                local = np.vstack([np.eye(3) * size[0], -np.eye(3) * size[0]])
            pts.append(local @ rot.T + mj.geom_pos[g])
            bodies.extend([b] * len(local))
        self._pt_body = torch.tensor(bodies, dtype=torch.long, device=self.device)
        self._pt_local = torch.tensor(np.vstack(pts), dtype=torch.float32, device=self.device)
        body_of = np.asarray(bodies)
        self._foot_masks = [torch.tensor(body_of == b, device=self.device) for b in feet]
        self._torso_mask = torch.tensor(body_of == torso, device=self.device)
        # one-hot body membership for the undesired-contact count (bodies other than the feet)
        ub = sorted({b for b in bodies if b not in feet and b not in ankles})
        self._undesired_bodies = ub
        onehot = np.zeros((len(bodies), len(ub)), np.float32)
        for k, b in enumerate(bodies):
            if b in ub:
                onehot[k, ub.index(b)] = 1.0
        self._undesired_onehot = torch.tensor(onehot, device=self.device)

    def _noise_amplitudes(self):
        cfg, dev = self.cfg, self.device
        n = cfg.observation_space
        amp = torch.zeros(n, device=dev)
        amp[0:3] = cfg.noise_lin_vel
        amp[3:6] = cfg.noise_ang_vel
        amp[6:9] = cfg.noise_gravity
        amp[12:12 + self.num_joints] = cfg.noise_joint_pos
        amp[12 + self.num_joints:12 + 2 * self.num_joints] = cfg.noise_joint_vel
        amp[-gt_scan_count():] = cfg.noise_height_scan
        return amp

    def _refresh_origins(self, env_ids):
        rows = self.env_levels[env_ids]
        cols = self.env_cols[env_ids]
        xy = self._origin_table[rows, cols]
        self.origin_xy[env_ids] = xy
        self.origin_z[env_ids] = self.ground(xy[:, 0], xy[:, 1])

    def _height_scan(self, d):
        base_xy, base_z = d.root_pos_w[:, :2], d.root_pos_w[:, 2]
        _, _, yaw = euler_xyz_from_quat(d.root_quat_w)
        gx = torch.tensor(SCAN_X, device=self.device)
        gy = torch.tensor(SCAN_Y, device=self.device)
        gx, gy = torch.meshgrid(gx, gy, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        wx = base_xy[:, :1] + c * gx - s * gy
        wy = base_xy[:, 1:2] + s * gx + c * gy
        ground = self.ground(wx, wy)
        return (base_z[:, None] + SCAN_SENSOR_HEIGHT - ground - 0.5)

    def _contacts(self):
        """Geometric contact proxy at the end of a policy step: foot / torso / undesired-body contacts."""
        k = self.kinematics_state()
        xp = k["xpos"][:, self._pt_body].to(torch.float32)
        q = k["xquat"][:, self._pt_body].to(torch.float32)
        wp = xp + _quat_rotate_wxyz(q, self._pt_local)
        clearance = wp[..., 2] - self.ground(wp[..., 0], wp[..., 1])
        touching = (clearance < self.cfg.contact_threshold).float()  # (N, K)
        self.foot_contact = torch.stack([(touching > 0)[:, m].any(-1) for m in self._foot_masks], -1)
        self.torso_contact = (touching * self._torso_mask.float()).sum(-1) > 0
        body_touch = (touching @ self._undesired_onehot) > 0
        self.undesired_count = body_touch.float().sum(-1)

    # ------------------------------------------------------------------ MDP
    def _pre_physics_step(self, actions):
        self.prev_actions = self.actions
        self.actions = actions.clone()
        self.targets = self.default_pos + self.cfg.action_scale * self.actions
        self._update_commands()
        self._push()

    def _apply_action(self):
        self.ctrl[:, self._act] = self.targets.to(self.ctrl.dtype)

    def _update_commands(self):
        self.cmd_timer -= 1
        due = (self.cmd_timer <= 0).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() > 0:
            self._resample(due)
        d = self.robot.data
        heading_err = wrap_to_pi(self.heading_target - euler_xyz_from_quat(d.root_quat_w)[2])
        self.commands[:, 2] = torch.clamp(self.cfg.heading_control_stiffness * heading_err, -1.0, 1.0)
        self.commands[:, 2][self._standing] = 0.0

    def _resample(self, env_ids):
        cfg, n, dev, g = self.cfg, len(env_ids), self.device, self.rng
        cmd = torch.stack([
            sample_uniform(*cfg.lin_vel_x, (n,), dev, g),
            sample_uniform(*cfg.lin_vel_y, (n,), dev, g),
            sample_uniform(*cfg.ang_vel_z, (n,), dev, g),
        ], dim=-1)
        standing = torch.rand(n, device=dev) < cfg.rel_standing_envs
        cmd[standing] = 0.0
        self.commands[env_ids] = cmd
        self.heading_target[env_ids] = sample_uniform(-math.pi, math.pi, (n,), dev, g)
        self._standing[env_ids] = standing
        self.cmd_timer[env_ids] = self._resample_steps

    def _push(self):
        self.push_timer -= 1
        due = (self.push_timer <= 0).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() == 0:
            return
        n = due.numel()
        v = sample_uniform(-self.cfg.push_vel, self.cfg.push_vel, (n, 2), self.device, self.rng)
        a = self.robot._free_dadr
        qv = self.backend.qvel
        qv[due, a:a + 2] += v.to(qv.dtype)
        self.push_timer[due] = torch.randint(self._push_steps[0], self._push_steps[1] + 1, (n,),
                                             device=self.device)

    def _get_observations(self):
        d = self.robot.data
        q = d.joint_pos - self.default_pos
        qd = d.joint_vel
        scan = self._height_scan(d)  # raw (Isaac: noise, then clip)
        clean_scan = torch.clamp(scan, -1.0, 1.0)
        clean = torch.cat([
            d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, self.commands, q, qd, self.actions,
            clean_scan,
        ], dim=-1)
        noise = (2.0 * torch.rand_like(clean) - 1.0) * self.noise_amp
        noisy = clean + noise
        n_scan = gt_scan_count()
        noisy[:, -n_scan:] = torch.clamp(scan + noise[:, -n_scan:], -1.0, 1.0)
        return {"policy": noisy, "critic": clean}

    def _get_dones(self):
        self._contacts()
        time_out = self.episode_length_buf >= self.max_episode_length
        return self.torso_contact.clone(), time_out

    def _get_rewards(self):
        cfg, d = self.cfg, self.robot.data
        cmd, dt = self.commands, self.step_dt
        lin_b, ang_b, grav = d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b
        std2 = cfg.track_std ** 2
        track_lin = torch.exp(-((cmd[:, :2] - lin_b[:, :2]) ** 2).sum(-1) / std2)
        track_ang = torch.exp(-(cmd[:, 2] - ang_b[:, 2]) ** 2 / std2)

        # torque of the implicit PD drive, clipped to the effort limit (the motor's force range)
        q, qd = d.joint_pos, d.joint_vel
        tau = torch.clamp(self._kp * (self.targets - q) - self._kv * qd, -self._effort, self._effort)
        self.torque = tau
        dof_acc = (qd - self.prev_joint_vel) / dt
        self.prev_joint_vel = qd.clone()

        # feet air time (first contact after a swing), gated by the command
        contact = self.foot_contact
        first = contact & (self.air_time > 0)
        self.last_air_time = torch.where(first, self.air_time, self.last_air_time)
        air_reward = ((self.last_air_time - cfg.feet_air_threshold) * first.float()).sum(-1)
        air_reward = air_reward * (torch.norm(cmd[:, :2], dim=-1) > 0.1).float()
        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + dt)

        foot_vel_xy = d.body_lin_vel_w[:, :, :2]
        feet_slide = (torch.norm(foot_vel_xy, dim=-1) * contact.float()).sum(-1)

        soft = d.soft_joint_pos_limits
        limits = ((q - soft[..., 0]).clamp(max=0.0) + (q - soft[..., 1]).clamp(min=0.0)).abs().sum(-1)
        dev = (q - self.default_pos).abs()
        dev_hip = dev[:, self._hip_idx].sum(-1)
        dev_arm = dev[:, self._arm_idx].sum(-1)
        dev_torso = dev[:, self._torso_idx].sum(-1)

        terms = {
            "track_lin_vel_xy": track_lin,
            "track_ang_vel_z": track_ang,
            "lin_vel_z_l2": cfg.rew_lin_vel_z * lin_b[:, 2] ** 2,
            "ang_vel_xy_l2": cfg.rew_ang_vel_xy * (ang_b[:, :2] ** 2).sum(-1),
            "dof_torques_l2": cfg.rew_dof_torques * (tau ** 2).sum(-1),
            "dof_acc_l2": cfg.rew_dof_acc * (dof_acc ** 2).sum(-1),
            "action_rate_l2": cfg.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1),
            "feet_air_time": cfg.rew_feet_air_time * air_reward,
            "feet_slide": cfg.rew_feet_slide * feet_slide,
            "undesired_contacts": cfg.rew_undesired * self.undesired_count,
            "flat_orientation_l2": cfg.rew_flat_orientation * (grav[:, :2] ** 2).sum(-1),
            "joint_pos_limits": cfg.rew_joint_limits * limits,
            "joint_deviation": cfg.rew_joint_deviation * (dev_hip + dev_arm + dev_torso),
        }
        self.reward_terms = terms  # per-env terms of the last step (diagnostics)
        return sum(terms.values())

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._update_curriculum(env_ids)
        cfg, n, dev, g = self.cfg, len(env_ids), self.device, self.rng
        d = self.robot.data
        xy = self.origin_xy[env_ids] + sample_uniform(-cfg.reset_base_xy, cfg.reset_base_xy, (n, 2), dev, g)
        z = self.ground(xy[:, 0], xy[:, 1]) + ROOT_HEIGHT
        yaw = sample_uniform(-math.pi, math.pi, (n,), dev, g)
        root_pose = torch.cat([xy, z[:, None], _yaw_quat_xyzw(yaw)], dim=-1)
        root_vel = sample_uniform(-cfg.reset_base_vel, cfg.reset_base_vel, (n, 6), dev, g)
        scale = sample_uniform(*cfg.reset_joint_scale, (n, self.num_joints), dev, g)
        joint_pos = self.default_pos[env_ids] * scale
        joint_vel = torch.zeros(n, self.num_joints, device=dev)
        qpos, qvel = self.default_state_rows(env_ids)
        self.robot.write_root_pose(qpos, root_pose)
        self.robot.write_root_velocity(qpos, qvel, root_vel)
        self.robot.write_joint_state(qpos, qvel, joint_pos, joint_vel)
        self._write_rows(env_ids, qpos, qvel)
        self.targets[env_ids] = self.default_pos[env_ids]
        self.ctrl[env_ids.unsqueeze(-1), self._act] = self.default_pos[env_ids].to(self.ctrl.dtype)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
        self.cmd_timer[env_ids] = 0
        self._resample(env_ids)
        self.push_timer[env_ids] = torch.randint(self._push_steps[0], self._push_steps[1] + 1, (n,),
                                                 device=dev)
        self._refresh_origins(env_ids)

    def _update_curriculum(self, env_ids):
        cfg = self.cfg
        d = self.robot.data
        walked = torch.norm(d.root_pos_w[env_ids, :2] - self.origin_xy[env_ids], dim=-1)
        cmd_dist = torch.norm(self.commands[env_ids, :2], dim=-1) * cfg.episode_length_s * 0.5
        up = walked > gt.SIZE / 2
        down = (walked < cmd_dist) & ~up
        lev = self.env_levels[env_ids] + up.long() - down.long()
        full = lev >= gt.NUM_ROWS
        lev = torch.where(full, torch.randint(0, gt.NUM_ROWS, lev.shape, device=self.device),
                          lev.clamp(min=0))
        self.env_levels[env_ids] = lev


def gt_scan_count() -> int:
    return len(SCAN_X) * len(SCAN_Y)
