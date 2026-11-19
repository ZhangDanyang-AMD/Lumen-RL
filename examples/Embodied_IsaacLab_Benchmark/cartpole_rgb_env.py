"""Isaac-Cartpole-RGB-Camera-Direct-v0 on one hipPhysics engine, rendered by hipPhysics or hipRT_render.

  * physics: the Cartpole model (hipphysics_lab cartpole MJCF) with the cart actuator, decimation 2, dt 1/120;
  * camera: one 100x100 RGB side camera per env at (-5, 0, 2) looking along +x, vertical fov 47.2 degrees
    (Isaac's 24 mm focal length and 20.955 mm aperture), rendered after every control step, of the state the
    policy acts on. ``renderer`` picks the renderer:
      hipphysics  hipPhysics' own RayTracer (classic look), every world in one launch;
      hiprt       hipRT_render (HIPRT) through hiprt_camera.HiprtTiledCamera: all worlds in one TLAS on Isaac's
                  20 m env grid, Newton look, 8 cameras per launch;
  * observation: the uint8 frames (N, H, W, 3); the policy does Isaac's preprocessing.

Import order: torch before hipphysics (LLVM symbol clash). run_bench_cartpole.sh sets the environment.
"""

from __future__ import annotations

import math

import numpy as np
import torch

RENDERERS = {
    "hipphysics": "hipPhysics RayTracer, classic look, all worlds in one launch (hipphysics_lab bind_cameras)",
    "hiprt": "hipRT_render (HIPRT 3.0, instinct_render hiprt_bridge), Newton look, all worlds in one TLAS, "
             "8 cameras per render_batch launch (hiprt_camera.HiprtTiledCamera)",
}

# Isaac Lab v2.3.0 cartpole camera task.
W = 100
H = 100
DECIMATION = 2
EP_LEN = 300            # ceil(5.0 s / (2 * 1/120 s)); the time-out fires at EP_LEN - 1
MAX_CART = 3.0
POLE_INIT = 0.125 * math.pi
ACTION_SCALE = 100.0
CAM_POS = (-5.0, 0.0, 2.0)
CAM_RIGHT = (0.0, -1.0, 0.0)
CAM_UP = (0.0, 0.0, 1.0)
CAM_FOVY = 47.2         # 2 * atan(20.955 / (2 * 24)) in degrees; square image, so vertical = horizontal
ENV_SPACING = 20.0      # CartpoleRGBCameraEnvCfg scene env_spacing
ACTION_CLIP = 1.0


def cartpole_camera_xml() -> str:
    from hipphysics_lab.tasks.cartpole import model as cp

    xyaxes = " ".join(str(v) for v in CAM_RIGHT + CAM_UP)
    pos = " ".join(str(v) for v in CAM_POS)
    cam = f'<camera name="side" pos="{pos}" xyaxes="{xyaxes}" fovy="{CAM_FOVY}"/>'
    return cp.mjcf().replace("</worldbody>", cam + "</worldbody>", 1)


class RgbCartpole:
    """Cartpole with one camera per world on a single hipPhysics engine. Mirrors Isaac's CartpoleCameraEnv."""

    def __init__(self, num_envs: int, device: str, seed: int, renderer: str):
        import mujoco

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        gpu = self.device.index or 0
        xml = cartpole_camera_xml()
        mj = mujoco.MjModel.from_xml_string(xml)
        if renderer == "hipphysics":
            from hipphysics_lab.renderers import HipRendererCfg
            from hipphysics_lab.sensors import CameraCfg, bind_cameras

            cams = {"side": CameraCfg(prim_path="{ENV_REGEX_NS}/side", spawn=None, width=W, height=H,
                                      renderer_cfg=HipRendererCfg(preset="classic"))}
            binding = bind_cameras(cams, xml=xml, gpu_id=gpu, num_envs=self.num_envs)
            self.eng = binding.scene.engine
            self._render = lambda: binding.render()["side"]  # noqa: E731
            self.camera = binding
        elif renderer == "hiprt":
            import hipphysics as hp
            from hipphysics_render_state.contract import PoseBuffer
            from hiprt_camera import HiprtTiledCamera

            _, self.eng = hp.create_engine(xml, self.num_envs, gpu)
            self.camera = HiprtTiledCamera(
                mj, PoseBuffer(self.eng, str(self.device)), cam_pos=CAM_POS, cam_right=CAM_RIGHT, cam_up=CAM_UP,
                fovy_deg=CAM_FOVY, width=W, height=H, env_spacing=ENV_SPACING, device=self.device)
            self._render = self.camera.render
        else:
            raise ValueError(f"unknown renderer {renderer!r}")
        self.mj = mj
        self.nq, self.nv, self.nu = mj.nq, mj.nv, mj.nu
        joint = lambda name: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)  # noqa: E731
        self.cart_q = int(mj.jnt_qposadr[joint("slider_to_cart")])
        self.pole_q = int(mj.jnt_qposadr[joint("cart_to_pole")])
        self.cart_v = int(mj.jnt_dofadr[joint("slider_to_cart")])
        self.pole_v = int(mj.jnt_dofadr[joint("cart_to_pole")])
        self.cart_act = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_actuator")
        self.qpos0 = torch.as_tensor(np.asarray(mj.qpos0), dtype=torch.float64, device=self.device)

        dev = self.device
        self.stream = torch.cuda.ExternalStream(self.eng.stream_handle, device=dev)
        if callable(getattr(self.eng, "set_keyframes", None)):
            self.eng.set_keyframes(np.asarray(mj.qpos0, np.float64)[None, :])
        v = self.eng.state_views()
        self._views = v  # keep the GpuArrays alive
        n = self.num_envs
        self.qpos = torch.as_tensor(v["qpos_d"], device=dev).reshape(n, self.nq)
        self.qvel = torch.as_tensor(v["qvel_d"], device=dev).reshape(n, self.nv)
        self.ctrl = torch.as_tensor(v["ctrl_d"], device=dev).reshape(n, self.nu)
        self.qacc_warmstart = torch.as_tensor(v["qacc_warmstart"], device=dev).reshape(n, self.nv)
        self.episode_len = torch.zeros(n, dtype=torch.long, device=dev)
        self.frames = None
        self.generator = torch.Generator(device=dev)
        self.generator.manual_seed(int(seed))
        self.reset_all()

    @property
    def action_dim(self) -> int:
        return 1

    def render(self) -> torch.Tensor:
        """Render the current state without stepping (for the render-only measurement)."""
        return self._render()

    # ---- stream ordering, as hip.HipBackend -------------------------------------------------------
    def _to_engine(self):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def _to_torch(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    # ---- resets -------------------------------------------------------------------------------------
    def _reset_rows(self, rows: torch.Tensor) -> None:
        n = int(rows.numel())
        if n == 0:
            return
        rows_np = np.ascontiguousarray(rows.cpu().numpy().astype(np.int32))
        self._to_engine()
        self.eng.reset_rows(rows_np, np.zeros(n, dtype=np.int32))
        self._to_torch()
        q = self.qpos0.expand(n, -1).clone()
        # cart at the default (0), velocity 0; pole angle uniform in +-0.125 pi (camera cfg reset)
        q[:, self.pole_q] = (torch.rand(n, device=self.device, generator=self.generator) * 2 - 1) * POLE_INIT
        self.qpos[rows] = q
        self.qvel[rows] = 0.0
        self.ctrl[rows] = 0.0
        self.qacc_warmstart[rows] = 0.0
        self.episode_len[rows] = 0

    def reset(self) -> torch.Tensor:
        return self.frames

    def reset_all(self) -> torch.Tensor:
        self._reset_rows(torch.arange(self.num_envs, device=self.device))
        self._to_engine()
        self.eng.sync()
        self._to_torch()
        self.frames = self._render()
        return self.frames

    # ---- step ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """actions: (N, 1) float32. Returns (frames, reward, terminated, time_out)."""
        force = (ACTION_SCALE * actions.reshape(self.num_envs, 1).float().clamp(-ACTION_CLIP, ACTION_CLIP)).double()
        for _ in range(DECIMATION):
            self.ctrl[:, self.cart_act] = force[:, 0]
            self._to_engine()
            self.eng.step_n(1)
            self._to_torch()
        self.episode_len += 1
        cart = self.qpos[:, self.cart_q]
        pole = self.qpos[:, self.pole_q]
        cart_vel = self.qvel[:, self.cart_v]
        pole_vel = self.qvel[:, self.pole_v]
        terminated = (cart.abs() > MAX_CART) | (pole.abs() > math.pi / 2)
        time_out = self.episode_len >= EP_LEN - 1
        reward = (
            1.0 * (~terminated).float()
            - 2.0 * terminated.float()
            - 1.0 * pole.square()
            - 0.01 * cart_vel.abs()
            - 0.005 * pole_vel.abs()
        ).to(torch.float32)
        done = terminated | time_out
        rows = done.nonzero(as_tuple=False).squeeze(-1)
        self._reset_rows(rows)
        self.frames = self._render()
        return self.frames, reward, terminated, time_out
