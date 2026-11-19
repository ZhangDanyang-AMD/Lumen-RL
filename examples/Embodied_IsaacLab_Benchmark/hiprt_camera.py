"""hipRT_render (HIPRT_Render's instinct_render) as the camera of a hipPhysics engine with many worlds.

The hipPhysics engine writes float32 geom poses for every world (PoseBuffer: geom_xpos (N, G, 3), geom_xquat
(N, G, 4) w x y z). This module turns each (world, geom) into one HIPRT instance of a tessellated mesh and places the
worlds on Isaac Lab's env grid (env_spacing apart), so all worlds share ONE TLAS, the way Isaac's cloned envs share
one stage and its tiled camera. Per frame it rebuilds the TLAS from the current poses (hiprt_bridge
update_instances) and renders one camera per world with render_batch, at most kMaxCameras (8) cameras per launch.

Shading is instinct_render's Newton look (HIPRT_LOOK_NEWTON: Lambert albedo, hemispheric ambient 0.5, one white
directional light, no gamma, 8-bit truncation), i.e. what HiprtRendererCfg draws for Isaac Lab 3.0's cartpole camera.
Only box geoms are tessellated, which is all the cartpole has.

HIPRT_PERSIST defaults to 0 here (plain grid launch). The bridge's persistent schedule waits on the host for the
previous launch's per-tile costs, which serializes the 128 small launches of 1024 cameras (MI350X, 1024 x 100x100:
87k frames/s with it, 159k without).

Needs HIPRT built (hipRT_render/instinct_render/build_hiprt.sh) and HIPRT_RENDER_ROOT pointing at the hipRT_render
checkout (default /src/hipRT_render).
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HIPRT_RENDER_ROOT = os.environ.get("HIPRT_RENDER_ROOT", "/src/hipRT_render")
MAX_CAMERAS = 8  # kMaxCameras in hiprt/trace_kernel.h
BOX = 6  # mjtGeom mjGEOM_BOX
LIGHT_DIRECTION = (-0.57735026, 0.57735026, -0.57735026)  # Newton's default directional light
BACKGROUND_U8 = (0xEE, 0xEE, 0xEE)


def _bridge():
    """instinct_render's torch extension, compiled from a copy of hiprt_bridge.cu: torch's hipify writes the .hip
    next to its source, and the checkout tracks its own hiprt_bridge.hip."""
    if HIPRT_RENDER_ROOT not in sys.path:
        sys.path.insert(0, HIPRT_RENDER_ROOT)
    from instinct_render import core

    os.environ.setdefault("HIP_PATH", "/opt/rocm/core-7.14")
    os.environ.setdefault("HIPRT_PERSIST", "0")
    src = Path(tempfile.gettempdir()) / "hiprt_bridge_src" / "hiprt_bridge.cu"
    src.parent.mkdir(exist_ok=True)
    if not src.exists() or src.read_bytes() != Path(core.BRIDGE).read_bytes():
        shutil.copy(core.BRIDGE, src)
    core.BRIDGE = src
    return core


def _box_mesh(hx: float, hy: float, hz: float):
    """24 vertices (flat normals) and 12 outward triangles of a box with half extents (hx, hy, hz)."""
    verts, normals, faces = [], [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3, np.float32)
            n[axis] = sign
            u = np.zeros(3, np.float32)
            v = np.zeros(3, np.float32)
            u[(axis + 1) % 3] = 1.0
            v[(axis + 2) % 3] = 1.0
            if sign < 0:
                u, v = v, u
            base = len(verts)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                verts.append(n + a * u + b * v)
                normals.append(n)
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    scale = np.array([hx, hy, hz], np.float32)
    return np.asarray(verts, np.float32) * scale, np.asarray(faces, np.int32), np.asarray(normals, np.float32)


def grid_origins(num_envs: int, spacing: float) -> np.ndarray:
    """Isaac Lab GridCloner layout: ceil(sqrt(N)) rows, centered on the origin."""
    rows = int(math.ceil(math.sqrt(num_envs)))
    cols = int(math.ceil(num_envs / rows))
    idx = np.arange(num_envs)
    r, c = idx // cols, idx % cols
    x = -(r - (rows - 1) / 2.0) * spacing
    y = (c - (cols - 1) / 2.0) * spacing
    return np.stack([x, y, np.zeros_like(x)], axis=1).astype(np.float32)


def _quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


class HiprtTiledCamera:
    """One world-fixed camera per world of a hipPhysics engine, rendered by hipRT_render."""

    def __init__(self, mj_model, poses, *, cam_pos, cam_right, cam_up, fovy_deg: float, width: int, height: int,
                 env_spacing: float, device: torch.device):
        core = _bridge()
        self.ext = core._ext()
        self.device = device
        self.width, self.height = int(width), int(height)
        self.poses = poses
        n, g = poses.num_worlds, poses.ngeom
        self.num_envs, self.ngeom = n, g
        types = np.asarray(mj_model.geom_type)
        if np.any(types != BOX):
            raise NotImplementedError(f"only box geoms are tessellated; geom types {sorted(set(types.tolist()))}")
        t0 = time.perf_counter()

        meshes = [_box_mesh(*mj_model.geom_size[i]) for i in range(g)]
        verts, faces, normals, starts, counts, face_offsets = [], [], [], [], [], [0]
        vstart = 0
        for _w in range(n):
            for v, f, nrm in meshes:
                verts.append(v)
                faces.append(f)
                normals.append(nrm)
                starts.append(vstart)
                counts.append(len(v))
                face_offsets.append(face_offsets[-1] + len(f))
                vstart += len(v)
        count = n * g
        verts_t = torch.from_numpy(np.concatenate(verts)).to(device)
        self.handle = self.ext.create(self._kernel(core), str(core.HIPRT_PREFIX / "include"))
        self.ext.build_geometries(
            self.handle, verts_t, torch.from_numpy(np.concatenate(faces)).to(device),
            torch.tensor(starts, dtype=torch.int64), torch.tensor(counts, dtype=torch.int64),
            torch.tensor(face_offsets, dtype=torch.int64))
        info = torch.full((count, core.SLOT_COUNT, 3), -1, dtype=torch.int64, device=device)
        params = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], device=device).expand(count, 5).contiguous()
        self.ext.set_materials(
            self.handle, torch.zeros(verts_t.shape[0], 2, device=device),
            torch.from_numpy(np.concatenate(normals)).to(device), torch.zeros(1, dtype=torch.uint8, device=device),
            info, torch.ones(count, 3, device=device), params)
        rgba = torch.as_tensor(np.asarray(mj_model.geom_rgba, np.float32)[:, :3], device=device)
        self.albedo = rgba.repeat(n, 1).contiguous()

        origins = grid_origins(n, env_spacing)
        self.origins = torch.as_tensor(origins, device=device)
        tan = math.tan(math.radians(fovy_deg) / 2.0)
        right = np.asarray(cam_right, np.float32)
        up = np.asarray(cam_up, np.float32)
        fwd = np.cross(up, right)  # MuJoCo camera: looks along -(x cross y)
        rows = np.zeros((n, 13), np.float32)
        rows[:, 0:3] = origins + np.asarray(cam_pos, np.float32)
        rows[:, 3:6], rows[:, 6:9], rows[:, 9:12], rows[:, 12] = right, up, fwd, tan
        self.camera_rows = torch.from_numpy(rows)
        self.light = (
            torch.tensor([[-c for c in LIGHT_DIRECTION]], device=device),
            torch.ones(1, 3, device=device),
            torch.tensor([0 | 2], dtype=torch.int32, device=device),
        )
        self.background = tuple((c + 0.5) / 255.0 for c in BACKGROUND_U8)
        self.image = torch.empty(n, self.height, self.width, 3, device=device)
        self.update()
        self.build_s = time.perf_counter() - t0

    @staticmethod
    def _kernel(core) -> str:
        source = "#define HIPRT_LOOK_NEWTON 1\n#define HIPRT_NEWTON_MAX_DISTANCE 1.000000e+03f\n"
        source += Path(core.KERNEL).read_text()
        out = Path(tempfile.gettempdir()) / "hiprt_trace_kernel_newton_bench.h"
        if not out.exists() or out.read_text() != source:
            out.write_text(source)
        return str(out)

    def update(self) -> None:
        """Instance transforms from the engine's current geom poses; rebuilds the TLAS."""
        pos = self.poses.geom_xpos.torch + self.origins[:, None, :]
        rot = _quat_wxyz_to_mat(self.poses.geom_xquat.torch)
        mats = torch.cat([rot, pos[..., None]], dim=-1).reshape(self.num_envs * self.ngeom, 3, 4)
        self.ext.update_instances(self.handle, mats.contiguous(), self.albedo)

    def render(self) -> torch.Tensor:
        """(N, H, W, 3) uint8, one image per world."""
        self.update()
        light_vec, light_color, light_kind = self.light
        bg = self.background
        for start in range(0, self.num_envs, MAX_CAMERAS):
            stop = min(start + MAX_CAMERAS, self.num_envs)
            view = self.image[start:stop]
            self.ext.render_batch(self.handle, view, self.camera_rows[start:stop], light_vec, light_color,
                                  light_kind, 0.5, 1.0, bg[0], bg[1], bg[2], False, 0.0, 1, 1, 0, view, view)
        return (self.image.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    def close(self) -> None:
        if self.handle:
            self.ext.destroy(self.handle)
            self.handle = 0
