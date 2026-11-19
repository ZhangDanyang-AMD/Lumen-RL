"""Isaac Lab v2.3.0 ROUGH_TERRAINS_CFG as one hipPhysics heightfield (benchmark terrain for g1_env.py).

Isaac generates the rough terrain as a grid of 8 m sub-terrains (10 difficulty rows x 20 columns) on a 20 m flat
border, one mesh per sub-terrain. Here the same grid is one MuJoCo heightfield (0.1 m horizontal scale, Isaac's
horizontal_scale), so the hipPhysics heightfield collider (HP_HFIELD_COLLISION=1) runs it:

  sub-terrain       proportion  shape (Isaac 2.3 mesh / height-field terrains)
  pyramid_stairs    0.2         stairs up to the centre, step height lerp(0.05, 0.23, difficulty), step width 0.3 m,
                                3 m platform, 1 m flat border
  pyramid_stairs_inv 0.2        the same stairs going down (a pit)
  boxes             0.2         0.45 m grid cells, height U(0.05, 0.2) each, 2 m platform
  random_rough      0.2         height chosen from {0.02, ..., 0.10} per point, 0.25 m flat border
  hf_pyramid_slope  0.1         slope lerp(0, 0.4, difficulty) up to the centre, 2 m platform
  hf_pyramid_slope_inv 0.1      the same slope going down

Difficulty of row r is (r + U(0, 1)) / rows, as Isaac's TerrainGenerator; the sub-terrain type is set by the column
(cumulative proportions). The robot's start tile is (level, column), the tile centre (Isaac's env_origins).
Heights are float32 metres. Grid point (i, j) sits at (i * RES, j * RES): hf[i, j] is the height at x = i * RES,
y = j * RES (x-major here; the MJCF elevation is row-major with rows along y, see g1_env.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

SIZE = 8.0  # sub-terrain side (m)
BORDER = 20.0  # flat border around the whole grid (m)
NUM_ROWS = 10  # difficulty levels
NUM_COLS = 20  # sub-terrain columns
RES = 0.1  # horizontal scale (m per grid point), Isaac horizontal_scale
N_TILE = round(SIZE / RES)  # grid points per sub-terrain side (80)
B = round(BORDER / RES)  # border grid points on each side (200)
NX = NUM_ROWS * N_TILE + 2 * B  # 1200 points along x (120 m)
NY = NUM_COLS * N_TILE + 2 * B  # 2000 points along y (200 m)
CENTER = (N_TILE - 1) * RES / 2  # local tile centre (3.95 m)

PROPORTIONS = (
    ("pyramid_stairs", 0.2),
    ("pyramid_stairs_inv", 0.2),
    ("boxes", 0.2),
    ("random_rough", 0.2),
    ("hf_pyramid_slope", 0.1),
    ("hf_pyramid_slope_inv", 0.1),
)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _pyramid_stairs(cheb: np.ndarray, step_h: float) -> np.ndarray:
    step_w, platform, border = 0.3, 1.5, CENTER - 1.0
    k_steps = int(math.floor((border - platform) / step_w))
    k = np.floor((cheb - platform) / step_w)
    z = np.where(cheb < platform, k_steps * step_h, np.where(k < k_steps, (k_steps - 1 - k) * step_h, 0.0))
    return np.where(cheb >= border, 0.0, z)


def _boxes(cheb: np.ndarray, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    grid = 0.45
    n = int(math.ceil(N_TILE * RES / grid)) + 1
    cells = rng.uniform(0.05, 0.2, (n, n))
    z = cells[(x / grid).astype(int), (y / grid).astype(int)]
    return np.where(cheb < 1.0, 0.0, z)


def _random_rough(cheb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    levels = np.arange(0.02, 0.10 + 1e-9, 0.02)
    z = levels[rng.integers(0, len(levels), cheb.shape)]
    return np.where(cheb >= CENTER - 0.25, 0.0, z)


def _slope(cheb: np.ndarray, s: float, inverted: bool) -> np.ndarray:
    z = s * np.clip(CENTER - np.maximum(cheb, 1.0), 0.0, None)
    return -z if inverted else z


def _tile(kind: str, d: float, rng: np.random.Generator) -> np.ndarray:
    i = np.arange(N_TILE) * RES
    x, y = np.meshgrid(i, i, indexing="ij")
    cheb = np.maximum(np.abs(x - CENTER), np.abs(y - CENTER))
    if kind == "pyramid_stairs":
        return _pyramid_stairs(cheb, _lerp(0.05, 0.23, d))
    if kind == "pyramid_stairs_inv":
        return -_pyramid_stairs(cheb, _lerp(0.05, 0.23, d))
    if kind == "boxes":
        return _boxes(cheb, x, y, rng)
    if kind == "random_rough":
        return _random_rough(cheb, rng)
    if kind == "hf_pyramid_slope":
        return _slope(cheb, _lerp(0.0, 0.4, d), inverted=False)
    if kind == "hf_pyramid_slope_inv":
        return _slope(cheb, _lerp(0.0, 0.4, d), inverted=True)
    raise ValueError(kind)


@dataclass
class Terrain:
    hf: np.ndarray  # (NX, NY) float32 absolute heights
    kinds: tuple  # sub-terrain name per column
    hmin: float
    zmax: float

    @property
    def half_size(self):
        """MJCF hfield half sizes (sx, sy): the grid spans [0, 2 sx] x [0, 2 sy] with its centre at (sx, sy)."""
        return (NX - 1) * RES / 2, (NY - 1) * RES / 2

    def elevation_rows(self) -> np.ndarray:
        """Normalised elevations for the MJCF attribute, row-major, shape (NY, NX). MuJoCo stores the rows in reverse
        y order (hfield_data[::-1] is this grid, checked with mj_ray), so the rows are flipped here."""
        return ((self.hf - self.hmin) / self.zmax).T[::-1].astype(np.float32)

    def origin(self, row: int, col: int) -> tuple:
        """Tile centre of (difficulty row, column) in world x, y (Isaac's env_origins)."""
        return B * RES + row * SIZE + CENTER, B * RES + col * SIZE + CENTER


def build(seed: int = 0) -> Terrain:
    rng = np.random.default_rng(seed)
    cum = np.cumsum([p for _, p in PROPORTIONS])
    cum = cum / cum[-1]
    hf = np.zeros((NX, NY), np.float32)
    kinds = []
    for col in range(NUM_COLS):
        kind = PROPORTIONS[int(np.searchsorted(cum, (col + 0.5) / NUM_COLS))][0]
        kinds.append(kind)
        for row in range(NUM_ROWS):
            d = (row + rng.uniform()) / NUM_ROWS
            x0, y0 = B + row * N_TILE, B + col * N_TILE
            hf[x0:x0 + N_TILE, y0:y0 + N_TILE] = _tile(kind, d, rng)
    hmin, hmax = float(hf.min()), float(hf.max())
    return Terrain(hf=hf, kinds=tuple(kinds), hmin=hmin, zmax=hmax - hmin)


class Ground:
    """Bilinear ground height at world (x, y) on the device (the height-scan and contact-proxy queries)."""

    def __init__(self, hf: np.ndarray, device):
        self.h = torch.as_tensor(hf, dtype=torch.float32, device=device)
        self.nx, self.ny = hf.shape

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = (x / RES).clamp(0.0, self.nx - 1.001)
        fy = (y / RES).clamp(0.0, self.ny - 1.001)
        i0, j0 = fx.floor().long(), fy.floor().long()
        tx, ty = fx - i0, fy - j0
        h = self.h
        return (h[i0, j0] * (1 - tx) * (1 - ty) + h[i0 + 1, j0] * tx * (1 - ty)
                + h[i0, j0 + 1] * (1 - tx) * ty + h[i0 + 1, j0 + 1] * tx * ty)
