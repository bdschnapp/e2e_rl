"""Lane-corridor occupancy grid (pure numpy, no pygame).

Ported from e2e_rl LaneDrivingEnv._build_occupancy_grid / _sample_centerline /
_world_bounds / _grid_from_bounds / _cell_centers_world. The grid spans the world
extent [0, width_m] x [0, height_m]; cells within (lane_half + shoulder) of the
resampled centerline are free (0), else blocked (100).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GridMeta:
    origin_x: float
    origin_y: float
    res_m: float
    width: int
    height: int


def resample_centerline(xs: np.ndarray, ys: np.ndarray, ds_m: float) -> np.ndarray:
    """Resample a polyline at ~uniform arclength. Returns (N,2) in meters."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg_len = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s[-1]
    if total <= 0.0:
        return np.vstack([xs, ys]).T
    new_s = np.arange(0.0, total, ds_m)
    rx = np.interp(new_s, s, xs)
    ry = np.interp(new_s, s, ys)
    return np.stack([rx, ry], axis=1)


def build_occupancy_grid(xs: np.ndarray, ys: np.ndarray, world) -> tuple[np.ndarray, GridMeta]:
    """Build the lane-corridor occupancy grid over the world extent.

    `world` is a WorldConfig (uses width_m, height_m, grid_res_m, corridor_half_m,
    lane_sample_ds_m). Returns (grid[H,W] uint8, GridMeta).
    """
    x_min, y_min = 0.0, 0.0
    x_max, y_max = world.width_m, world.height_m
    res = world.grid_res_m
    width = int(np.ceil((x_max - x_min) / res))
    height = int(np.ceil((y_max - y_min) / res))
    meta = GridMeta(origin_x=x_min, origin_y=y_min, res_m=res, width=width, height=height)

    centerline = resample_centerline(np.asarray(xs, float), np.asarray(ys, float),
                                     world.lane_sample_ds_m)
    kdt = cKDTree(centerline)

    gx = (np.arange(width) + 0.5) * res + x_min
    gy = (np.arange(height) + 0.5) * res + y_min
    Xc, Yc = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
    dists, _ = kdt.query(pts, k=1, workers=-1)

    free_mask = dists.reshape(height, width) <= world.corridor_half_m
    grid = np.full((height, width), 100, dtype=np.uint8)
    grid[free_mask] = 0
    return grid, meta
