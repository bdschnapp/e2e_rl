"""Lidar raycasting over an occupancy grid (pure numpy).

Ported from e2e_rl ObstacleAvoidance.get_obstacle_distances. The ONLY change:
the grid is indexed via GridMeta (origin + res_m) instead of the global
METERS_PER_PIXEL. In training grid_res == meters_per_pixel, so this is bit-parity;
it also lets the deploy grid use a different resolution cleanly.

Parity-critical: keep the int() truncation (not round) on pixel indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .occupancy import GridMeta


@dataclass(frozen=True)
class Pose:
    x: float    # meters (world)
    y: float    # meters (world)
    yaw: float  # radians


def raycast(
    occ_grid: np.ndarray,
    meta: GridMeta,
    pose: Pose,
    *,
    num_sensors: int = 24,
    fov_deg: float = 120.0,
    max_range_m: float = 20.0,
    step_m: float = 0.05,
) -> np.ndarray:
    """Normalized lidar distances [0,1] per beam (1.0 = clear to max range)."""
    H, W = occ_grid.shape
    res = meta.res_m
    x0 = int((pose.x - meta.origin_x) / res)
    y0 = int((pose.y - meta.origin_y) / res)
    max_steps = int(max_range_m / step_m)

    fov = np.deg2rad(fov_deg)
    angles = np.linspace(-fov / 2.0, fov / 2.0, num_sensors, dtype=np.float32)
    distances = np.ones(num_sensors, dtype=np.float32)

    for i, a in enumerate(angles):
        theta = pose.yaw + a
        dx = np.cos(theta)
        dy = np.sin(theta)
        for step in range(1, max_steps + 1):
            dist_m = step * step_m
            px = int(x0 + (dist_m / res) * dx)
            py = int(y0 + (dist_m / res) * dy)
            if px < 0 or px >= W or py < 0 or py >= H:
                distances[i] = dist_m / max_range_m
                break
            if occ_grid[py, px] > 0:
                distances[i] = dist_m / max_range_m
                break
    return distances
