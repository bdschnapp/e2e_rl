"""Batched lidar (blocker #3): vectorised fixed-length march.

The scalar raycast is a double Python loop (per beam, per step) with an early
``break`` on the first blocked cell. That sequential form cannot run N envs on the
GPU. Here every (env, beam, step) sample point is materialised at once as an
(N, B, M) tensor, corridor membership is tested analytically (corridor.py), and
the first hit along the step axis is taken with ``argmax`` on the blocked mask.
"""

from __future__ import annotations

import numpy as np

from .backend import xp
from .corridor import min_dist_to_centerline


def raycast(xs, ys, cfg, pose_x, pose_y, pose_yaw, window: int = 1,
            march_step_m: float = None):
    """Normalized lidar ranges (N, B) in [0,1] (1.0 = clear to max range).

    pose_x/pose_y/pose_yaw: (N,) mount pose per env. Corridor membership replaces
    the occupancy-grid cell test; a marching point is 'blocked' if it leaves the
    world or exits the lane corridor.

    Speed: all marching math is float32 with a coarse march step (the normalized
    output is low-resolution anyway) and a tight local segment window (the nearest
    centerline segment to a marching point is at its own x-bin +/-1). This is the
    per-step hot path at large N, so it is the throughput-critical routine.
    """
    B = cfg.obs.lidar_beams
    fov = np.float32(np.deg2rad(cfg.obs.lidar_fov_deg))
    maxr = np.float32(cfg.obs.lidar_range_m)
    # Honour the configured march step (e2e_rl's get_obstacle_distances uses
    # step_m=0.05). Previously this floored to 0.15 for throughput, which silently
    # coarsened the range reading and diverged from the e2e_rl lidar definition.
    step = np.float32(march_step_m if march_step_m is not None
                      else cfg.obs.lidar_step_m)
    M = int(maxr / step)
    corridor = np.float32(cfg.world.corridor_half_m)
    W, H = np.float32(cfg.world.width_m), np.float32(cfg.world.height_m)
    N = pose_x.shape[0]
    f32 = xp.float32

    xs32 = xs.astype(f32); ys32 = ys.astype(f32)
    px = pose_x.astype(f32); py = pose_y.astype(f32); pyaw = pose_yaw.astype(f32)
    angles = xp.linspace(-fov / 2.0, fov / 2.0, B).astype(f32)   # (B,)
    theta = pyaw[:, None] + angles[None, :]                       # (N,B)
    dist_m = (xp.arange(1, M + 1, dtype=f32) * step)              # (M,)
    ct = xp.cos(theta)[:, :, None]
    st = xp.sin(theta)[:, :, None]
    dm = dist_m[None, None, :]
    wx = px[:, None, None] + dm * ct                              # (N,B,M) float32
    wy = py[:, None, None] + dm * st

    oob = (wx < 0) | (wx > W) | (wy < 0) | (wy > H)
    d = min_dist_to_centerline(xs32, ys32, wx.reshape(N, B * M),
                               wy.reshape(N, B * M), window=window).reshape(N, B, M)
    blocked = oob | (d > corridor)                               # (N,B,M)

    any_hit = xp.any(blocked, axis=2)                            # (N,B)
    first = xp.argmax(blocked, axis=2)
    hit_dist = dist_m[first]
    norm = hit_dist / maxr
    return xp.where(any_hit, norm, xp.float32(1.0)).astype(f32)
