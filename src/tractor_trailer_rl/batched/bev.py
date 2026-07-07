"""Vectorised bird's-eye-view (BEV) rasteriser — no pygame, GPU-parallel.

The pygame BEV in e2e_rl renders each env on the CPU (a per-env bottleneck that
doesn't parallelise). But a top-down occupancy image is just a **membership query**
at a grid of sample points: is this point inside the lane corridor? on an obstacle?
in the world? We already have those primitives analytically (corridor.py), so a
whole batch of BEV images is one big vectorised gather — the same idea as the lidar
march, but on an (S x S) grid instead of along rays.

For each env we lay an ``S x S`` grid in the VEHICLE frame (forward = image-up,
lateral = image-cols), rotate+translate it to world coordinates by the anchor pose,
and evaluate membership at all ``N * S * S`` points at once. Cost is ~S*S points/env
(1024 for 32x32) — comparable to a single lidar frame — and runs entirely on the
active backend (numpy or cupy).

Convention: 1.0 = free / drivable (inside the lane corridor, in-world, no obstacle);
0.0 = blocked. The vehicle sits at the grid centre (implicit), so it is not drawn.
"""

from __future__ import annotations

from .backend import xp
from .corridor import min_dist_to_centerline


def render_bev(xs, ys, cfg, ax, ay, ayaw, ox=None, oy=None, orad=None, ovalid=None):
    """(N, S, S) float32 BEV occupancy in [0,1], centred on the anchor pose.

    xs (P,), ys (N,P): shared centreline. ax, ay, ayaw (N,): per-env anchor pose
    (position + heading; forward maps to image-up). Optional obstacle circles
    ox/oy/orad/ovalid (N,K). S = cfg.obs.bev_size, half-extent = cfg.obs.bev_range_m.
    """
    S = int(cfg.obs.bev_size)
    R = float(cfg.obs.bev_range_m)
    N = ax.shape[0]
    f32 = xp.float32

    coords = xp.linspace(-R, R, S).astype(f32)           # (S,)
    fwd = coords[::-1]                                    # rows: top = farthest ahead
    lat = coords                                          # cols: left(-) .. right(+)
    LF, LL = xp.meshgrid(fwd, lat, indexing="ij")        # (S,S) forward, lateral

    cyaw = xp.cos(ayaw).astype(f32); syaw = xp.sin(ayaw).astype(f32)   # (N,)
    LF_ = LF[None]; LL_ = LL[None]                        # (1,S,S)
    # world = anchor + forward*heading + lateral*left_normal
    # heading = (cos, sin); left normal = (-sin, cos)
    wx = ax[:, None, None].astype(f32) + LF_ * cyaw[:, None, None] - LL_ * syaw[:, None, None]
    wy = ay[:, None, None].astype(f32) + LF_ * syaw[:, None, None] + LL_ * cyaw[:, None, None]

    fx = wx.reshape(N, S * S); fy = wy.reshape(N, S * S)  # (N, S*S)
    dist = min_dist_to_centerline(xs.astype(f32), ys.astype(f32), fx, fy)
    W = f32(cfg.world.width_m); H = f32(cfg.world.height_m)
    corridor = f32(cfg.world.corridor_half_m)
    in_world = (fx >= 0) & (fx <= W) & (fy >= 0) & (fy <= H)
    free = (dist <= corridor) & in_world                 # (N, S*S) bool

    if ox is not None and orad is not None:
        dx = fx[:, :, None] - ox[:, None, :]             # (N, S*S, K)
        dy = fy[:, :, None] - oy[:, None, :]
        d2 = dx * dx + dy * dy
        on_obs = xp.any(ovalid[:, None, :] & (d2 < (orad[:, None, :] ** 2)), axis=2)
        free = free & (~on_obs)

    return free.reshape(N, S, S).astype(f32)
