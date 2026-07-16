"""Signed-distance-field BEV rasteriser (Phase-2B vision rescue — NEW, additive).

The frozen `bev.py::render_bev` returns a BINARY occupancy image (1=free, 0=blocked).
A binary mask is flat almost everywhere, so a small CNN has to synthesise the
task-relevant quantity (distance-to-corridor-edge) from sparse edge gradients. This
module instead renders a SIGNED DISTANCE field: every pixel encodes how far it is
inside/outside the drivable corridor — a dense, smooth, task-aligned signal with
gradients the CNN can exploit from step 0. Optionally appends two CoordConv channels
(normalised forward/lateral grid position) so the CNN can localise within the
ego-centric grid (translation-invariance is a liability here).

Convention (channel 0): +1 at the corridor centreline, 0 at the corridor edge,
negative outside the corridor / out of world / inside an obstacle. Grid geometry and
anchor pose exactly mirror render_bev so it is a drop-in image replacement.
"""

from __future__ import annotations

from .backend import xp
from .corridor import min_dist_to_centerline


def render_bev_sdf(xs, ys, cfg, ax, ay, ayaw, coord=False,
                   ox=None, oy=None, orad=None, ovalid=None):
    """(N, C, S, S) float32 signed-distance BEV in [-1,1]. C = 1 (+2 if coord).

    Same grid/anchor convention as render_bev: xs (P,), ys (N,P) centreline;
    ax, ay, ayaw (N,) anchor pose (forward -> image-up). corridor half-width from
    cfg.world.corridor_half_m sets the zero level.
    """
    S = int(cfg.obs.bev_size)
    R = float(cfg.obs.bev_range_m)
    N = ax.shape[0]
    f32 = xp.float32

    coords = xp.linspace(-R, R, S).astype(f32)           # (S,)
    fwd = coords[::-1]                                    # rows: top = farthest ahead
    lat = coords                                          # cols: left(-) .. right(+)
    LF, LL = xp.meshgrid(fwd, lat, indexing="ij")        # (S,S)

    cyaw = xp.cos(ayaw).astype(f32); syaw = xp.sin(ayaw).astype(f32)
    LF_ = LF[None]; LL_ = LL[None]
    wx = ax[:, None, None].astype(f32) + LF_ * cyaw[:, None, None] - LL_ * syaw[:, None, None]
    wy = ay[:, None, None].astype(f32) + LF_ * syaw[:, None, None] + LL_ * cyaw[:, None, None]

    fx = wx.reshape(N, S * S); fy = wy.reshape(N, S * S)
    dist = min_dist_to_centerline(xs.astype(f32), ys.astype(f32), fx, fy)   # (N, S*S)
    corridor = f32(cfg.world.corridor_half_m)
    W = f32(cfg.world.width_m); H = f32(cfg.world.height_m)
    in_world = (fx >= 0) & (fx <= W) & (fy >= 0) & (fy <= H)

    # signed distance to the corridor edge, normalised by corridor half-width:
    # +1 at centreline, 0 at edge, negative outside (clipped to [-1,1]).
    signed = xp.clip((corridor - dist) / corridor, -1.0, 1.0).astype(f32)

    if ox is not None and orad is not None:
        dx = fx[:, :, None] - ox[:, None, :]
        dy = fy[:, :, None] - oy[:, None, :]
        inside_obs = xp.any(ovalid[:, None, :] & ((dx * dx + dy * dy) < (orad[:, None, :] ** 2)), axis=2)
        signed = xp.where(inside_obs, f32(-1.0), signed)

    signed = xp.where(in_world, signed, f32(-1.0))
    sdf = signed.reshape(N, S, S).astype(f32)
    if not coord:
        return sdf[:, None, :, :]

    fwd_n = (LF / R).astype(f32)                          # (S,S) in [-1,1]
    lat_n = (LL / R).astype(f32)
    fwd_c = xp.broadcast_to(fwd_n[None], (N, S, S)).astype(f32)
    lat_c = xp.broadcast_to(lat_n[None], (N, S, S)).astype(f32)
    return xp.stack([sdf, fwd_c, lat_c], axis=1).astype(f32)
