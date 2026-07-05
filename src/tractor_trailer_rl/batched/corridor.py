"""Analytic lane corridor: distance-to-centerline instead of a rasterised grid.

The scalar env builds a per-env occupancy grid with a scipy ``cKDTree`` (blocker
#2), then does Python-loop collision/proximity queries against it (blocker #4).
For a GPU-parallel env, storing N large grids and rebuilding them with a CPU
KD-tree is the wrong shape. Since the lane corridor is defined purely as "within
``corridor_half`` of the centerline", we instead compute the point-to-polyline
distance analytically and on demand:

  * collision  = body point out-of-world OR distance-to-centerline > corridor_half
  * proximity  = ramp on (corridor_half - dist) - veh_half_w
  * lidar      = march sample points and test corridor membership (lidar.py)

This is bit-for-bit *different* from the rasterised grid (which quantises the
corridor to ``grid_res_m`` cells) but is the same corridor to within one cell, and
is arguably more accurate (no rasterisation aliasing). The x-samples ``xs`` are a
uniform arange, so the nearest segment to a query is found by an index division
(O(1)) with a small neighbour window rather than an O(P) scan.
"""

from __future__ import annotations

from .backend import xp


def _pt_seg_dist(qx, qy, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    t = (( qx - ax) * abx + (qy - ay) * aby) / xp.where(denom > 0, denom, 1.0)
    t = xp.clip(t, 0.0, 1.0)
    px = ax + t * abx
    py = ay + t * aby
    return xp.sqrt((qx - px) ** 2 + (qy - py) ** 2)


def min_dist_to_centerline(xs, ys, qx, qy, window: int = 3):
    """Min distance from query points to each env's centerline polyline.

    xs: (P,) shared uniform samples. ys: (N,P). qx,qy: (N,K). Returns (N,K).
    Uses the uniform-x structure: nearest segment index ~ (qx-x0)/seg_dx, then a
    +/- ``window`` neighbourhood is checked exactly (covers local curvature).
    """
    N, P = ys.shape
    S = P - 1
    seg_dx = float(xs[1] - xs[0])
    x0 = float(xs[0])
    si = xp.clip(xp.floor((qx - x0) / seg_dx).astype(xp.int64), 0, S - 1)  # (N,K)
    ar = xp.arange(N)[:, None]

    best = None
    for d in range(-window, window + 1):
        sidx = xp.clip(si + d, 0, S - 1)                    # (N,K)
        ax = xs[sidx]; bx = xs[sidx + 1]                    # gather from shared xs
        ay = ys[ar, sidx]; by = ys[ar, sidx + 1]           # gather per-env ys
        dist = _pt_seg_dist(qx, qy, ax, ay, bx, by)
        best = dist if best is None else xp.minimum(best, dist)
    return best


def body_points(veh, cfg):
    """(N,6) world x,y of the vehicle body samples used for collision.
    3 along the tractor + 3 along the trailer (axle->hitch), matching the scalar
    ``LaneFollowingEnv._collision`` sample set."""
    tl = cfg.vehicle.tractor_length_m
    L = cfg.vehicle.trailer_length_m
    cpx, spx = xp.cos(veh.p), xp.sin(veh.p)
    ctx, stx = xp.cos(veh.tyaw), xp.sin(veh.tyaw)
    xs_pts = [veh.x + f * tl * cpx for f in (-0.5, 0.0, 0.5)]
    ys_pts = [veh.y + f * tl * spx for f in (-0.5, 0.0, 0.5)]
    xs_pts += [veh.tx + f * L * ctx for f in (0.0, 0.5, 1.0)]
    ys_pts += [veh.ty + f * L * stx for f in (0.0, 0.5, 1.0)]
    return xp.stack(xs_pts, axis=1), xp.stack(ys_pts, axis=1)


def collision(xs, ys, veh, cfg):
    """(N,) bool: any body point out-of-world or outside the lane corridor."""
    px, py = body_points(veh, cfg)                       # (N,6)
    W = cfg.world.width_m; H = cfg.world.height_m
    oob = (px < 0) | (px > W) | (py < 0) | (py > H)
    dist = min_dist_to_centerline(xs, ys, px, py)
    blocked = dist > cfg.world.corridor_half_m
    return xp.any(oob | blocked, axis=1)


MAX_PROXIMITY_PENALTY = 5.0


def proximity_penalty(xs, ys, veh, cfg):
    """(N,) per-step proximity penalty (analytic port of proximity.py).

    Clearance to the corridor edge = corridor_half - dist_to_centerline - veh_half.
    Ramp quadratically from 0 at ``threshold`` to MAX at 0 clearance. Threshold is
    the derived 0.5*(corridor_half - veh_half) (fix C)."""
    half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    corridor = cfg.world.corridor_half_m
    threshold = 0.5 * max(0.1, corridor - half_w)
    qx = xp.stack([veh.x, veh.tx], axis=1)
    qy = xp.stack([veh.y, veh.ty], axis=1)
    dist = min_dist_to_centerline(xs, ys, qx, qy)         # (N,2)
    clearance = xp.clip((corridor - dist) - half_w, 0.0, None)
    clearance = xp.min(clearance, axis=1)                 # nearest of the two bodies
    t = xp.clip(1.0 - clearance / threshold, 0.0, 1.0)
    pen = t * t * MAX_PROXIMITY_PENALTY
    return xp.where(clearance >= threshold, 0.0, pen)
