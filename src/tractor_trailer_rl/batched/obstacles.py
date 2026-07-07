"""Batched in-lane obstacles + the "hidden local planner" reference path (Track D).

Ported from the pygame ``ObstacleAvoidance`` env, but vectorised and analytic (no
occupancy grid / pygame mask -- consistent with the corridor.py approach). Every
routine keeps the leading ``(N,)`` env axis. Obstacles are circles ``(x, y, r)``
stored as ``(N, K)`` arrays with a validity mask (unused slots are ignored).

Three groups:

* placement  (`place_host`)      -- host/numpy, at reset only. Places circles near
  each env's centreline with a controllable easy/squeeze/blocked mix so the
  stop-gate has a learnable difficulty signal.
* analysis   (`plan_offsets`, `difficulty`) -- backend, at reset only. The APF
  lateral-shift planner produces the obstacle-avoiding reference offset (the reward
  tracks centreline+offset, "hidden" from the observation); `difficulty` is the
  max-free-lateral-gap metric that gates the stop reward.
* per-step   (`lidar`, `collision`, `proximity`) -- backend, hot path. Analytic
  ray-circle / point-circle tests, min-combined with the corridor equivalents.
"""

from __future__ import annotations

import numpy as np

from .backend import xp


# ------------------------------------------------------------------ placement
def place_host(rng, xs_np, ys_np, cfg):
    """Place one obstacle cluster per env on the host, by layout category.

    xs_np: (P,) shared samples. ys_np: (M,P) centrelines for the M envs being reset.
    Each env draws a category from ``cfg.obstacle.layout_probs`` giving a controlled
    difficulty spread (clear/shift = passable go-around; squeeze = hard-but-passable;
    blocked = impassable). Returns numpy arrays (M,K): ox, oy, orad, olat (signed
    lateral offset), ostation (x), ovalid. K = cfg.obstacle.max_obstacles.
    """
    oc = cfg.obstacle
    M, P = ys_np.shape
    K = int(oc.max_obstacles)
    ox = np.zeros((M, K)); oy = np.zeros((M, K)); orad = np.zeros((M, K))
    olat = np.zeros((M, K)); ostation = np.zeros((M, K))
    ovalid = np.zeros((M, K), dtype=bool)

    x0, x1 = float(xs_np[0]), float(xs_np[-1])
    lo_x = x0 + oc.start_margin_m
    hi_x = min(x1 - oc.goal_margin_m, cfg.world.width_m * 0.85 - oc.goal_margin_m)
    if hi_x <= lo_x:                      # world too short for the margins -> no obstacles
        return ox, oy, orad, olat, ostation, ovalid
    seg_dx = float(xs_np[1] - xs_np[0])
    veh_w = float(max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m))
    L = float(cfg.world.lane_centerline_half_width_m)

    cats = list(oc.layout_probs.keys())
    probs = np.array([oc.layout_probs[c] for c in cats], dtype=float)
    probs = probs / probs.sum()

    def _add(m, slot, sx, cy, nx, ny, lat, r):
        ox[m, slot] = sx + lat * nx; oy[m, slot] = cy + lat * ny
        orad[m, slot] = r; olat[m, slot] = lat; ostation[m, slot] = sx
        ovalid[m, slot] = True
        return slot + 1

    for m in range(M):
        cat = cats[int(rng.choice(len(cats), p=probs))]
        sx = float(rng.uniform(lo_x, hi_x))
        pi = int(np.clip(round((sx - x0) / seg_dx), 0, P - 1))
        cy = float(ys_np[m, pi])
        pj = min(pi + 1, P - 1); pk = max(pi - 1, 0)
        tx = float(xs_np[pj] - xs_np[pk]); ty = float(ys_np[m, pj] - ys_np[m, pk])
        tn = np.hypot(tx, ty) + 1e-9
        tx, ty = tx / tn, ty / tn
        nx, ny = -ty, tx                            # left normal
        side = float(rng.choice([-1.0, 1.0]))
        slot = 0
        if cat == "clear":                          # obstacle hugs the lane edge -> easy
            _add(m, slot, sx, cy, nx, ny, side * float(rng.uniform(3.6, 4.4)),
                 float(rng.uniform(0.8, 1.3)))
        elif cat == "shift":                        # mid-lane obstacle -> shift over
            _add(m, slot, sx, cy, nx, ny, side * float(rng.uniform(2.0, 3.0)),
                 float(rng.uniform(1.0, 1.6)))
        elif cat == "squeeze":                      # opposing pair, passable central gap
            r = float(rng.uniform(1.0, 1.5))
            gap = veh_w * float(rng.uniform(1.25, 1.6))     # > vehicle width -> passable
            d = 0.5 * gap + r
            d = min(d, L + r - 0.1)                          # keep partly inside the lane
            slot = _add(m, slot, sx, cy, nx, ny, +d, r)
            _add(m, slot, sx, cy, nx, ny, -d, float(rng.uniform(1.0, 1.5)))
        else:                                       # "blocked": big central obstacle -> stop
            _add(m, slot, sx, cy, nx, ny, side * float(rng.uniform(0.0, 0.6)),
                 float(rng.uniform(2.4, 3.0)))
    return ox, oy, orad, olat, ostation, ovalid


# ------------------------------------------------------------- hidden planner
def _box_smooth(a, w):
    """(N,P) box filter of odd width w along axis 1 (edge-padded)."""
    w = int(w)
    if w <= 1:
        return a
    pad = w // 2
    ap = xp.pad(a, ((0, 0), (pad, pad)), mode="edge")
    c = xp.cumsum(ap, axis=1)
    z = xp.zeros((a.shape[0], 1), dtype=c.dtype)
    c = xp.concatenate([z, c], axis=1)
    return (c[:, w:] - c[:, :-w]) / float(w)


def plan_offsets(xs, ys, ox, oy, orad, ovalid, cfg):
    """Artificial-potential-field lateral shift -> (N,P) offset added to ys.

    For each path point, every intruding obstacle within the longitudinal
    influence window pushes the reference away (side = away from the obstacle);
    contributions sum, are clipped to the corridor, and box-smoothed. The offset is
    applied as a y-shift (the perpendicular normal is ~vertical for the
    near-horizontal thesis paths); the reward tracks ys+offset.
    """
    oc = cfg.obstacle
    N, P = ys.shape
    veh_w = float(max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m))
    max_shift = max(0.0, cfg.world.lane_centerline_half_width_m - 0.5 * veh_w)

    # tangent / left-normal at every path point (uniform xs)
    xl = xp.concatenate([xs[:1], xs[:-1]]); xr = xp.concatenate([xs[1:], xs[-1:]])
    txr = (xr - xl)[None, :]                          # (1,P)
    yl = xp.concatenate([ys[:, :1], ys[:, :-1]], axis=1)
    yr = xp.concatenate([ys[:, 1:], ys[:, -1:]], axis=1)
    tyr = yr - yl                                     # (N,P)
    tn = xp.sqrt(txr * txr + tyr * tyr) + 1e-9
    tx = txr / tn; ty = tyr / tn                      # (N,P)
    nx = -ty; ny = tx

    px = xs[None, :, None]                            # (1,P,1)
    py = ys[:, :, None]                               # (N,P,1)
    vx = ox[:, None, :] - px                          # (N,P,K)
    vy = oy[:, None, :] - py
    lat = vx * nx[:, :, None] + vy * ny[:, :, None]   # signed lateral of obstacle
    lon = vx * tx[:, :, None] + vy * ty[:, :, None]
    r_infl = orad + 0.75 * veh_w                      # (N,K)
    pen = r_infl[:, None, :] - xp.abs(lat)            # penetration into the path corridor
    active = ovalid[:, None, :] & (pen > 0) & (xp.abs(lon) < oc.influence_radius_m)
    direction = -xp.sign(lat)                         # push away from the obstacle
    strength = pen / (r_infl[:, None, :] + 1e-9)
    contrib = xp.where(active, direction * strength * (r_infl[:, None, :] + 0.2), 0.0)
    offset = xp.sum(contrib, axis=2)                  # (N,P)
    offset = xp.clip(offset, -max_shift, max_shift)
    return _box_smooth(offset, 11)


# ------------------------------------------------------------------ difficulty
def difficulty(olat, ostation, orad, ovalid, cfg):
    """(N,) layout difficulty in [0,1] from the max free lateral gap.

    At each obstacle's station, scan the lane laterally, mark points blocked by any
    obstacle sharing that station, take the widest contiguous free gap, and map it:
    difficulty = clip(1 - (gap - veh_w)/(lane_w - veh_w), 0, 1). 0 = full lane
    free, 1 = gap <= vehicle width (impassable). The scalar is the max over stations.
    """
    oc = cfg.obstacle
    N, K = olat.shape
    L = float(cfg.world.lane_centerline_half_width_m)
    veh_w = float(max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m))
    lane_w = 2.0 * L
    G = int(oc.difficulty_bins)
    ell = xp.linspace(-L, L, G)                       # (G,)
    bin_w = float(ell[1] - ell[0]) if G > 1 else lane_w

    ost_k = ostation[:, :, None, None]               # (N,K,1,1) anchor station
    ost_j = ostation[:, None, None, :]               # (N,1,1,K) other obstacles
    or_k = orad[:, :, None, None]
    or_j = orad[:, None, None, :]
    lat_j = olat[:, None, None, :]
    val_j = ovalid[:, None, None, :]
    ellg = ell[None, None, :, None]                  # (1,1,G,1)
    near = xp.abs(ost_j - ost_k) < (or_j + or_k + 1.0)
    covers = xp.abs(ellg - lat_j) < or_j
    blocked = xp.any(val_j & near & covers, axis=3)  # (N,K,G)
    free = ~blocked

    # longest run of free bins along G (reset-time only; G ~ 61 iterations)
    run = xp.zeros((N, K)); best = xp.zeros((N, K))
    for g in range(G):
        fg = free[:, :, g]
        run = xp.where(fg, run + 1.0, 0.0)
        best = xp.maximum(best, run)
    max_gap = best * bin_w                            # (N,K)

    denom = max(lane_w - veh_w, 1e-6)
    diff_k = xp.clip(1.0 - (max_gap - veh_w) / denom, 0.0, 1.0)
    diff_k = xp.where(ovalid, diff_k, 0.0)           # stations with no obstacle => 0
    return xp.max(diff_k, axis=1) if K > 0 else xp.zeros(N)


def exp_scale(d, k):
    """Convex difficulty scale (e^{k d} - 1)/(e^k - 1) in [0,1]."""
    k = float(k)
    return (xp.exp(k * d) - 1.0) / (float(np.exp(k)) - 1.0)


# ------------------------------------------------------------------ per-step
def lidar(ox, oy, orad, ovalid, px, py, pyaw, cfg):
    """(N,B) normalised nearest ray-circle range in [0,1] (1.0 = no obstacle hit).

    Same beam layout / range as the corridor lidar (corridor.raycast) so the two
    can be min-combined into one lidar vector.
    """
    B = int(cfg.obs.lidar_beams)
    fov = np.float32(np.deg2rad(cfg.obs.lidar_fov_deg))
    maxr = np.float32(cfg.obs.lidar_range_m)
    f32 = xp.float32
    angles = xp.linspace(-fov / 2.0, fov / 2.0, B).astype(f32)         # (B,)
    theta = pyaw.astype(f32)[:, None] + angles[None, :]                # (N,B)
    dirx = xp.cos(theta)[:, :, None]; diry = xp.sin(theta)[:, :, None]  # (N,B,1)
    relx = (ox - px[:, None]).astype(f32)[:, None, :]                  # (N,1,K)
    rely = (oy - py[:, None]).astype(f32)[:, None, :]
    tca = relx * dirx + rely * diry                                    # (N,B,K)
    d2 = (relx * relx + rely * rely) - tca * tca
    r2 = (orad * orad).astype(f32)[:, None, :]
    hit = ovalid[:, None, :] & (tca > 0) & (d2 < r2)
    thc = xp.sqrt(xp.clip(r2 - d2, 0.0, None))
    hd = xp.where(hit, tca - thc, maxr)
    dmin = xp.min(hd, axis=2)                                          # (N,B)
    dmin = xp.clip(dmin, 0.0, maxr)
    return (dmin / maxr).astype(f32)


def _body_min_surface_dist(ox, oy, orad, ovalid, body_x, body_y):
    """(N,) min distance from any vehicle body point to any obstacle surface."""
    dx = body_x[:, :, None] - ox[:, None, :]         # (N,S,K)
    dy = body_y[:, :, None] - oy[:, None, :]
    d = xp.sqrt(dx * dx + dy * dy) - orad[:, None, :]
    big = xp.full_like(d, 1e9)
    d = xp.where(ovalid[:, None, :], d, big)
    return xp.min(d.reshape(d.shape[0], -1), axis=1)   # (N,)


def collision(ox, oy, orad, ovalid, body_x, body_y, cfg):
    """(N,) bool: any vehicle body point inside an obstacle (+ body half-width)."""
    half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    return _body_min_surface_dist(ox, oy, orad, ovalid, body_x, body_y) < half_w


MAX_PROXIMITY_PENALTY = 5.0


def proximity(ox, oy, orad, ovalid, body_x, body_y, cfg):
    """(N,) quadratic proximity penalty as the vehicle nears an obstacle surface."""
    half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    threshold = max(0.5, cfg.world.lane_shoulder_m + 1.0)
    clearance = xp.clip(_body_min_surface_dist(ox, oy, orad, ovalid, body_x, body_y) - half_w,
                        0.0, None)
    t = xp.clip(1.0 - clearance / threshold, 0.0, 1.0)
    pen = t * t * MAX_PROXIMITY_PENALTY
    return xp.where(clearance >= threshold, 0.0, pen)
