"""Proximity penalty (pure port of e2e_rl LaneDrivingEnv._min_clearance_to_occ_m
+ _proximity_penalty).

Fix C: the threshold is DERIVED from the lane corridor (0.5 * (corridor_half -
veh_half_w)) so a well-centred vehicle is penalty-free at any lane scale, instead
of the hardcoded 2.0 m that exceeded the lab's 1.41 m lane (always-on penalty).
"""

from __future__ import annotations

import numpy as np

MAX_PROXIMITY_PENALTY = 5.0


def derived_threshold_m(cfg) -> float:
    """Fix C threshold from the lane corridor and vehicle width."""
    veh_half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    centred_clearance = max(0.1, cfg.world.corridor_half_m - veh_half_w)
    return 0.5 * centred_clearance


def min_clearance_to_occ_m(occ_grid, meta, positions, threshold_m, veh_half_w) -> float:
    """Approx clearance (m) from the vehicle bodies to the nearest blocked cell."""
    if occ_grid is None:
        return float("inf")
    search_m = threshold_m + veh_half_w
    r_cells = int(np.ceil(search_m / meta.res_m))
    dy_arr, dx_arr = np.mgrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    dist_arr = np.hypot(dx_arr, dy_arr) * meta.res_m
    within = dist_arr <= search_m
    dy_w = dy_arr[within]; dx_w = dx_arr[within]; d_w = dist_arr[within]

    min_clearance = float("inf")
    for vx, vy in positions:
        gx0 = int((vx - meta.origin_x) / meta.res_m)
        gy0 = int((vy - meta.origin_y) / meta.res_m)
        gx = gx0 + dx_w; gy = gy0 + dy_w
        oob = (gx < 0) | (gx >= meta.width) | (gy < 0) | (gy >= meta.height)
        gx_c = np.clip(gx, 0, meta.width - 1)
        gy_c = np.clip(gy, 0, meta.height - 1)
        blocked = oob | (occ_grid[gy_c, gx_c] == 100)
        if blocked.any():
            clearance = max(0.0, float(d_w[blocked].min()) - veh_half_w)
            min_clearance = min(min_clearance, clearance)
    return min_clearance


def proximity_penalty(occ_grid, meta, positions, cfg) -> float:
    """Per-step proximity penalty, ramping quadratically from 0 at the threshold
    to MAX at zero clearance."""
    if occ_grid is None or meta is None:
        return 0.0
    threshold = derived_threshold_m(cfg)
    veh_half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    clearance = min_clearance_to_occ_m(occ_grid, meta, positions, threshold, veh_half_w)
    if clearance >= threshold:
        return 0.0
    t = 1.0 - clearance / threshold
    return t ** 2 * MAX_PROXIMITY_PENALTY
