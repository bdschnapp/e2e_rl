"""Path mixture generator (first-class port of train_lab_model._patch_path_generator).

Produces a per-episode reference path (xx, yy, goal_pose) as a mixture over six
geometry kinds. Driven by PathConfig (probabilities, spacing, vert_offset). The
path spacing is a config field (parity default 1.0 m; the speed-scaled-lookahead
improvement sets it ~0.1 m). Curvature is defined in METERS, so the bend shapes
are scale-invariant.

Kinds: straight, gentle, sharp, winding, lab_seam, lab_corner.
`mild_only` collapses the mixture to straight+gentle (baseline isolation).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def generate_path(np_random, world, path_cfg):
    """Return (xx, yy, goal_pose, kind). np_random is a numpy Generator."""
    world_width_m = world.width_m
    x0 = path_cfg.x_start_m
    x_end = world_width_m - path_cfg.x_end_margin_m
    vert_offset = path_cfg.vert_offset_m
    spacing = path_cfg.spacing_m

    xx = np.arange(x0, x_end, spacing)
    n_pts = len(xx)

    kinds = ["straight", "gentle", "sharp", "winding", "lab_seam", "lab_corner",
             "lab_chicane"]
    kp = {"straight": 0.5, "gentle": 0.5} if path_cfg.mild_only else path_cfg.kind_probs
    probs = np.array([float(kp.get(k, 0.0)) for k in kinds], dtype=float)
    probs = probs / probs.sum()          # unlisted kinds -> 0.0; renormalise
    mode = str(np_random.choice(kinds, p=probs))

    if mode == "straight":
        y_local = np.zeros(n_pts, dtype=float)

    elif mode == "gentle":
        n = 6
        ctrl_x = np.linspace(x0, x_end, n)
        ctrl_y = -1.0 + 2.0 * np_random.random(n)
        ctrl_y[0] = (-0.25 / 50) + np_random.random() * (0.5 / 50)
        cs = CubicSpline(ctrl_x, ctrl_y, bc_type=((1, 0.0), "not-a-knot"))
        y_local = cs(xx) * 5.0

    elif mode == "sharp":
        n_bends = int(np_random.integers(4, 9))
        margin = 20.0
        base_centers = np.linspace(x0 + margin, x_end - margin, n_bends)
        gap = (x_end - x0 - 2 * margin) / max(1, n_bends - 1)
        jitter = (np_random.random(n_bends) - 0.5) * gap * 0.4
        bend_xs = sorted((base_centers + jitter).tolist())
        sign = 1.0 if np_random.random() < 0.5 else -1.0
        y_local = np.zeros(n_pts, dtype=float)
        bs = path_cfg.bend_scale
        for bend_x in bend_xs:
            # scale WIDTH only (radius ~ width^2/dy) so turns widen without the
            # lateral excursion (dy) growing out of the world.
            width = float(np_random.uniform(1.0, 1.5)) * bs
            dy = sign * float(np_random.uniform(2.5, 4.0))
            sign = -sign
            ramp = (np.tanh((xx - bend_x) / width) + 1.0) * 0.5
            y_local += dy * ramp

    elif mode == "lab_seam":
        bend_x = float(np_random.uniform(x0 + 6.0, x_end - 4.0))
        width = float(np_random.uniform(0.4, 0.7))
        sign = 1.0 if np_random.random() < 0.5 else -1.0
        dy = sign * float(np_random.uniform(1.5, 3.0)) * path_cfg.curve_scale
        ramp = (np.tanh((xx - bend_x) / width) + 1.0) * 0.5
        y_local = dy * ramp

    elif mode == "lab_corner":
        sign = 1.0 if np_random.random() < 0.5 else -1.0
        bend1_x = float(np_random.uniform(x0 + 10.0, x0 + 15.0))
        width1 = float(np_random.uniform(0.3, 0.8))
        dy1 = sign * float(np_random.uniform(2.0, 6.0)) * path_cfg.curve_scale
        bend2_x = float(np_random.uniform(bend1_x + 100.0, x_end - 10.0))
        width2 = float(np_random.uniform(0.3, 0.8))
        dy2 = -dy1
        ramp1 = (np.tanh((xx - bend1_x) / width1) + 1.0) * 0.5
        ramp2 = (np.tanh((xx - bend2_x) / width2) + 1.0) * 0.5
        y_local = dy1 * ramp1 + dy2 * ramp2

    elif mode == "lab_chicane":
        # y = sign * [ P*tanh(N*(x-c1)) {x<mid} ; -P*tanh(N*(x-c2)) {x>=mid} ]
        # committed out-and-back bump: entry/exit at -P, flat +P middle; two turns
        # `spacing` apart, centred in the path span. curve_scale scales P.
        P = path_cfg.chicane_amplitude_m * path_cfg.curve_scale
        Nn = path_cfg.chicane_slope
        spacing_turns = path_cfg.chicane_turn_spacing_m
        sign = 1.0 if np_random.random() < 0.5 else -1.0
        span = xx[-1] - xx[0]
        c1 = xx[0] + max(2.0, 0.5 * (span - spacing_turns))
        c2 = c1 + spacing_turns
        mid = 0.5 * (c1 + c2)
        bump = np.where(xx < mid, P * np.tanh(Nn * (xx - c1)),
                        -P * np.tanh(Nn * (xx - c2)))
        y_local = sign * bump

    else:  # winding
        n_bends = int(np_random.choice([3, 5]))
        s = float(np_random.uniform(0.5, 0.8))
        width = float(np_random.uniform(0.8, 1.5)) * path_cfg.bend_scale
        initial_sign = 1.0 if np_random.random() < 0.5 else -1.0
        margin = 18.0
        base_xs = np.linspace(x0 + margin, x_end - margin, n_bends)
        gap = (x_end - x0 - 2.0 * margin) / max(1, n_bends - 1)
        jitter = (np_random.random(n_bends) - 0.5) * gap * 0.3
        bend_xs = sorted((base_xs + jitter).tolist())
        interval_slopes = [0.0]
        for i in range(n_bends - 1):
            sgn = initial_sign if (i % 2 == 0) else -initial_sign
            interval_slopes.append(sgn * s)
        interval_slopes.append(0.0)
        y_local = np.zeros(n_pts, dtype=float)
        for bend_x, slope_prev, slope_next in zip(bend_xs, interval_slopes[:-1], interval_slopes[1:]):
            delta = slope_next - slope_prev
            if abs(delta) < 1e-9:
                continue
            arg = (xx - bend_x) / width
            arg_0 = (x0 - bend_x) / width
            ln_cosh = np.log(np.cosh(arg))
            ln_cosh_0 = float(np.log(np.cosh(arg_0)))
            y_local += (delta / 2.0) * ((xx - x0) + width * (ln_cosh - ln_cosh_0))

    yy = y_local + vert_offset
    x_g = xx[-1]
    y_g = yy[-1]
    yaw_g = float(np.arctan2(yy[-1] - yy[-2], xx[-1] - xx[-2]))
    return xx, yy, (x_g, y_g, yaw_g), mode
