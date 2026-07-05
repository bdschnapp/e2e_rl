"""Batched path geometry (vectorised port of ``tractor_trailer_rl.geometry``).

All envs share the same x-samples ``xs`` (shape (P,), an ``arange`` identical
across the batch) and differ only in ``ys`` (shape (N, P)). Query points are
shape (N,). Every function returns (N,) results. Parity-critical conventions are
preserved exactly: cross-track sign flips when the nearest path point is above the
query; the heading tangent uses ``arctan`` (not arctan2) on the nearest segment;
curvature is a 3-point finite difference clamped to [1, P-2] and clipped.
"""

from __future__ import annotations

import numpy as np

from .backend import xp


def _gather(ys, idx):
    """ys: (N,P), idx: (N,) -> (N,) gather along axis 1."""
    n = ys.shape[0]
    return ys[xp.arange(n), idx]


def nearest_index(xs, ys, px, py):
    dx = xs[None, :] - px[:, None]
    dy = ys - py[:, None]
    return xp.argmin(dx * dx + dy * dy, axis=1)


def path_errors(xs, ys, px, py, yaw, *, reverse, error_theta_scale=1.0, tangent_offset=1):
    """Cross-track error e_y (N,) and heading error e_psi (N,)."""
    P = xs.shape[0]
    dx = xs[None, :] - px[:, None]
    dy = ys - py[:, None]
    dist = xp.sqrt(dx * dx + dy * dy)
    i = xp.argmin(dist, axis=1)                      # (N,)
    n = ys.shape[0]
    arangeN = xp.arange(n)

    error = dist[arangeN, i]
    ys_i = ys[arangeN, i]
    error = xp.where(ys_i > py, -error, error)

    off = max(1, int(tangent_offset))
    j = i + off
    use_fwd = j <= (P - 1)
    a = xp.where(use_fwd, i, xp.clip(i - off, 0, P - 1))
    b = xp.where(use_fwd, j, i)
    denom = xs[b] - xs[a]                              # (N,) nonzero for uniform xs
    num = _gather(ys, b) - _gather(ys, a)
    theta = xp.where(denom != 0, xp.arctan(num / xp.where(denom != 0, denom, 1.0)),
                     xp.where(num > 0, np.pi / 2, -np.pi / 2))
    raw = yaw - theta
    if reverse:
        raw = (raw % (2.0 * np.pi)) - np.pi
    return error, raw * error_theta_scale


def curvature_at(xs, ys, qx, qy, lookahead_steps, max_curvature=0.3):
    """Signed path curvature (1/m) (N,) at nearest(query)+lookahead."""
    P = xs.shape[0]
    dx = xs[None, :] - qx[:, None]
    dy = ys - qy[:, None]
    nearest = xp.argmin(dx * dx + dy * dy, axis=1)
    i = xp.clip(nearest + int(lookahead_steps), 1, P - 2)

    xm1, x0, xp1 = xs[i - 1], xs[i], xs[i + 1]
    ym1 = _gather(ys, i - 1); y0 = _gather(ys, i); yp1 = _gather(ys, i + 1)
    x_p = (xp1 - xm1) * 0.5
    y_p = (yp1 - ym1) * 0.5
    x_pp = xp1 - 2.0 * x0 + xm1
    y_pp = yp1 - 2.0 * y0 + ym1
    denom = (x_p ** 2 + y_p ** 2) ** 1.5 + 1e-6
    kappa = (x_p * y_pp - y_p * x_pp) / denom
    return xp.clip(kappa, -max_curvature, max_curvature)
