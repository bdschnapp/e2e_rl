"""Pure path-geometry helpers (numpy only).

Ported verbatim-equivalent from e2e_rl LineFollowing.get_errors /
compute_curvature, de-globalised (config values passed in) and with the
forward/reverse heading convention selectable by a flag instead of a subclass.

Parity-critical details preserved exactly:
  * cross-track sign flips when the nearest path point is ABOVE the query (yy>y).
  * the heading tangent uses ``np.arctan(dy/dx)`` (NOT arctan2) on the nearest
    segment — this is the original convention. For a reverse vehicle (yaw ~ pi)
    that yields heading error ~ pi, which the reverse branch wraps to ~0 (the
    2026-06-26 fix A).
  * curvature uses a 3-point finite difference at the nearest+lookahead index,
    clamped to [1, N-2], clipped to +/- max_curvature.
"""

from __future__ import annotations

import numpy as np


def wrap_to_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def nearest_index(xs: np.ndarray, ys: np.ndarray, px: float, py: float) -> int:
    dx = xs - px
    dy = ys - py
    return int(np.argmin(dx * dx + dy * dy))


def path_errors(
    xs: np.ndarray,
    ys: np.ndarray,
    px: float,
    py: float,
    yaw: float,
    *,
    reverse: bool,
    error_theta_scale: float = 1.0,
    tangent_offset: int = 1,
) -> tuple[float, float]:
    """Cross-track distance error and heading error of a point (px, py, yaw)
    against the polyline (xs, ys).

    tangent_offset: index step used for the heading tangent. 1 == the original
    e2e_rl nearest/nearest+1 segment (parity). >1 is the speed-scaled preview.
    """
    n = len(xs)
    dx = xs - px
    dy = ys - py
    dist = np.sqrt(dx * dx + dy * dy)
    i = int(np.argmin(dist))

    error = float(dist[i])
    if ys[i] > py:
        error = -error

    # Heading tangent on the nearest segment (arctan, not arctan2 — original).
    j = i + max(1, int(tangent_offset))
    if j <= n - 1:
        a, b = i, j
    else:
        a, b = i - max(1, int(tangent_offset)), i
        a = max(0, a)
    denom = (xs[b] - xs[a])
    theta = float(np.arctan((ys[b] - ys[a]) / denom)) if denom != 0 else (
        np.pi / 2 if (ys[b] - ys[a]) > 0 else -np.pi / 2)

    raw = yaw - theta
    if reverse:
        # fix A: a correctly-reversing vehicle faces ~pi from the forward
        # tangent; wrap so it reads ~0 instead of ~pi.
        raw = (raw % (2.0 * np.pi)) - np.pi
    return error, float(raw * error_theta_scale)


def curvature_at(
    xs: np.ndarray,
    ys: np.ndarray,
    query_x: float,
    query_y: float,
    lookahead_steps: int,
    max_curvature: float = 0.3,
) -> float:
    """Signed path curvature (1/m) at nearest(query) + lookahead_steps.
    Ported from e2e_rl compute_curvature."""
    n = len(xs)
    dx = xs - query_x
    dy = ys - query_y
    nearest_idx = int(np.argmin(dx * dx + dy * dy))
    i = nearest_idx + int(lookahead_steps)
    i = max(1, min(i, n - 2))
    x_im1, x_i, x_ip1 = xs[i - 1], xs[i], xs[i + 1]
    y_im1, y_i, y_ip1 = ys[i - 1], ys[i], ys[i + 1]
    x_p = (x_ip1 - x_im1) * 0.5
    y_p = (y_ip1 - y_im1) * 0.5
    x_pp = (x_ip1 - 2.0 * x_i + x_im1)
    y_pp = (y_ip1 - 2.0 * y_i + y_im1)
    denom = (x_p ** 2 + y_p ** 2) ** 1.5 + 1e-6
    kappa = (x_p * y_pp - y_p * x_pp) / denom
    return float(np.clip(kappa, -max_curvature, max_curvature))


def lookahead_steps_for_speed(
    speed: float, preview_time_s: float, spacing_m: float,
    dist_min_m: float, dist_max_m: float,
) -> int:
    """Convert a preview TIME into a path-sample index offset, given fine fixed
    spacing. lookahead_distance = clip(|v|*T, dist_min, dist_max)."""
    dist = float(np.clip(abs(speed) * preview_time_s, dist_min_m, dist_max_m))
    return max(1, int(round(dist / spacing_m)))
