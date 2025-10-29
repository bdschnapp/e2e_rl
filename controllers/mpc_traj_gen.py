import numpy as np
from VehicleModels.tractor_trailer import StateSpaceTractorTrailer


def generate_trajectory(x, y, vehicle: StateSpaceTractorTrailer):
    """
    Build (N+1, 4) trajectory rows = [X, Y, psi1, psi2] for TractorTrailerSteeringMPC.
      - Row 0 is the current measured state from `vehicle`:
          X,Y        -> trailer axle position
          psi1       -> trailer yaw
          psi2       -> hitch angle = tractor_yaw - trailer_yaw (wrapped)
      - Rows 1..N are path-based references sampled ahead along (x,y)

    Uses:
      vehicle.dt      (sampling time)
      vehicle.xd      (longitudinal speed, m/s)
      vehicle.p       (tractor yaw)
      vehicle.trailer.x, .y, .yaw
      vehicle.lf, vehicle.lr  (to compute L1 = wheelbase)

    Assumptions:
      Hitch at tractor rear axle -> L2C ≈ 0 (small epsilon for numeric stability).
      Horizon N defaults to 40 (tweak here if you want it elsewhere).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must be equal-length arrays with at least 2 points.")

    # --- model params from your classes ---
    Ts = float(getattr(vehicle, "dt", 0.1))
    vx = float(max(1e-6, getattr(vehicle, "xd", 1.0)))               # forward speed for spacing
    L1 = float(getattr(vehicle, "lf", 1.2) + getattr(vehicle, "lr", 1.6))  # tractor wheelbase
    L2C = np.finfo(float).eps  # hitch at rear axle -> ~0 to avoid singularities
    N = 40

    # --- current measured trailer state (row 0) ---
    X0 = float(vehicle.trailer.x)
    Y0 = float(vehicle.trailer.y)
    psi1_0 = float(vehicle.trailer.yaw)      # trailer yaw
    psi2_0 = float(_wrap_to_pi(vehicle.p - vehicle.trailer.yaw))  # hitch angle

    # --- arclength along given path & nearest index to trailer ---
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total_len = float(s[-1])

    dists = np.hypot(x - X0, y - Y0)
    i0 = int(np.argmin(dists))
    s0 = float(s[i0])

    # sampling step ahead along path ~ how far the trailer moves in one MPC step
    ds = max(1e-3, vx * Ts)

    # build N reference samples ahead (row 0 is measured, so we produce N future rows)
    s_targets = np.clip(s0 + ds * np.arange(1, N + 1), 0.0, total_len)
    x_ref = np.interp(s_targets, s, x)
    y_ref = np.interp(s_targets, s, y)

    # --- compute path tangent heading psi1 and curvature kappa on the resampled points ---
    psi1_path, kappa = _heading_and_curvature(x_ref, y_ref)

    # --- solve psi2 from curvature using same closed-form as your MPC (continuity branch) ---
    psi2_1, psi2_2 = _solve_psi2_batch(kappa, L1, L2C)
    psi2_ref = _pick_continuous_branch(psi2_1, psi2_2, psi2_0)

    # --- assemble trajectory ---
    traj = np.zeros((N + 1, 4), dtype=float)
    traj[0, :] = [X0, Y0, psi1_0, psi2_0]
    traj[1:, 0] = x_ref
    traj[1:, 1] = y_ref
    traj[1:, 2] = psi1_path
    traj[1:, 3] = psi2_ref

    return traj


# ------------------------ helpers ------------------------
def _wrap_to_pi(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _central_deriv(v):
    v = np.asarray(v, dtype=float)
    n = v.size
    dv = np.zeros_like(v)
    if n >= 3:
        dv[1:-1] = 0.5 * (v[2:] - v[:-2])
        dv[0] = v[1] - v[0]
        dv[-1] = v[-1] - v[-2]
    elif n == 2:
        dv[0] = v[1] - v[0]
        dv[1] = v[1] - v[0]
    return dv

def _heading_and_curvature(x_ref, y_ref):
    # Use uniform param t on the resampled sequence
    dx_dt = _central_deriv(x_ref)
    dy_dt = _central_deriv(y_ref)
    d2x_dt2 = _central_deriv(dx_dt)
    d2y_dt2 = _central_deriv(dy_dt)

    denom = (dx_dt*dx_dt + dy_dt*dy_dt)**1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(denom > 1e-12, (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denom, 0.0)

    psi1 = np.arctan2(dy_dt, dx_dt)
    psi1 = np.where(np.isfinite(psi1), psi1, 0.0)
    kappa = np.where(np.isfinite(kappa), kappa, 0.0)
    return psi1, kappa

def _solve_psi2_batch(kappa, L1, L2C):
    # Solve sin(psi2) - L1*kappa*cos(psi2) + L2C*kappa = 0
    kappa = np.asarray(kappa, dtype=float)
    R = np.sqrt(1.0 + (L1 * kappa) ** 2)
    alpha = np.arctan2(-L1 * kappa, 1.0)
    rhs = np.clip(-L2C * kappa / R, -1.0, 1.0)
    asin_rhs = np.arcsin(rhs)
    psi2_1 = _wrap_to_pi(asin_rhs - alpha)
    psi2_2 = _wrap_to_pi(np.pi - asin_rhs - alpha)
    return psi2_1, psi2_2

def _pick_continuous_branch(psi2_1, psi2_2, psi2_0):
    psi2_1 = np.asarray(psi2_1, dtype=float)
    psi2_2 = np.asarray(psi2_2, dtype=float)
    n = psi2_1.size
    out = np.empty(n, dtype=float)
    if n == 0:
        return out

    # choose first based on proximity to current hitch angle
    first_cands = np.array([psi2_1[0], psi2_2[0]])
    idx = int(np.argmin(np.abs(_wrap_to_pi(first_cands - psi2_0))))
    out[0] = first_cands[idx]

    # propagate continuity
    for i in range(1, n):
        cands = np.array([psi2_1[i], psi2_2[i]])
        idx = int(np.argmin(np.abs(_wrap_to_pi(cands - out[i-1]))))
        out[i] = cands[idx]
    return out