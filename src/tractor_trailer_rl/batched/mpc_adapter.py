"""Unbatched MPC evaluated inside the batched env (classical benchmark baseline).

MPC is not batched (one OSQP solve per env per step) — by design. This adapter runs
the vetted e2e_rl OSQP MPC (`controllers/mpc.py`, standalone: numpy/scipy/osqp) in
the SAME batched env as PP/PID/RL, so the RL-vs-classical comparison is fair. Per
step it loops over the (small-N) envs, builds a duck-typed vehicle from the batched
state + the env's path, solves, and converts the desired steer angle to a steer-rate.
Use a small n_envs (per-env QP solves are the cost). The MPC's geometry/limits are
set to the target vehicle (L1, L2, umax, steer-rate) since e2e_rl's MPC ignores args.

`generate_trajectory` (+ helpers) is copied from e2e_rl controllers/mpc_traj_gen.py
to avoid pulling in the e2e_rl env/VehicleModels import chain.
"""

from __future__ import annotations

import os
import io
import contextlib
import importlib.util
import numpy as np

from .backend import to_numpy

_E2ERL_MPC = "/home/ben/Ben/Thesis/e2e_rl/controllers/mpc.py"


def _load_mpc_classes():
    spec = importlib.util.spec_from_file_location("_e2erl_mpc", _E2ERL_MPC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TractorTrailerSteeringMPC, mod.ReverseTractorTrailerMPC


# ------------------------ trajectory generator (copied, self-contained) --------
def _wrap_to_pi(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _central_deriv(v):
    v = np.asarray(v, dtype=float); n = v.size; dv = np.zeros_like(v)
    if n >= 3:
        dv[1:-1] = 0.5 * (v[2:] - v[:-2]); dv[0] = v[1] - v[0]; dv[-1] = v[-1] - v[-2]
    elif n == 2:
        dv[0] = dv[1] = v[1] - v[0]
    return dv


def _heading_and_curvature(x_ref, y_ref):
    dx, dy = _central_deriv(x_ref), _central_deriv(y_ref)
    d2x, d2y = _central_deriv(dx), _central_deriv(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-12, (dx * d2y - dy * d2x) / denom, 0.0)
    psi1 = np.arctan2(dy, dx)
    return np.where(np.isfinite(psi1), psi1, 0.0), np.where(np.isfinite(kappa), kappa, 0.0)


def _solve_psi2_batch(kappa, L1, L2C):
    kappa = np.asarray(kappa, dtype=float)
    R = np.sqrt(1.0 + (L1 * kappa) ** 2)
    alpha = np.arctan2(-L1 * kappa, 1.0)
    asin_rhs = np.arcsin(np.clip(-L2C * kappa / R, -1.0, 1.0))
    return _wrap_to_pi(asin_rhs - alpha), _wrap_to_pi(np.pi - asin_rhs - alpha)


def _pick_continuous_branch(p1, p2, psi2_0):
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float); n = p1.size
    out = np.empty(n)
    if n == 0:
        return out
    fc = np.array([p1[0], p2[0]]); out[0] = fc[int(np.argmin(np.abs(_wrap_to_pi(fc - psi2_0))))]
    for i in range(1, n):
        c = np.array([p1[i], p2[i]]); out[i] = c[int(np.argmin(np.abs(_wrap_to_pi(c - out[i - 1]))))]
    return out


def generate_trajectory(x, y, vehicle, reverse=False, horizon=16):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    Ts = float(vehicle.dt); vx_abs = max(1e-6, abs(float(vehicle.xd)))
    L1 = float(vehicle.lf + vehicle.lr); L2C = np.finfo(float).eps; N = int(horizon)
    X0, Y0 = float(vehicle.trailer.x), float(vehicle.trailer.y)
    psi1_0 = float(vehicle.trailer.yaw); psi2_0 = float(_wrap_to_pi(vehicle.p - vehicle.trailer.yaw))
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    total_len = float(s[-1])
    i0 = int(np.argmin(np.hypot(x - X0, y - Y0))); s0 = float(s[i0])
    ds = max(1e-3, vx_abs * Ts)
    steps = np.arange(1, N + 1)
    s_targets = np.clip(s0 + (-ds if reverse else ds) * steps, 0.0, total_len)
    x_ref = np.interp(s_targets, s, x); y_ref = np.interp(s_targets, s, y)
    psi1_full, kappa_full = _heading_and_curvature(x, y)
    psi1_path = np.arctan2(np.interp(s_targets, s, np.sin(psi1_full)),
                           np.interp(s_targets, s, np.cos(psi1_full)))
    kappa = np.interp(s_targets, s, kappa_full)
    if reverse:
        psi1_path = _wrap_to_pi(psi1_path + np.pi); kappa = -kappa
    p1, p2 = _solve_psi2_batch(kappa, L1, L2C)
    psi2_ref = _pick_continuous_branch(p1, p2, psi2_0)
    traj = np.zeros((N + 1, 4))
    traj[0, :] = [X0, Y0, psi1_0, psi2_0]
    traj[1:, 0] = x_ref; traj[1:, 1] = y_ref; traj[1:, 2] = psi1_path; traj[1:, 3] = psi2_ref
    return traj


class _Trailer:
    __slots__ = ("x", "y", "yaw")


class _VehShim:
    __slots__ = ("dt", "xd", "p", "lf", "lr", "trailer")

    def __init__(self, dt, xd, p, tx, ty, tyaw, lf, lr):
        self.dt = dt; self.xd = xd; self.p = p; self.lf = lf; self.lr = lr
        self.trailer = _Trailer(); self.trailer.x = tx; self.trailer.y = ty; self.trailer.yaw = tyaw


class MPCController:
    """e2e_rl OSQP MPC driven per-env inside the batched env. Construct with the
    BatchedLaneFollowingEnv (``env.env`` of the SB3 adapter)."""

    def __init__(self, cfg, reverse, batched_env):
        self.env = batched_env; self.reverse = reverse
        self.dt = float(cfg.vehicle.dt)
        self.lf = float(cfg.vehicle.lf); self.lr = float(cfg.vehicle.lr)
        self.max_rate = float(np.deg2rad(cfg.action.steering_action_deg))
        Fwd, Rev = _load_mpc_classes()
        self.mpc = (Rev if reverse else Fwd)()
        self.mpc.L1 = self.lf + self.lr
        self.mpc.L2 = float(cfg.vehicle.trailer_length_m)
        self.mpc.Ts = self.dt
        self.mpc.umin = -float(cfg.vehicle.max_steer_angle_rad)
        self.mpc.umax = float(cfg.vehicle.max_steer_angle_rad)
        self.mpc.delta_umin = -self.max_rate * self.dt
        self.mpc.delta_umax = self.max_rate * self.dt
        self.N = int(self.mpc.N)
        self._prev = None

    def reset(self, n):
        self._prev = np.zeros(n)

    def predict(self, obs, dones=None, deterministic=True):
        v = self.env.vehicle
        x = to_numpy(v.x); p = to_numpy(v.p); xd = to_numpy(v.xd); s = to_numpy(v.s)
        tx = to_numpy(v.tx); ty = to_numpy(v.ty); tyaw = to_numpy(v.tyaw)
        xs = to_numpy(self.env.xs); ys = to_numpy(self.env.ys)
        n = len(x)
        if self._prev is None or len(self._prev) != n:
            self._prev = np.zeros(n)
        if dones is not None:
            self._prev = np.where(np.asarray(dones), 0.0, self._prev)
        out = np.zeros(n)
        for i in range(n):
            shim = _VehShim(self.dt, xd[i], p[i], tx[i], ty[i], tyaw[i], self.lf, self.lr)
            traj = generate_trajectory(xs, ys[i], shim, reverse=self.reverse, horizon=self.N)
            vx = xd[i] if abs(xd[i]) > 1e-3 else (-1.0 if self.reverse else 1.0)
            self.mpc.v = []  # clear warm-start so per-env solves don't cross-contaminate
            try:
                with contextlib.redirect_stdout(io.StringIO()):  # silence OSQP chatter
                    delta = float(self.mpc.solve(traj, state=(vx, self._prev[i])))
            except Exception:
                delta = float(s[i])
            self._prev[i] = delta
            out[i] = np.clip((delta - s[i]) / self.dt, -self.max_rate, self.max_rate)
        return out[:, None].astype(np.float32), None
