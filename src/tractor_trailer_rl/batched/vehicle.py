"""Batched tractor + kinematic trailer dynamics (blocker #1 + batched core).

A vectorised re-implementation of ``vehicle.bicycle.StateSpaceVehicleModel`` +
``vehicle.tractor_trailer`` that steps N environments at once. Every scalar state
field becomes a shape-(N,) array on the active backend (numpy/cupy). The per-step
ZOH discretisation uses the batched matrix exponential (expm.py) instead of the
per-env scipy ``cont2discrete``.

Parity is preserved exactly against the scalar model, including the two upstream
quirks: the longitudinal PID never updates ``e_prev`` (stays 0), and the tyre
matrices use ``s_eff = sign(xd)*s`` while the input vector uses the raw ``s``.
"""

from __future__ import annotations

import numpy as np

from .backend import xp
from .expm import zoh_discretize


class BatchedTractorTrailer:
    def __init__(self, cfg, n_envs: int):
        v = cfg.vehicle
        self.n = int(n_envs)
        self.dt = float(v.dt)
        self.m, self.Iz = float(v.m), float(v.Iz)
        self.Cf, self.Cr = float(v.Cf), float(v.Cr)
        self.lf, self.lr = float(v.lf), float(v.lr)
        self.Cd, self.A, self.rho = float(v.Cd), float(v.A), float(v.rho)
        self.kp, self.ki, self.kd = float(v.kp), float(v.ki), float(v.kd)
        self.steer_tau = float(v.steer_tau_s)
        self.velocity_tau = float(v.velocity_tau_s)
        self.max_steer = float(getattr(v, "max_steer_angle_rad", np.pi / 4))
        self.L = float(v.trailer_length_m)  # kinematic trailer wheelbase
        self._alloc()

    def _alloc(self):
        z = lambda: xp.zeros(self.n, dtype=xp.float64)
        # tractor state
        self.x, self.y, self.p = z(), z(), z()
        self.xd, self.yd, self.pd = z(), z(), z()
        self.s = z()
        self.Fx = z()
        self.i = z()                 # PID integrator
        self._s_target = z()
        self._xd_target = z()
        # trailer state
        self.tx, self.ty = z(), z()
        self.tyaw, self.tyaw_rate = z(), z()

    # ------------------------------------------------------------------ reset
    def reset_where(self, mask, xd, x, y, p, trailer_yaw=None):
        """Reset the masked envs to the given (broadcastable) initial values.

        xd, x, y, p, trailer_yaw are shape-(N,) arrays (values for ALL envs; only
        the ``mask`` entries are written). trailer_yaw None => aligned to p.
        """
        mask = xp.asarray(mask)
        w = lambda cur, new: xp.where(mask, xp.asarray(new, dtype=xp.float64), cur)
        self.xd = w(self.xd, xd)
        self.x, self.y, self.p = w(self.x, x), w(self.y, y), w(self.p, p)
        for arr in ("yd", "pd", "s", "Fx", "i", "_s_target"):
            setattr(self, arr, xp.where(mask, 0.0, getattr(self, arr)))
        self._xd_target = w(self._xd_target, xd)
        tyaw = self.p if trailer_yaw is None else xp.asarray(trailer_yaw, dtype=xp.float64)
        hitch_x = self.x - self.lr * xp.cos(self.p)
        hitch_y = self.y - self.lr * xp.sin(self.p)
        self.tyaw = xp.where(mask, tyaw, self.tyaw)
        self.tx = xp.where(mask, hitch_x - self.L * xp.cos(self.tyaw), self.tx)
        self.ty = xp.where(mask, hitch_y - self.L * xp.sin(self.tyaw), self.ty)
        self.tyaw_rate = xp.where(mask, 0.0, self.tyaw_rate)

    def align_trailer(self, mask=None):
        """Rigidly align the trailer behind the tractor (tractor-only mode)."""
        hitch_x = self.x - self.lr * xp.cos(self.p)
        hitch_y = self.y - self.lr * xp.sin(self.p)
        tyaw = self.p
        tx = hitch_x - self.L * xp.cos(tyaw)
        ty = hitch_y - self.L * xp.sin(tyaw)
        if mask is None:
            self.tyaw, self.tx, self.ty = tyaw, tx, ty
            self.tyaw_rate = xp.zeros_like(self.tyaw_rate)
        else:
            self.tyaw = xp.where(mask, tyaw, self.tyaw)
            self.tx = xp.where(mask, tx, self.tx)
            self.ty = xp.where(mask, ty, self.ty)
            self.tyaw_rate = xp.where(mask, 0.0, self.tyaw_rate)

    # ------------------------------------------------------------------ dynamics
    def _ct_matrices(self):
        s_eff = xp.where(self.xd > 0, self.s, -self.s)
        safe_xd = xp.where(xp.abs(self.xd) > 0.01, xp.abs(self.xd), 0.01)
        cs, sn = xp.cos(s_eff), xp.sin(s_eff)
        m, Iz, Cf, Cr, lf, lr = self.m, self.Iz, self.Cf, self.Cr, self.lf, self.lr

        A11 = -(self.Cd * self.rho * self.A * safe_xd) / (2 * m)
        A12 = (Cf * sn) / (m * safe_xd)
        A13 = (Cf * sn * lf) / (m * safe_xd)
        A22 = -(Cr + Cf * cs) / (m * safe_xd)
        A23 = (Cr * lr - Cf * lf * cs) / (m * safe_xd)
        A32 = (Cr * lr - Cf * lf * cs) / (Iz * safe_xd)
        A33 = -(Cr * lr ** 2 + Cf * lf ** 2 * cs) / (Iz * safe_xd)
        zeros = xp.zeros_like(A11)
        A = xp.stack([
            xp.stack([A11, A12, A13], axis=1),
            xp.stack([zeros, A22, A23], axis=1),
            xp.stack([zeros, A32, A33], axis=1),
        ], axis=1)  # (N,3,3)

        B11 = (1 + cs) / (2 * m)
        B21 = sn / (2 * m)
        B22 = -(Cf * cs / m)
        B31 = (lf * sn) / (2 * Iz)
        B32 = (lf * Cf * cs) / Iz
        B = xp.stack([
            xp.stack([B11, zeros], axis=1),
            xp.stack([B21, B22], axis=1),
            xp.stack([B31, B32], axis=1),
        ], axis=1)  # (N,3,2)
        return A, B

    def step(self, steer_rate, velocity_cmd):
        """Advance all envs one dt. steer_rate, velocity_cmd are (N,) arrays."""
        dt = self.dt
        a0 = xp.asarray(steer_rate, dtype=xp.float64)
        a1 = xp.asarray(velocity_cmd, dtype=xp.float64)

        # 1. steer-rate -> target, clamp to the vehicle's max steer angle
        self._s_target = xp.clip(self._s_target + a0 * dt, -self.max_steer, self.max_steer)
        # 2. steer actuator lag (tau==0 => instantaneous, parity)
        if self.steer_tau > 1e-9:
            self.s = self.s + (self._s_target - self.s) * (dt / self.steer_tau)
        else:
            self.s = self._s_target
        # 3. velocity target lag
        if self.velocity_tau > 1e-9:
            self._xd_target = self._xd_target + (a1 - self._xd_target) * (dt / self.velocity_tau)
        else:
            self._xd_target = a1

        # 4. longitudinal PID (e_prev stays 0 for parity) using pre-update xd
        e = self._xd_target - self.xd
        self.i = self.i + e * dt
        xdd = self.kp * e + self.ki * self.i + self.kd * e / dt
        self.Fx = self.m * xdd

        # 5-6. continuous matrices + batched ZOH
        A, B = self._ct_matrices()
        Ad, Bd = zoh_discretize(A, B, dt)

        # 7. state update  state_dot = Ad@state + Bd@u
        state = xp.stack([self.xd, self.yd, self.pd], axis=1)[..., None]   # (N,3,1)
        u = xp.stack([self.Fx, self.s], axis=1)[..., None]                  # (N,2,1)
        sd = (xp.matmul(Ad, state) + xp.matmul(Bd, u))[..., 0]              # (N,3)
        self.xd, self.yd, self.pd = sd[:, 0], sd[:, 1], sd[:, 2]

        # 8. pose integration
        self.p = self.p + self.pd * dt
        self.x = self.x + self.xd * dt * xp.cos(self.p) - self.yd * dt * xp.sin(self.p)
        self.y = self.y + self.yd * dt * xp.cos(self.p) + self.xd * dt * xp.sin(self.p)

        # trailer kinematic update (tractor-only envs re-aligned by the env after)
        hitch_x = self.x - self.lr * xp.cos(self.p)
        hitch_y = self.y - self.lr * xp.sin(self.p)
        angle_diff = self.p - self.tyaw
        self.tyaw_rate = (self.xd / self.L) * xp.sin(angle_diff)
        self.tyaw = self.tyaw + self.tyaw_rate * dt
        self.tx = hitch_x - self.L * xp.cos(self.tyaw)
        self.ty = hitch_y - self.L * xp.sin(self.tyaw)
