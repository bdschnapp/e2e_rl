"""Single-track (bicycle) tractor dynamics with first-class actuator lag.

Ported from e2e_rl VehicleModels/vehicle_model.py. Two changes vs upstream:
  1. No ``from e2erl_utils.config import *`` — parameters come from a
     VehicleConfig (or an args dict), nothing global.
  2. Actuator lag is FIRST-CLASS (steer_tau_s / velocity_tau_s) instead of the
     ``_patch_actuator_lag`` monkeypatch. ``tau == 0`` short-circuits to the exact
     upstream behaviour (parity); ``tau > 0`` applies first-order lag, which the
     Smith predictor's delayed shadow relies on.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal


class StateSpaceVehicleModel:
    def __init__(self, vehicle_config=None, *, args: dict | None = None):
        # Accept either a VehicleConfig or a raw args dict (back-compat).
        if vehicle_config is not None:
            a = vehicle_config.vehicle_args
            self.rho = vehicle_config.rho
            self.kp, self.ki, self.kd = vehicle_config.kp, vehicle_config.ki, vehicle_config.kd
            self.steer_tau = float(vehicle_config.steer_tau_s)
            self.velocity_tau = float(vehicle_config.velocity_tau_s)
        elif args is not None:
            a = args
            self.rho = 1.225
            self.kp, self.ki, self.kd = 3.0, 0.3, 0.05
            self.steer_tau = 0.0
            self.velocity_tau = 0.0
        else:
            raise ValueError("StateSpaceVehicleModel needs vehicle_config or args")

        self.m = a["m"]; self.Iz = a["Iz"]
        self.Cf = a["Cf"]; self.Cr = a["Cr"]
        self.lf = a["lf"]; self.lr = a["lr"]
        self.Cd = a["Cd"]; self.A = a["A"]
        self.dt = a["dt"]

        # state
        self.xd = 1.0
        self.xdd = self.ydd = self.yd = 0.0
        self.x = self.y = 0.0
        self.p = self.pd = self.pdd = 0.0
        # controls
        self.s = 0.0
        self.Fx = 0.0
        # PID state
        self.i = 0.0
        self.e_prev = 0.0
        # actuator-lag targets
        self._s_target = 0.0
        self._xd_target = self.xd

    # ------------------------------------------------------------------
    def reset(self, xd, x=0.0, y=0.0, p=0.0):
        self.xdd = 0.0
        self.xd = xd
        self.x = x
        self.ydd = 0.0
        self.yd = 0.0
        self.y = y
        self.pdd = 0.0
        self.pd = 0.0
        self.p = p
        self.s = 0.0
        self.Fx = 0.0
        self.i = 0.0
        self.e_prev = 0.0
        self._s_target = 0.0
        self._xd_target = float(xd)

    def longitudinal_PID_controller(self, r):
        e = r - self.xd
        p = self.kp * e
        self.i += e * self.dt
        i = self.ki * self.i
        d = self.kd * (e - self.e_prev) / self.dt
        xdd = p + i + d
        self.Fx = self.m * xdd
        # NOTE: upstream e2e_rl never updates self.e_prev (it stays 0). Preserved
        # for bit-parity — do NOT add `self.e_prev = e` here.
        return self.Fx

    def _get_ct_matrices(self):
        s = self.s if self.xd > 0 else -self.s
        safe_xd = abs(self.xd) if abs(self.xd) > 0.01 else 0.01

        A11 = -1 * (self.Cd * self.rho * self.A * safe_xd) / (2 * self.m)
        A12 = (self.Cf * np.sin(s)) / (self.m * safe_xd)
        A13 = (self.Cf * np.sin(s) * self.lf) / (self.m * safe_xd)
        A22 = -1 * (self.Cr + (self.Cf * np.cos(s))) / (self.m * safe_xd)
        A23 = ((self.Cr * self.lr) - (self.Cf * self.lf * np.cos(s))) / (self.m * safe_xd)
        A32 = ((self.Cr * self.lr) - (self.Cf * self.lf * np.cos(s))) / (self.Iz * safe_xd)
        A33 = -1 * ((self.Cr * np.square(self.lr)) + (self.Cf * np.square(self.lf) * np.cos(s))) / (self.Iz * safe_xd)
        A = np.array([[A11, A12, A13], [0, A22, A23], [0, A32, A33]])

        B11 = (1 + np.cos(s)) / (2 * self.m)
        B12 = 0
        B21 = np.sin(s) / (2 * self.m)
        B22 = -1 * (self.Cf * np.cos(s) / self.m)
        B31 = (self.lf * np.sin(s)) / (2 * self.Iz)
        B32 = (self.lf * self.Cf * np.cos(s)) / self.Iz
        B = np.array([[B11, B12], [B21, B22], [B31, B32]])
        return A, B

    def _discretize(self, A, B):
        sys_d = signal.cont2discrete((A, B, np.eye(A.shape[0]), 0), self.dt, method="zoh")
        return sys_d[0], sys_d[1]

    def loop(self, action):
        # 1. integrate commanded steer-rate into the target, clamp to +/- pi/4
        self._s_target = float(np.clip(self._s_target + action[0] * self.dt, -np.pi / 4, np.pi / 4))
        # 2. actual tire angle: lag toward target (tau==0 => instantaneous = parity)
        if self.steer_tau > 1e-9:
            self.s = self.s + (self._s_target - self.s) * (self.dt / self.steer_tau)
        else:
            self.s = self._s_target
        # 3. velocity target lag before the PID (tau==0 => use command directly)
        if self.velocity_tau > 1e-9:
            self._xd_target = self._xd_target + (action[1] - self._xd_target) * (self.dt / self.velocity_tau)
        else:
            self._xd_target = float(action[1])

        self.longitudinal_PID_controller(self._xd_target)

        A_c, B_c = self._get_ct_matrices()
        A_d, B_d = self._discretize(A_c, B_c)
        state = np.array([self.xd, self.yd, self.pd])
        u = np.array([self.Fx, self.s])
        state_dot = A_d @ state + B_d @ u

        self.xd = state_dot[0]
        self.yd = state_dot[1]
        self.pd = state_dot[2]

        self.p += self.pd * self.dt
        self.x += self.xd * self.dt * np.cos(self.p) - self.yd * self.dt * np.sin(self.p)
        self.y += self.yd * self.dt * np.cos(self.p) + self.xd * self.dt * np.sin(self.p)

        return np.array([self.x, self.y, self.xd, self.yd, self.p, self.pd, self.s], dtype=float)
