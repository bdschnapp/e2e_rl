"""Batched PID controllers (ported from e2e_rl controllers/pid.py).

Like the PP controllers these are functions of the observation errors, but with a
per-env integral term (anti-windup clamped) that must be reset on episode
boundaries — so ``predict`` takes an optional ``dones`` mask to zero the integrator
for envs that just reset. Obs layout (trailer): [s, hitch, e_y, e_psi, e_y_t,
e_psi_t, k1, k2, ...].

Forward: delta_des = -k_ff*kappa - Kp*e_y - Ki*∫e_y - Kd*e_psi + Kp_t*e_y_t
Reverse: delta_des =  k_hitch*hitch + k_ff*kappa + Kp*e_y_t + Ki*∫e_y_t + Kd*e_psi_t
then steer_rate = clip((delta_des - s)/dt, +/- max_rate).
"""

from __future__ import annotations

import numpy as np

# Shunt-truck tuned gains (this repo's tuner). The e2e_rl gains below don't transfer
# to the shunt geometry (0.00 completion; reverse k_hitch sign flips, as with PP).
# Benchmark result: PP >= PID here (fwd tied ~1.0 / PP tighter CTE; reverse PP 1.00 vs
# PID 0.91), so PP remains the guided-reward teacher; PID is a classical baseline.
FWD_GAINS = dict(Kp=0.194, Ki=0.053, Kd=1.57, Kp_t=0.178, k_ff=1.928, max_integral=0.30)
REV_GAINS = dict(k_hitch=-2.756, Kp=0.388, Ki=0.047, Kd=0.488, k_ff=2.112, max_integral=0.30)
FWD_GAINS_E2ERL = dict(Kp=0.023144086375283868, Ki=0.021720485228594016, Kd=0.25054255773306,
                       Kp_t=0.9542741814269738, k_ff=0.4682180246283856, max_integral=0.30)
REV_GAINS_E2ERL = dict(k_hitch=1.6405736866132936, Kp=0.16521211214797693, Ki=0.0036599697090570205,
                       Kd=0.34744427686266666, k_ff=1.4645172790409453, max_integral=0.30)


class BatchedPID:
    def __init__(self, cfg, reverse: bool, gains: dict | None = None):
        self.reverse = reverse
        self.dt = float(cfg.vehicle.dt)
        self.max_rate = float(np.deg2rad(cfg.action.steering_action_deg))
        self.g = gains or (REV_GAINS if reverse else FWD_GAINS)
        self.max_int = float(self.g.get("max_integral", 0.30))
        self._integral = None

    def reset(self, n):
        self._integral = np.zeros(n)

    def predict(self, obs, dones=None, deterministic=True):
        o = np.asarray(obs); n = o.shape[0]
        if self._integral is None or len(self._integral) != n:
            self._integral = np.zeros(n)
        if dones is not None:
            self._integral = np.where(np.asarray(dones), 0.0, self._integral)
        s = o[:, 0]; k1 = o[:, 6]; g = self.g
        if self.reverse:
            hitch, e_y_t, e_psi_t = o[:, 1], o[:, 4], o[:, 5]
            self._integral = np.clip(self._integral + e_y_t * self.dt, -self.max_int, self.max_int)
            delta_des = (g["k_hitch"] * hitch + g["k_ff"] * k1 + g["Kp"] * e_y_t
                         + g["Ki"] * self._integral + g["Kd"] * e_psi_t)
        else:
            e_y, e_psi, e_y_t = o[:, 2], o[:, 3], o[:, 4]
            self._integral = np.clip(self._integral + e_y * self.dt, -self.max_int, self.max_int)
            delta_des = (-g["k_ff"] * k1 - g["Kp"] * e_y - g["Ki"] * self._integral
                         - g["Kd"] * e_psi + g.get("Kp_t", 0.0) * e_y_t)
        steer_rate = np.clip((delta_des - s) / self.dt, -self.max_rate, self.max_rate)
        return steer_rate[:, None].astype(np.float32), None
