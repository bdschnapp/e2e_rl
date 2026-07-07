"""LQR path-tracking controller for the tractor-trailer (classical optimal-linear baseline).

Optimal linear feedback on the same kinematic error dynamics used by the MPC -- the
canonical optimal-control baseline (and the one common in the articulated-vehicle
literature / prior ECE682 work). Because it is a static gain u = -K x (+ curvature
feed-forward), it is fully vectorised over envs (no per-env QP) and fast.

State  x = [e_lat, e_head, gamma]  (lead-unit cross-track & heading error, hitch angle).
Input  u = delta (steering angle).  Direction-specific model (matches the env dynamics;
the env's tractor yaw response to steering does NOT flip sign in reverse -- it uses |xd|
and s_eff=-s -- so reverse uses |V| on the steering input):

FORWARD (track tractor):
    e_y_dot   = V*e_psi ;  e_psi_dot = (V/L1)*delta - V*kappa ;  gamma_dot = (V/L1)*delta - (V/L2)*gamma
REVERSE (track trailer, hitch open-loop unstable -> LQR stabilises it):
    e_yt_dot  = V*e_psit; e_psit_dot = (V/L2)*gamma - V*kappa ;  gamma_dot = (|V|/L1)*delta - (V/L2)*gamma

Steady-state curvature feed-forward: delta_ff = sign(V)*L1*kappa (holds e_head, gamma on a
constant-curvature arc). Command u = delta_ff - K (x - x_ref(kappa)); converted to steer-rate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

from .backend import to_numpy

_S, _HITCH, _EY, _EPSI, _EYT, _EPSIT, _K1 = 0, 1, 2, 3, 4, 5, 6

# shunt-tuned state/input weights [e_lat, e_head, gamma], R on steering. Reverse needs
# a heavy control penalty (R) + strong hitch weight (q_gamma) to stay damped on the
# open-loop-unstable plant; forward is well-behaved at modest weights.
FWD_QR = dict(q_lat=1.0, q_head=2.0, q_gamma=1.0, R=1.0)
REV_QR = dict(q_lat=0.054, q_head=2.87, q_gamma=22.6, R=83.1)


class BatchedLQR:
    def __init__(self, cfg, reverse: bool, qr: dict | None = None):
        self.reverse = reverse
        self.dt = float(cfg.vehicle.dt)
        self.L1 = float(cfg.vehicle.lf + cfg.vehicle.lr)
        self.L2 = float(cfg.vehicle.trailer_length_m)
        self.delta_max = float(cfg.vehicle.max_steer_angle_rad)
        self.rate_max = float(np.deg2rad(cfg.action.steering_action_deg))
        v = float(cfg.action.fixed_speed_m_s)
        self.V = -v if reverse else v
        q = qr or (REV_QR if reverse else FWD_QR)
        V, L1, L2 = self.V, self.L1, self.L2
        if reverse:
            # Trailer LEADS and progresses in +s (toward goal) at along-path speed |V|,
            # so path-following kinematics (e_yt_dot = |V|*e_psit) use |V|, while the
            # yaw dynamics (tyaw_rate = (V/L2) sin gamma) keep the signed V. The tractor
            # yaw response to steering does not flip in reverse -> |V|/L1 on the input.
            Ac = np.array([[0.0, abs(V), 0.0], [0.0, 0.0, V / L2], [0.0, 0.0, -V / L2]])
            Bc = np.array([[0.0], [0.0], [abs(V) / L1]])
        else:
            Ac = np.array([[0.0, V, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -V / L2]])
            Bc = np.array([[0.0], [V / L1], [V / L1]])
        Ad, Bd, *_ = cont2discrete((Ac, Bc, np.eye(3), 0.0), self.dt, method="zoh")
        Q = np.diag([q["q_lat"], q["q_head"], q["q_gamma"]]).astype(float)
        R = np.array([[float(q["R"])]])
        P = solve_discrete_are(Ad, Bd, Q, R)
        self.K = (np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)).ravel()   # (3,)

    def reset(self, n):
        pass

    def predict(self, obs, dones=None, deterministic=True):
        o = np.asarray(to_numpy(obs))
        s = o[:, _S]; kappa = o[:, _K1]
        gamma = (o[:, _HITCH] + np.pi) % (2 * np.pi) - np.pi
        if self.reverse:
            e_lat, e_head = o[:, _EYT], o[:, _EPSIT]
        else:
            e_lat, e_head = o[:, _EY], o[:, _EPSI]
        # constant-curvature-arc feed-forward (holds e_head, gamma): steady-state
        # delta_ff = L1*kappa (both directions), hitch gamma_ref = sign(V)*L2*kappa.
        sgn = 1.0 if self.V >= 0 else -1.0
        delta_ff = self.L1 * kappa
        gamma_ref = sgn * self.L2 * kappa
        # x - x_ref (only gamma has a nonzero reference; e_lat,e_head tracked to 0)
        dx0 = e_lat
        dx1 = e_head
        dx2 = gamma - gamma_ref
        delta = delta_ff - (self.K[0] * dx0 + self.K[1] * dx1 + self.K[2] * dx2)
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        steer_rate = np.clip((delta - s) / self.dt, -self.rate_max, self.rate_max)
        return steer_rate[:, None].astype(np.float32), None
