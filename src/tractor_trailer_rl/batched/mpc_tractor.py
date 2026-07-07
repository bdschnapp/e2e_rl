"""Kinematic LTV MPC for tractor-trailer path tracking (classical baseline).

The inherited e2e_rl OSQP MPC servos the trailer pose with the *unstable* reverse hitch
dynamics inside the QP, so it oscillates/jackknifes for the long shunt-truck trailer
(L2/L1 ~ 4.2) -- forward it fails on any curve, reverse it fails outright. This
controller uses the textbook path-tracking formulation, tailored per direction:

FORWARD -- the tractor leads, so track the TRACTOR path errors and let the trailer
follow (the MPC analogue of pure pursuit):
    state x = [e_y, e_psi, gamma]           (tractor cross-track, heading, hitch)
    e_y_dot    = V * e_psi
    e_psi_dot  = (V/L1) * delta - V * kappa
    gamma_dot  = (V/L1) * delta - (V/L2) * gamma
    input = delta (steering angle, returned directly).

REVERSE -- the trailer leads and the hitch is open-loop unstable, so use the
flatness / virtual-input reformulation from the working Autoware trailer_ltv_mpc:
the QP plans the TRAILER as a stable bicycle with a *virtual* trailer steer delta_T,
with the hitch entering only through the inner-loop-closed (stable) dynamics; an
analytic inner map realises delta_T on the tractor and provides the anti-jackknife
counter-steer.
    state x = [e_yt, e_psit, gamma]         (trailer cross-track, heading, hitch)
    e_yt_dot   = V * e_psit
    e_psit_dot = (V/L2) * gamma - V * kappa
    gamma_dot  = K * (delta_T - gamma)      (inner loop closed)
    inner map:  delta_f = arctan( (L1/L2) sin(gamma) + (L1*K/V) * (delta_T - gamma) )
The 1/V factor flips the feedback sign in reverse -> the anti-jackknife counter-steer.

Both then convert to a steer-rate for the env. Per-env OSQP (MPC is not batched);
prediction matrices precomputed once (constant speed).
"""

from __future__ import annotations

import io
import contextlib
import numpy as np
from scipy.signal import cont2discrete
from scipy import sparse
import osqp

from .backend import to_numpy

# obs layout (trailer tasks): [s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2, ...]
_S, _HITCH, _EY, _EPSI, _EYT, _EPSIT, _K1 = 0, 1, 2, 3, 4, 5, 6

FWD_WEIGHTS = dict(q_lat=1.0, q_head=6.0, q_gamma=3.0, R=0.5, Rd=6.0, N=20)
# reverse (virtual-trailer cascade): heavy input-rate penalty (Rd) so it doesn't chase
# the cross-track measurement ripple, strong hitch weight, inner gain K ~ 4.
REV_WEIGHTS = dict(q_lat=0.12, q_head=0.73, q_gamma=3.97, R=12.9, Rd=54.3, N=30, K=4.4)


class KinematicTractorMPC:
    def __init__(self, cfg, reverse: bool, weights: dict | None = None):
        self.reverse = reverse
        self.dt = float(cfg.vehicle.dt)
        self.L1 = float(cfg.vehicle.lf + cfg.vehicle.lr)   # tractor wheelbase
        self.L2 = float(cfg.vehicle.trailer_length_m)      # trailer length (hitch@rear axle)
        self.delta_max = float(cfg.vehicle.max_steer_angle_rad)
        self.rate_max = float(np.deg2rad(cfg.action.steering_action_deg))
        v = float(cfg.action.fixed_speed_m_s)
        self.V = -v if reverse else v                      # signed speed
        w = weights or (REV_WEIGHTS if reverse else FWD_WEIGHTS)
        self.N = int(w["N"]); self.K = float(w.get("K", 3.0))
        self.Q = np.diag([w["q_lat"], w["q_head"], w["q_gamma"]]).astype(float)
        self.R = float(w["R"]); self.Rd = float(w["Rd"])
        self._prev = None
        self._build_prediction()

    def _build_prediction(self):
        V, L1, L2, dt, N, K = self.V, self.L1, self.L2, self.dt, self.N, self.K
        if self.reverse:
            # virtual-trailer cascade: input = delta_T (virtual trailer steer). Trailer
            # leads in +s at along-path speed |V| (e_yt_dot = |V|*e_psit); yaw dynamics
            # keep signed V (tyaw_rate = (V/L2) gamma).
            Ac = np.array([[0.0, abs(V), 0.0], [0.0, 0.0, V / L2], [0.0, 0.0, -K]])
            Bc = np.array([[0.0], [0.0], [K]])
        else:
            # tractor tracking: input = delta (steering angle) drives heading + hitch
            Ac = np.array([[0.0, V, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -V / L2]])
            Bc = np.array([[0.0], [V / L1], [V / L1]])
        Ec = np.array([[0.0], [-abs(V)], [0.0]])           # curvature disturbance (|V| along path)
        Ad, BEd, *_ = cont2discrete((Ac, np.hstack([Bc, Ec]), np.eye(3), 0.0), dt, method="zoh")
        Bd = BEd[:, :1]; Ed = BEd[:, 1:]
        nx, nu = 3, 1
        Apow = [np.eye(nx)]
        for _ in range(N):
            Apow.append(Apow[-1] @ Ad)
        Ax = np.zeros((N * nx, nx)); Bu = np.zeros((N * nx, N * nu)); Ew = np.zeros((N * nx, N))
        for i in range(1, N + 1):
            Ax[(i - 1) * nx:i * nx] = Apow[i]
            for j in range(1, i + 1):
                blk = Apow[i - j]
                Bu[(i - 1) * nx:i * nx, (j - 1) * nu:j * nu] = blk @ Bd
                Ew[(i - 1) * nx:i * nx, (j - 1):j] = blk @ Ed
        Qbar = np.kron(np.eye(N), self.Q)
        T = np.eye(N) - np.eye(N, k=-1)
        H = 2.0 * (Bu.T @ Qbar @ Bu + self.R * np.eye(N) + self.Rd * (T.T @ T))
        self.H = 0.5 * (H + H.T)
        self.Ax, self.Bu, self.Ew = Ax, Bu, Ew
        self._BuTQ = Bu.T @ Qbar
        self._Ac_con = sparse.csc_matrix(np.vstack([T, np.eye(N)]))
        self._Psparse = sparse.csc_matrix(self.H)

    def reset(self, n):
        self._prev = np.zeros(n)

    def _solve(self, x0, kappa_prev, prev_u):
        N = self.N
        lin = self.Ax @ x0 + self.Ew @ np.full(N, kappa_prev)
        f = 2.0 * (self._BuTQ @ lin)
        f[0] += -2.0 * self.Rd * prev_u
        rate = self.rate_max * self.dt
        lbA = np.concatenate([[prev_u - rate], np.full(N - 1, -rate)])
        ubA = np.concatenate([[prev_u + rate], np.full(N - 1, rate)])
        lb = np.concatenate([lbA, np.full(N, -self.delta_max)])
        ub = np.concatenate([ubA, np.full(N, self.delta_max)])
        prob = osqp.OSQP()
        with contextlib.redirect_stdout(io.StringIO()):
            prob.setup(P=self._Psparse, q=np.ascontiguousarray(f), A=self._Ac_con,
                       l=lb, u=ub, verbose=False, polish=True, max_iter=2000,
                       eps_abs=1e-5, eps_rel=1e-5)
            res = prob.solve()
        if res.x is None or res.info.status_val not in (1, 2):
            return prev_u
        return float(res.x[0])

    def _inner_map(self, delta_T, gamma):
        # Feedback-linearise the hitch to gamma_dot = K(delta_T - gamma). The env's
        # tractor yaw response to steering does NOT flip in reverse (dynamic model uses
        # |xd| and s_eff=-s), so the FF term carries sign(V) and the feedback uses |V|.
        sgn = 1.0 if self.V >= 0 else -1.0
        tan_df = (self.L1 / self.L2) * sgn * np.sin(gamma) + (self.L1 * self.K / abs(self.V)) * (delta_T - gamma)
        return float(np.clip(np.arctan(tan_df), -self.delta_max, self.delta_max))

    def predict(self, obs, dones=None, deterministic=True):
        o = np.asarray(to_numpy(obs)); n = o.shape[0]
        if self._prev is None or len(self._prev) != n:
            self._prev = np.zeros(n)
        if dones is not None:
            self._prev = np.where(np.asarray(dones), 0.0, self._prev)
        s = o[:, _S]; kappa = o[:, _K1]
        gamma = (o[:, _HITCH] + np.pi) % (2 * np.pi) - np.pi
        if self.reverse:
            e_lat, e_head = o[:, _EYT], o[:, _EPSIT]        # trailer leads -> track trailer
        else:
            e_lat, e_head = o[:, _EY], o[:, _EPSI]          # tractor leads -> track tractor
        out = np.zeros(n)
        for i in range(n):
            u = self._solve(np.array([e_lat[i], e_head[i], gamma[i]]), float(kappa[i]), float(self._prev[i]))
            self._prev[i] = u
            delta_f = self._inner_map(u, float(gamma[i])) if self.reverse else u
            out[i] = np.clip((delta_f - s[i]) / self.dt, -self.rate_max, self.rate_max)
        return out[:, None].astype(np.float32), None
