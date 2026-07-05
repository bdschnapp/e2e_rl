"""Batched pure-pursuit controllers (ported from e2e_rl controllers/pure_pursuit.py).

Both control laws are fixed proportional functions of quantities already present in
the observation vector (trailer layout: [s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1,
k2, ...lidar]), so a batched controller is just a function of obs -> steer-rate. It
exposes a ``predict(obs)`` method matching the SB3 policy API, so it drops straight
into the existing ``evaluate()`` loop:

  * as the GUIDED reward's reference controller, and
  * as an ORACLE to verify a path set is feasible (a tuned PP should complete gentle
    paths at ~1.0 both forward and reverse).

Forward:  delta_des = -(k_ff*atan(L*kappa) + k_y*e_y + k_theta*e_psi)
Reverse:  delta_des =  k_hitch*hitch + k_ff*kappa + k_y*e_y_t + k_theta*e_psi_t
          (trailer errors + hitch stabiliser, since the trailer leads in reverse)
then steer_rate = clip((delta_des - s)/dt, +/- max_rate).
"""

from __future__ import annotations

import numpy as np

# Forward gains from e2e_rl (controllers/tuned_params.json) — verified to complete the
# shunt-truck gentle paths at 1.00, so they transfer as-is.
FWD_GAINS = dict(k_ff=0.7023064120291692, k_y=0.17023881008095487, k_theta=2.778463838892346)
# Reverse: the e2e_rl gains (k_hitch=+0.80, k_ff=-2.0) DESTABILISE the shunt truck
# (jackknifes in ~20 steps) — the hitch-stabilisation sign flips at this geometry.
# Re-tuned for the shunt truck (2.95 m wheelbase, 12.5 m trailer): completes gentle
# reverse at 0.995. Verifies the paths are feasible; the reverse RL failures were
# training instability, not path difficulty.
REV_GAINS = dict(k_hitch=-2.591, k_y=0.087, k_theta=1.663, k_ff=0.711)
REV_GAINS_E2ERL = dict(k_hitch=0.7981827582320544, k_y=1.246169343066824,
                       k_theta=1.846962929234278, k_ff=-1.995183948132352)  # old scale


class BatchedPurePursuit:
    def __init__(self, cfg, reverse: bool, gains: dict | None = None):
        self.reverse = reverse
        self.dt = float(cfg.vehicle.dt)
        self.L = float(cfg.vehicle.lf + cfg.vehicle.lr)
        self.max_rate = float(np.deg2rad(cfg.action.steering_action_deg))
        self.g = gains or (REV_GAINS if reverse else FWD_GAINS)

    def predict(self, obs, deterministic=True):
        o = np.asarray(obs)
        s = o[:, 0]; k1 = o[:, 6]
        if self.reverse:
            hitch, e_y_t, e_psi_t = o[:, 1], o[:, 4], o[:, 5]
            delta_des = (self.g["k_hitch"] * hitch + self.g["k_ff"] * k1
                         + self.g["k_y"] * e_y_t + self.g["k_theta"] * e_psi_t)
        else:
            e_y, e_psi = o[:, 2], o[:, 3]
            delta_des = -(self.g["k_ff"] * np.arctan(self.L * k1)
                          + self.g["k_y"] * e_y + self.g["k_theta"] * e_psi)
        steer_rate = np.clip((delta_des - s) / self.dt, -self.max_rate, self.max_rate)
        return steer_rate[:, None].astype(np.float32), None
