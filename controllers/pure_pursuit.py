"""
Pure Pursuit controller for tractor-trailer forward lane following.

Computes a desired steering angle from:
  - Feed-forward: Ackermann angle from path curvature  arctan(L·κ)
  - Feed-back:    proportional on lateral error e_y and heading error e_θ

Then converts to a steering-rate action via  δ̇ = (δ_des - δ_current) / dt.

Tunable parameters
------------------
k_ff    : curvature feed-forward gain (dimensionless, 1.0 = exact Ackermann)
k_y     : lateral error gain  (rad/m)
k_theta : heading error gain  (rad/rad)
speed   : target forward speed (m/s)
"""

import numpy as np


class PurePursuitController:
    def __init__(
        self,
        k_ff: float = 1.0,
        k_y: float = 0.15,
        k_theta: float = 0.50,
        speed: float = 5.0,
    ):
        self.k_ff = k_ff
        self.k_y = k_y
        self.k_theta = k_theta
        self.speed = speed

    def reset(self):
        """Stateless — kept for API consistency with PID / MPC."""
        pass

    def step(
        self,
        e_y: float,
        e_theta: float,
        kappa: float,
        wheelbase: float,
        current_steer: float,
        dt: float,
        max_steer_rate: float,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        e_y           : lateral cross-track error, tractor (m)
        e_theta       : heading error (rad)
        kappa         : signed path curvature at lookahead (1/m)
        wheelbase     : lf + lr (m)
        current_steer : current steering angle δ (rad)
        dt            : integration timestep (s)
        max_steer_rate: action bound (rad/s)

        Returns
        -------
        action : np.ndarray [steering_rate, speed]
        """
        delta_des = (
            self.k_ff * np.arctan(wheelbase * kappa)
            + self.k_y * e_y
            + self.k_theta * e_theta
        )
        steer_rate = float(np.clip(
            (delta_des - current_steer) / dt,
            -max_steer_rate,
            max_steer_rate,
        ))
        return np.array([steer_rate, self.speed], dtype=np.float32)
