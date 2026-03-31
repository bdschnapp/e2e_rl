"""
Pure Pursuit controller for tractor-trailer forward lane following.

Computes a desired steering angle from:
  - Feed-forward: Ackermann angle from path curvature  arctan(L·κ)
  - Feed-back:    proportional on lateral error e_y and heading error e_θ

Then converts to a steering-rate action via  δ̇ = (δ_des - δ_current) / dt.

Also contains ReverseHitchPurePursuitController for reverse driving, which adds an
explicit hitch-angle stabilisation term needed to counteract the open-loop instability
of the tractor-trailer in reverse.

Tunable parameters (forward)
-----------------------------
k_ff    : curvature feed-forward gain (dimensionless, 1.0 = exact Ackermann)
k_y     : lateral error gain  (rad/m)
k_theta : heading error gain  (rad/rad)
speed   : target forward speed (m/s)

Tunable parameters (reverse)
-----------------------------
k_hitch : hitch-angle proportional gain (rad/rad) — primary stabiliser
k_y     : trailer lateral error gain (rad/m)
k_theta : trailer heading error gain (rad/rad)
k_ff    : signed curvature feed-forward gain
speed   : target reverse speed magnitude (m/s); action returns -speed
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
        delta_des = -(
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


class ReverseHitchPurePursuitController:
    """
    Proportional controller for reverse tractor-trailer driving.

    Control law
    -----------
      δ_des = k_hitch · ψ₂
            + k_ff    · κ
            + k_y     · e_y_t
            + k_theta · e_θ_t

    where ψ₂ = tractor_yaw − trailer_yaw (hitch angle).

    Stabilisation intuition
    -----------------------
    In reverse the system is open-loop unstable: an uncorrected hitch angle grows
    exponentially.  The k_hitch term steers the tractor in the direction of the
    hitch angle deviation, which pushes the trailer back toward alignment.
    (This is the opposite of forward driving intuition.)

    The path-tracking terms (k_y, k_theta, k_ff) use TRAILER errors because the
    trailer is the "leading" unit during reverse.

    The sign of k_ff may be positive or negative depending on your curvature
    convention; the tuner will find the correct value.
    """

    def __init__(
        self,
        k_hitch: float = 0.5,
        k_y: float = 0.15,
        k_theta: float = 0.50,
        k_ff: float = 0.0,
        speed: float = 1.0,
    ):
        self.k_hitch = k_hitch
        self.k_y = k_y
        self.k_theta = k_theta
        self.k_ff = k_ff
        self.speed = speed

    def reset(self):
        """Stateless — kept for API consistency."""
        pass

    def step(
        self,
        psi2: float,
        e_y_t: float,
        e_theta_t: float,
        kappa: float,
        current_steer: float,
        dt: float,
        max_steer_rate: float,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        psi2          : hitch angle = tractor_yaw − trailer_yaw (rad)
        e_y_t         : lateral cross-track error, trailer (m)
        e_theta_t     : heading error, trailer (rad)
        kappa         : signed path curvature at lookahead (1/m)
        current_steer : current steering angle δ (rad)
        dt            : integration timestep (s)
        max_steer_rate: action bound (rad/s)

        Returns
        -------
        action : np.ndarray [steering_rate, speed]  (speed is negative for reverse)
        """
        delta_des = (
            self.k_hitch * psi2
            + self.k_ff * kappa
            + self.k_y * e_y_t
            + self.k_theta * e_theta_t
        )
        steer_rate = float(np.clip(
            (delta_des - current_steer) / dt,
            -max_steer_rate,
            max_steer_rate,
        ))
        return np.array([steer_rate, -self.speed], dtype=np.float32)
