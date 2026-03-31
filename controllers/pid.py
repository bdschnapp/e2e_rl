"""
PID controller for tractor-trailer forward lane following.

Control law
-----------
  δ_des = k_ff · κ
        − Kp · e_y  −  Ki · ∫e_y dt  −  Kd · e_θ
        − Kp_t · e_y_t

Then converted to steering rate:  δ̇ = (δ_des − δ_current) / dt.

Design rationale
----------------
  e_y  (tractor CTE)  : primary position error — P+I to eliminate steady-state bias
  e_θ  (heading error): acts as derivative of e_y (ė_y ≈ v·sin(e_θ)) — damps overshoot
  e_y_t (trailer CTE) : corrects trailer lag; typically needs a gentler gain than tractor
  κ    (curvature FF) : Ackermann-style feed-forward for cornering; unity = exact linear FF

Anti-windup: integral is clamped to ±max_integral radians.

Tunable parameters: k_ff, Kp, Ki, Kd, Kp_t, speed, max_integral.

Also contains ReverseHitchPIDController for reverse driving.  See its docstring.
"""

import numpy as np


class PIDLaneController:
    def __init__(
        self,
        Kp: float = 0.30,
        Ki: float = 0.01,
        Kd: float = 0.80,
        Kp_t: float = 0.10,
        k_ff: float = 1.00,
        speed: float = 5.0,
        max_integral: float = 0.30,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kp_t = Kp_t
        self.k_ff = k_ff
        self.speed = speed
        self.max_integral = max_integral

        self._integral: float = 0.0

    def reset(self):
        """Must be called at the start of every episode to clear the integrator."""
        self._integral = 0.0

    def step(
        self,
        e_y: float,
        e_theta: float,
        e_y_t: float,
        kappa: float,
        current_steer: float,
        dt: float,
        max_steer_rate: float,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        e_y           : lateral cross-track error, tractor (m)
        e_theta       : heading error (rad)
        e_y_t         : lateral cross-track error, trailer (m)
        kappa         : signed path curvature (1/m)
        current_steer : current steering angle δ (rad)
        dt            : integration timestep (s)
        max_steer_rate: action bound (rad/s)

        Returns
        -------
        action : np.ndarray [steering_rate, speed]
        """
        # Clamped integration (anti-windup)
        self._integral = float(np.clip(
            self._integral + e_y * dt,
            -self.max_integral,
            self.max_integral,
        ))

        delta_des = (
            self.k_ff * kappa
            - self.Kp * e_y
            - self.Ki * self._integral
            - self.Kd * e_theta
            - self.Kp_t * e_y_t
        )
        steer_rate = float(np.clip(
            (delta_des - current_steer) / dt,
            -max_steer_rate,
            max_steer_rate,
        ))
        return np.array([steer_rate, self.speed], dtype=np.float32)


class ReverseHitchPIDController:
    """
    PID-style controller for reverse tractor-trailer driving.

    Control law
    -----------
      δ_des = k_hitch · ψ₂
            + k_ff    · κ
            + Kp      · e_y_t  +  Ki · ∫e_y_t dt  +  Kd · e_θ_t

    where ψ₂ = tractor_yaw − trailer_yaw (hitch angle).

    Design rationale
    ----------------
    Reverse driving is open-loop unstable: a non-zero hitch angle grows exponentially
    without active correction.  The k_hitch · ψ₂ term is the primary stabiliser — it
    steers the tractor in the direction of the hitch deviation, which returns the
    system toward straight alignment.

    Path tracking uses TRAILER errors (e_y_t, e_θ_t) because the trailer leads in
    reverse.  The integral term removes steady-state lateral offset; the Kd (heading)
    term damps oscillations.

    Anti-windup: integral is clamped to ±max_integral radians.

    Tunable parameters: k_hitch, k_ff, Kp, Ki, Kd, speed, max_integral.
    """

    def __init__(
        self,
        k_hitch: float = 0.5,
        Kp: float = 0.20,
        Ki: float = 0.01,
        Kd: float = 0.50,
        k_ff: float = 0.0,
        speed: float = 1.0,
        max_integral: float = 0.30,
    ):
        self.k_hitch = k_hitch
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.k_ff = k_ff
        self.speed = speed
        self.max_integral = max_integral

        self._integral: float = 0.0

    def reset(self):
        """Must be called at the start of every episode to clear the integrator."""
        self._integral = 0.0

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
        kappa         : signed path curvature (1/m)
        current_steer : current steering angle δ (rad)
        dt            : integration timestep (s)
        max_steer_rate: action bound (rad/s)

        Returns
        -------
        action : np.ndarray [steering_rate, speed]  (speed is negative for reverse)
        """
        self._integral = float(np.clip(
            self._integral + e_y_t * dt,
            -self.max_integral,
            self.max_integral,
        ))

        delta_des = (
            self.k_hitch * psi2
            + self.k_ff * kappa
            + self.Kp * e_y_t
            + self.Ki * self._integral
            + self.Kd * e_theta_t
        )
        steer_rate = float(np.clip(
            (delta_des - current_steer) / dt,
            -max_steer_rate,
            max_steer_rate,
        ))
        return np.array([steer_rate, -self.speed], dtype=np.float32)
