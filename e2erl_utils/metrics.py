"""
EpisodeMetricsLogger — per-step metric collection and summary statistics
for benchmarking the tractor-trailer RL controller against classical baselines.

Usage
-----
logger = EpisodeMetricsLogger()
obs, _ = env.reset()
while not done:
    t0 = time.perf_counter()
    action = controller(obs)
    t1 = time.perf_counter()
    obs, reward, terminated, truncated, info = env.step(action)
    logger.log_step(env, action, reward, inference_time_s=t1 - t0)
    done = terminated or truncated

summary = logger.compute_summary(terminated=terminated, truncated=truncated)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EpisodeMetricsLogger:
    """Collect per-step metrics during one episode and compute summaries."""

    # Per-step lists (populated by log_step)
    _cte_tractor: list = field(default_factory=list)
    _cte_trailer: list = field(default_factory=list)
    _heading_err_tractor: list = field(default_factory=list)
    _heading_err_trailer: list = field(default_factory=list)
    _hitch_angle: list = field(default_factory=list)
    _steering_rate: list = field(default_factory=list)   # action[0]  (rad/s)
    _target_speed: list = field(default_factory=list)    # action[1]  (m/s)
    _actual_speed: list = field(default_factory=list)    # vehicle.xd (m/s)
    _inference_time_s: list = field(default_factory=list)
    _reward: list = field(default_factory=list)
    # Obstacle-task extras (populated when lidar is available)
    _min_lidar: list = field(default_factory=list)
    _min_clearance_m: list = field(default_factory=list)

    def reset(self):
        """Clear all stored data for a new episode."""
        for lst in [
            self._cte_tractor, self._cte_trailer,
            self._heading_err_tractor, self._heading_err_trailer,
            self._hitch_angle, self._steering_rate, self._target_speed,
            self._actual_speed, self._inference_time_s, self._reward,
            self._min_lidar, self._min_clearance_m,
        ]:
            lst.clear()

    def log_step(self, env, action, reward: float, inference_time_s: float = 0.0):
        """
        Record one environment step.

        Parameters
        ----------
        env : a LineFollowing / ObstacleAvoidance environment instance
        action : array-like of shape (1,) or (2,)
            Steering-only actions are accepted for fixed-speed environments.
        reward : scalar reward returned by env.step()
        inference_time_s : wall-clock time taken by the controller (seconds)
        """
        # Path-tracking errors
        e_tractor, e_theta_tractor = env.get_vehicle_errors()
        e_trailer, e_theta_trailer = env.get_trailer_errors()

        # Hitch / articulation angle
        hitch = float(env.vehicle.p - env.vehicle.trailer.yaw)

        self._cte_tractor.append(float(e_tractor))
        self._cte_trailer.append(float(e_trailer))
        self._heading_err_tractor.append(float(e_theta_tractor))
        self._heading_err_trailer.append(float(e_theta_trailer))
        self._hitch_angle.append(hitch)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._steering_rate.append(float(action_arr[0]))
        if action_arr.size >= 2:
            target_speed = float(action_arr[1])
        else:
            target_speed = float(getattr(env, "fixed_speed_command", env.vehicle.xd))
        self._target_speed.append(target_speed)
        self._actual_speed.append(float(env.vehicle.xd))
        self._inference_time_s.append(float(inference_time_s))
        self._reward.append(float(reward))

        # Lidar (obstacle environments expose 'observation' with lidar suffix)
        obs = getattr(env, "observation", None)
        if obs is not None and len(obs) > 8:
            lidar_readings = obs[8:]  # indices 8-31 are normalised lidar distances
            self._min_lidar.append(float(np.min(lidar_readings)))

        # Obstacle clearance — requires env.obstacles attribute
        obstacles = getattr(env, "obstacles", None)
        if obstacles:
            vx, vy = env.vehicle.x, env.vehicle.y
            tx, ty = env.vehicle.trailer.x, env.vehicle.trailer.y
            min_d = math.inf
            for ox, oy, r in obstacles:
                d_tractor = math.hypot(vx - ox, vy - oy) - r
                d_trailer = math.hypot(tx - ox, ty - oy) - r
                min_d = min(min_d, d_tractor, d_trailer)
            self._min_clearance_m.append(float(min_d))

    def compute_summary(
        self,
        terminated: bool = False,
        truncated: bool = False,
        completed: Optional[bool] = None,
    ) -> dict:
        """
        Return a flat dict of summary statistics for the episode.

        Episode outcome keys
        --------------------
        completed        : bool — episode ended with success (not collision/jackknife/timeout)
        jackknifed       : bool — episode terminated due to |hitch| > ~90°
        collided         : bool — terminated without jackknife (i.e. collision / OOB)
        truncated        : bool — episode hit max_episode_steps

        Path-tracking metrics
        ---------------------
        mean_abs_cte_tractor, mean_abs_cte_trailer
        rms_cte_tractor, rms_cte_trailer
        max_abs_cte_tractor, max_abs_cte_trailer
        mean_abs_heading_err_tractor, mean_abs_heading_err_trailer

        Stability metrics
        -----------------
        mean_abs_hitch_angle, max_abs_hitch_angle, hitch_angle_variance

        Smoothness / efficiency metrics
        --------------------------------
        steering_rate_rms, steering_rate_variance
        speed_variance
        episode_length
        total_reward

        Computational metrics
        ---------------------
        mean_inference_ms, p95_inference_ms, max_inference_ms

        Obstacle-specific metrics (NaN when not applicable)
        ---------------------------------------------------
        mean_min_lidar, min_clearance_m
        """
        n = len(self._cte_tractor)
        if n == 0:
            return {"episode_length": 0}

        cte_t = np.array(self._cte_tractor)
        cte_tr = np.array(self._cte_trailer)
        he_t = np.array(self._heading_err_tractor)
        he_tr = np.array(self._heading_err_trailer)
        hitch = np.array(self._hitch_angle)
        sr = np.array(self._steering_rate)
        spd = np.array(self._actual_speed)
        inf_ms = np.array(self._inference_time_s) * 1000.0

        # Episode outcome
        jackknifed = terminated and (np.max(np.abs(hitch)) > math.pi / 2 * 0.95)
        if completed is None:
            completed_flag = terminated and not jackknifed and not truncated
        else:
            completed_flag = bool(completed)
        collided = terminated and not jackknifed and not truncated and not completed_flag

        summary = {
            # Outcome
            "completed": bool(completed_flag),
            "jackknifed": bool(jackknifed),
            "collided": bool(collided),
            "truncated": bool(truncated),
            "episode_length": n,
            "total_reward": float(np.sum(self._reward)),
            # CTE — tractor
            "mean_abs_cte_tractor": float(np.mean(np.abs(cte_t))),
            "rms_cte_tractor": float(np.sqrt(np.mean(cte_t ** 2))),
            "max_abs_cte_tractor": float(np.max(np.abs(cte_t))),
            # CTE — trailer
            "mean_abs_cte_trailer": float(np.mean(np.abs(cte_tr))),
            "rms_cte_trailer": float(np.sqrt(np.mean(cte_tr ** 2))),
            "max_abs_cte_trailer": float(np.max(np.abs(cte_tr))),
            # Heading errors
            "mean_abs_heading_err_tractor": float(np.mean(np.abs(he_t))),
            "mean_abs_heading_err_trailer": float(np.mean(np.abs(he_tr))),
            # Hitch / articulation
            "mean_abs_hitch_angle": float(np.mean(np.abs(hitch))),
            "max_abs_hitch_angle": float(np.max(np.abs(hitch))),
            "hitch_angle_variance": float(np.var(hitch)),
            # Smoothness
            "steering_rate_rms": float(np.sqrt(np.mean(sr ** 2))),
            "steering_rate_variance": float(np.var(sr)),
            "speed_variance": float(np.var(spd)),
            # Timing
            "mean_inference_ms": float(np.mean(inf_ms)),
            "p95_inference_ms": float(np.percentile(inf_ms, 95)),
            "max_inference_ms": float(np.max(inf_ms)),
            # Obstacle extras (NaN if not collected)
            "mean_min_lidar": float(np.mean(self._min_lidar)) if self._min_lidar else float("nan"),
            "min_clearance_m": float(np.min(self._min_clearance_m)) if self._min_clearance_m else float("nan"),
        }
        return summary
