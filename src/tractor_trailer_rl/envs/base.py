"""Config-driven lane-following env (collapses the e2e_rl 8-class hierarchy).

Direction (forward/reverse) and vehicle_kind (trailer/tractor-only) are Config
fields, not subclasses. Observation comes from the shared pure build_observation;
reward from the reward composer; action semantics from actions.modes. Termination
is pure numpy (no pygame): out-of-bounds, jackknife (|hitch|>pi/2), success
(x>0.85*W), and lane-corridor collision (vehicle footprint hits a blocked cell).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import Config, Direction, VehicleKind
from ..vehicle.tractor_trailer import StateSpaceTractorTrailer
from ..observation.builder import build_observation, EgoState, TrailerState
from ..observation.occupancy import build_occupancy_grid
from ..observation.layouts import observation_bounds
from ..reward.composer import running_reward, terminal_reward
from ..reward.proximity import proximity_penalty
from ..paths import generator as pathgen
from ..actions import modes as action_modes


class LaneFollowingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, render_mode=None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.vehicle = StateSpaceTractorTrailer(cfg.vehicle)

        self.action_space = action_modes.action_space(cfg)
        low, high = observation_bounds(cfg)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.xx = np.array([]); self.yy = np.array([])
        self.goal_pose = (0.0, 0.0, 0.0)
        self.path_kind = "straight"
        self.occ_grid = None; self.occ_meta = None
        self.episode_speed = cfg.action.fixed_speed_m_s
        self.success = False
        self._step_count = 0
        self.max_episode_steps = cfg.max_episode_steps

    # ------------------------------------------------------------------ helpers
    def _align_trailer(self):
        v = self.vehicle
        v.trailer.yaw = v.p
        v.trailer.yaw_rate = 0.0
        hitch_x = v.x - v.lr * np.cos(v.p)
        hitch_y = v.y - v.lr * np.sin(v.p)
        v.trailer.x = hitch_x - v.trailer.L * np.cos(v.p)
        v.trailer.y = hitch_y - v.trailer.L * np.sin(v.p)

    def _ego(self) -> EgoState:
        v = self.vehicle
        return EgoState(x=v.x, y=v.y, yaw=v.p, steer=v.s, xd=v.xd)

    def _trailer(self) -> TrailerState:
        t = self.vehicle.trailer
        return TrailerState(x=t.x, y=t.y, yaw=t.yaw)

    def _get_obs(self):
        return build_observation(self.xx, self.yy, self._ego(), self._trailer(),
                                 self.cfg, occ_grid=self.occ_grid,
                                 occ_meta=self.occ_meta).vector

    def _point_on_path(self, percent):
        percent = float(np.clip(percent, 0.0, 1.0))
        idx = int(percent * (len(self.xx) - 1))
        idx = int(np.clip(idx, 0, len(self.xx) - 2))
        x, y = self.xx[idx], self.yy[idx]
        yaw = float(np.arctan2(self.yy[idx + 1] - self.yy[idx],
                               self.xx[idx + 1] - self.xx[idx]))
        return x, y, yaw

    def _sample_speed(self):
        sr = self.cfg.speed_random
        if not sr.enabled:
            return self.cfg.action.fixed_speed_m_s
        if sr.explicit_min is not None and sr.explicit_max is not None:
            return float(self.np_random.uniform(sr.explicit_min, sr.explicit_max))
        lo, hi = sr.per_kind.get(self.path_kind, (1.0, 5.0))
        return float(self.np_random.uniform(lo, hi))

    # ------------------------------------------------------------------ bounds/term
    def _in_bounds(self):
        v = self.vehicle
        return (0 <= v.x <= self.cfg.world.width_m) and (0 <= v.y <= self.cfg.world.height_m)

    def _jackknife(self):
        h = abs((self.vehicle.p - self.vehicle.trailer.yaw + np.pi) % (2 * np.pi) - np.pi)
        return h > np.pi / 2

    def _collision(self):
        """Numpy lane-corridor collision: any sampled body point in a blocked/OOB
        cell. Replaces the pygame mask overlap; corridor-leaving terminates."""
        if self.occ_grid is None:
            return False
        m = self.occ_meta
        v = self.vehicle
        pts = []
        # tractor body samples
        for f in (-0.5, 0.0, 0.5):
            pts.append((v.x + f * self.cfg.vehicle.tractor_length_m * np.cos(v.p),
                        v.y + f * self.cfg.vehicle.tractor_length_m * np.sin(v.p)))
        # trailer body samples (axle to hitch)
        t = v.trailer
        for f in (0.0, 0.5, 1.0):
            pts.append((t.x + f * t.L * np.cos(t.yaw), t.y + f * t.L * np.sin(t.yaw)))
        for px, py in pts:
            gx = int((px - m.origin_x) / m.res_m)
            gy = int((py - m.origin_y) / m.res_m)
            if gx < 0 or gx >= m.width or gy < 0 or gy >= m.height:
                return True
            if self.occ_grid[gy, gx] == 100:
                return True
        return False

    def _terminated(self):
        if not self._in_bounds():
            return True
        if self._jackknife():
            return True
        if self._collision():
            return True
        env_len = self.cfg.world.width_m * 0.85
        if self.vehicle.x > env_len or self.vehicle.trailer.x > env_len:
            self.success = True
            return True
        return False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.xx, self.yy, self.goal_pose, self.path_kind = pathgen.generate_path(
            self.np_random, self.cfg.world, self.cfg.path)
        self.occ_grid, self.occ_meta = build_occupancy_grid(self.xx, self.yy, self.cfg.world)

        x, y, yaw = self._point_on_path(0.1)
        speed = self._sample_speed()
        # Recovery-curriculum spawn perturbation (defaults zero => parity).
        sp = self.cfg.spawn
        if sp.lateral_offset_m:
            lat = float(self.np_random.uniform(-sp.lateral_offset_m, sp.lateral_offset_m))
            x += -np.sin(yaw) * lat   # shift perpendicular to the path tangent
            y += np.cos(yaw) * lat
        heading = yaw + (np.pi if self.cfg.direction == Direction.REVERSE else 0.0)
        if sp.heading_offset_rad:
            heading += float(self.np_random.uniform(-sp.heading_offset_rad, sp.heading_offset_rad))
        self.vehicle.reset(speed, x=x, y=y, p=heading)
        self.episode_speed = -speed if self.cfg.direction == Direction.REVERSE else speed
        self.vehicle.fixed_speed_command = self.episode_speed
        if self.cfg.is_tractor_only:
            self._align_trailer()
        elif sp.hitch_offset_rad:
            # Open the hitch (tractor_yaw - trailer_yaw) while keeping the trailer
            # attached at the rear-axle hitch point.
            v = self.vehicle
            dh = float(self.np_random.uniform(-sp.hitch_offset_rad, sp.hitch_offset_rad))
            hitch_x = v.x - v.lr * np.cos(v.p)
            hitch_y = v.y - v.lr * np.sin(v.p)
            v.trailer.yaw = v.p - dh
            v.trailer.yaw_rate = 0.0
            v.trailer.x = hitch_x - v.trailer.L * np.cos(v.trailer.yaw)
            v.trailer.y = hitch_y - v.trailer.L * np.sin(v.trailer.yaw)
        self.success = False
        self._step_count = 0
        return self._get_obs(), self._info()

    def step(self, action):
        self._step_count += 1
        res = action_modes.resolve(action, self.cfg, self.episode_speed)

        if res.stop_now:
            self.vehicle.xd = 0.0
            return self._get_obs(), -abs(self.cfg.action.stop_penalty), True, False, self._info()

        self.vehicle.loop(np.array([res.steer_rate, res.velocity_cmd], dtype=float))
        if self.cfg.is_tractor_only:
            self._align_trailer()

        terminated = self._terminated()
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            reward = terminal_reward(self.cfg, self.success)
        else:
            obs_res = build_observation(self.xx, self.yy, self._ego(), self._trailer(),
                                        self.cfg, occ_grid=self.occ_grid, occ_meta=self.occ_meta)
            e = obs_res.errors
            prox = proximity_penalty(self.occ_grid, self.occ_meta,
                                     [(self.vehicle.x, self.vehicle.y),
                                      (self.vehicle.trailer.x, self.vehicle.trailer.y)], self.cfg)
            reward = running_reward(
                self.cfg, e_y=e["e_y"], e_psi=e["e_psi"],
                e_y_t=e.get("e_y_t", 0.0), e_psi_t=e.get("e_psi_t", 0.0),
                hitch=e.get("hitch", 0.0), xd=self.vehicle.xd, proximity=prox)

        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _info(self):
        v = self.vehicle
        return {
            "tractor_pos": (v.x, v.y), "tractor_yaw": v.p,
            "trailer_pos": (v.trailer.x, v.trailer.y), "trailer_yaw": v.trailer.yaw,
            "path_kind": self.path_kind, "success": self.success,
            "progress": self._progress_fraction(),
        }

    def _progress_fraction(self):
        env_len = self.cfg.world.width_m * 0.85
        start_x = self.cfg.path.x_start_m
        cur = max(self.vehicle.x, self.vehicle.trailer.x)
        return float(np.clip((cur - start_x) / (env_len - start_x), 0.0, 1.0))
