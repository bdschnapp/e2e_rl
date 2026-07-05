"""Batched, auto-resetting lane-following env (blocker #5).

Runs N envs with a leading (N,) axis. Termination (out-of-bounds, jackknife,
corridor collision, success) and the stop-signal action become boolean masks
instead of scalar ``if`` returns, and done envs are reset in place (internal
auto-reset — the scalar env relies on SB3's VecEnv for this).

Path generation runs on the host (numpy) for the done envs only and is written
into the shared ``xs`` (P,) / per-env ``ys`` (N,P) arrays; it happens at reset,
not on the per-step hot path. Everything else (dynamics, geometry, reward, lidar,
collision) is vectorised on the active backend.

API mirrors a Gymnasium vector env:
  reset(seed)      -> obs (N,D)
  step(actions)    -> obs (N,D), reward (N,), terminated (N,), truncated (N,), info
The single-env observation/action spaces are on ``.single_observation_space`` /
``.single_action_space``; batched training reads ``.num_envs``.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from ..config import Config, Direction
from ..observation.layouts import observation_bounds
from ..actions import modes as action_modes
from ..paths import generator as pathgen
from .backend import xp, to_numpy
from .vehicle import BatchedTractorTrailer
from . import geometry as geo
from . import reward as rw
from . import corridor as cor
from .lidar import raycast


class BatchedLaneFollowingEnv:
    def __init__(self, cfg: Config, num_envs: int, path_pool_size: int = 4096):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.path_pool_size = int(path_pool_size)
        self.vehicle = BatchedTractorTrailer(cfg, num_envs)

        self.single_action_space = action_modes.action_space(cfg)
        low, high = observation_bounds(cfg)
        self.single_observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_dim = action_modes.action_dim(cfg)
        # evenly-spaced steer-rate bins for the DISCRETE action mode
        self._max_rate = float(np.deg2rad(cfg.action.steering_action_deg))
        self._steer_bins = xp.linspace(-self._max_rate, self._max_rate, cfg.action.n_steer_bins)
        # guided reward: PP-imitation warmup (alpha 1) decaying to multiplicative over
        # ~100k total env-steps. Reference = the tuned pure-pursuit gains.
        self._guided = (cfg.reward.mode == "guided")
        self._guide_steps = 0
        self._guide_transition = 100_000
        from .pure_pursuit import FWD_GAINS, REV_GAINS
        self._pp_gains = REV_GAINS if cfg.is_reverse else FWD_GAINS

        # cached constants
        lk = cfg.lookahead
        self.k1s, self.k2s = lk.k1_samples, lk.k2_samples
        self.maxk = cfg.obs.curvature_observation
        self.scale = cfg.obs.error_theta_scale
        self.tan_off = 1  # heading_preview_s==0 => nearest+1 (parity)
        self.reverse = cfg.direction == Direction.REVERSE
        self.tractor_only = cfg.is_tractor_only
        self.max_steps = cfg.max_episode_steps

        # path storage (xs shared once we generate the first path)
        self.xs = None                                   # (P,) backend array
        self._xs_np = None                               # (P,) host
        self.ys = None                                   # (N,P) backend
        self._ys_np = None                               # (N,P) host  (legacy path)
        self.path_kind = ["straight"] * self.num_envs

        # On-device path pool: generate a fixed pool of reference paths ONCE on the
        # host, then each reset samples pool rows with a GPU gather. This removes the
        # per-env host path-generation loop from the per-step hot path (the reset
        # bottleneck) while preserving the exact path-kind distribution.
        self._pool_ys = None        # (POOL,P) backend
        self._pool_ys_np = None     # (POOL,P) host
        self._pool_lo = None        # (POOL,) host speed lower bound per pool path
        self._pool_hi = None        # (POOL,) host speed upper bound per pool path
        self._env_pool_idx = np.zeros(self.num_envs, dtype=np.int64)  # which pool path each env uses

        self.episode_speed = xp.zeros(self.num_envs, dtype=xp.float64)
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.success = xp.zeros(self.num_envs, dtype=bool)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ paths
    def _gen_path_host(self, env_idx):
        xx, yy, _, kind = pathgen.generate_path(self._rng, self.cfg.world, self.cfg.path)
        if self._xs_np is None:
            self._xs_np = np.asarray(xx, dtype=np.float64)
            self.xs = xp.asarray(self._xs_np)
            P = len(xx)
            self._ys_np = np.zeros((self.num_envs, P), dtype=np.float64)
        self._ys_np[env_idx] = yy
        self.path_kind[env_idx] = kind

    def _sample_speed_host(self, kind):
        sr = self.cfg.speed_random
        if not sr.enabled:
            return self.cfg.action.fixed_speed_m_s
        if sr.explicit_min is not None and sr.explicit_max is not None:
            return float(self._rng.uniform(sr.explicit_min, sr.explicit_max))
        lo, hi = sr.per_kind.get(kind, (1.0, 5.0))
        return float(self._rng.uniform(lo, hi))

    def _build_pool(self):
        """Generate the fixed path pool once on the host; move ys to the backend."""
        pool = self.path_pool_size
        rows = []
        lo = np.zeros(pool); hi = np.zeros(pool)
        for k in range(pool):
            xx, yy, _, kind = pathgen.generate_path(self._rng, self.cfg.world, self.cfg.path)
            if self._xs_np is None:
                self._xs_np = np.asarray(xx, dtype=np.float64)
                self.xs = xp.asarray(self._xs_np)
            rows.append(yy)
            sr = self.cfg.speed_random
            if not sr.enabled:
                lo[k] = hi[k] = self.cfg.action.fixed_speed_m_s
            elif sr.explicit_min is not None and sr.explicit_max is not None:
                lo[k], hi[k] = sr.explicit_min, sr.explicit_max
            else:
                lo[k], hi[k] = sr.per_kind.get(kind, (1.0, 5.0))
        self._pool_ys_np = np.asarray(rows, dtype=np.float64)      # (POOL,P)
        self._pool_ys = xp.asarray(self._pool_ys_np)
        self._pool_lo, self._pool_hi = lo, hi

    def _reset_idxs(self, idxs):
        """Sample pool paths + spawn for the given env indices (vectorised gather)."""
        if len(idxs) == 0:
            return
        if self._pool_ys is None:
            return self._reset_idxs_legacy(idxs)
        idxs = np.asarray(idxs, dtype=np.int64)
        # assign fresh pool paths to the done envs
        self._env_pool_idx[idxs] = self._rng.integers(0, self.path_pool_size, size=len(idxs))
        self.ys = self._pool_ys[xp.asarray(self._env_pool_idx)]     # (N,P) GPU gather

        P = self._xs_np.shape[0]
        spawn_i = int(np.clip(int(0.1 * (P - 1)), 0, P - 2)) if P >= 2 else 0
        sp = self.cfg.spawn
        pool_i = self._env_pool_idx                                  # (N,)
        ys0 = self._pool_ys_np[pool_i, spawn_i]                      # (N,) host
        ys1 = self._pool_ys_np[pool_i, spawn_i + 1]
        x0 = np.full(self.num_envs, self._xs_np[spawn_i])
        yaw0 = np.arctan2(ys1 - ys0, self._xs_np[spawn_i + 1] - self._xs_np[spawn_i])
        head0 = yaw0 + (np.pi if self.reverse else 0.0)
        speeds = self._rng.uniform(self._pool_lo[pool_i], self._pool_hi[pool_i])
        hitch0 = np.zeros(self.num_envs)
        if sp.lateral_offset_m or sp.heading_offset_rad or ((not self.tractor_only) and sp.hitch_offset_rad):
            for e in idxs:
                if sp.lateral_offset_m:
                    lat = self._rng.uniform(-sp.lateral_offset_m, sp.lateral_offset_m)
                    x0[e] += -np.sin(yaw0[e]) * lat; ys0[e] += np.cos(yaw0[e]) * lat
                if sp.heading_offset_rad:
                    head0[e] += self._rng.uniform(-sp.heading_offset_rad, sp.heading_offset_rad)
                if (not self.tractor_only) and sp.hitch_offset_rad:
                    hitch0[e] = self._rng.uniform(-sp.hitch_offset_rad, sp.hitch_offset_rad)

        mask_np = np.zeros(self.num_envs, dtype=bool); mask_np[idxs] = True
        mask = xp.asarray(mask_np)
        has_hitch = (not self.tractor_only) and bool(sp.hitch_offset_rad)
        self.vehicle.reset_where(mask, xp.asarray(speeds), xp.asarray(x0),
                                 xp.asarray(ys0), xp.asarray(head0),
                                 trailer_yaw=xp.asarray(head0 - hitch0) if has_hitch else None)
        if self.tractor_only:
            self.vehicle.align_trailer(mask)
        ep = xp.asarray(np.where(self.reverse, -speeds, speeds))
        self.episode_speed = xp.where(mask, ep, self.episode_speed)
        self.step_count[idxs] = 0
        self.success = xp.where(mask, False, self.success)

    def _reset_idxs_legacy(self, idxs):
        """(Re)generate paths + spawn per-env on the host (fallback / no pool)."""
        if len(idxs) == 0:
            return
        for e in idxs:
            self._gen_path_host(int(e))
        self.ys = xp.asarray(self._ys_np)

        P = self._xs_np.shape[0]
        spawn_i = int(np.clip(int(0.1 * (P - 1)), 0, P - 2)) if P >= 2 else 0
        sp = self.cfg.spawn
        speeds = np.zeros(self.num_envs)
        xs0 = np.full(self.num_envs, self._xs_np[spawn_i])
        ys0 = self._ys_np[:, spawn_i].copy()
        yaw0 = np.arctan2(self._ys_np[:, spawn_i + 1] - self._ys_np[:, spawn_i],
                          self._xs_np[spawn_i + 1] - self._xs_np[spawn_i])
        head0 = yaw0 + (np.pi if self.reverse else 0.0)
        hitch0 = np.zeros(self.num_envs)
        for e in idxs:
            e = int(e)
            speeds[e] = self._sample_speed_host(self.path_kind[e])
            if sp.lateral_offset_m:
                lat = self._rng.uniform(-sp.lateral_offset_m, sp.lateral_offset_m)
                xs0[e] += -np.sin(yaw0[e]) * lat
                ys0[e] += np.cos(yaw0[e]) * lat
            if sp.heading_offset_rad:
                head0[e] += self._rng.uniform(-sp.heading_offset_rad, sp.heading_offset_rad)
            if (not self.tractor_only) and sp.hitch_offset_rad:
                hitch0[e] = self._rng.uniform(-sp.hitch_offset_rad, sp.hitch_offset_rad)

        mask_np = np.zeros(self.num_envs, dtype=bool); mask_np[np.asarray(idxs, dtype=int)] = True
        mask = xp.asarray(mask_np)
        trailer_yaw = xp.asarray(head0 - hitch0)
        self.vehicle.reset_where(mask, xp.asarray(speeds), xp.asarray(xs0),
                                 xp.asarray(ys0), xp.asarray(head0),
                                 trailer_yaw=None if not (not self.tractor_only and sp.hitch_offset_rad) else trailer_yaw)
        if self.tractor_only:
            self.vehicle.align_trailer(mask)

        ep = xp.asarray(np.where(self.reverse, -speeds, speeds))
        self.episode_speed = xp.where(mask, ep, self.episode_speed)
        self.step_count[np.asarray(idxs, dtype=int)] = 0
        self.success = xp.where(mask, False, self.success)

    # ------------------------------------------------------------------ observe
    def _lidar_pose(self):
        v = self.vehicle
        if not self.reverse:
            return v.x, v.y, v.p
        if self.tractor_only:
            rx = v.x - self.cfg.vehicle.lr * xp.cos(v.p)
            ry = v.y - self.cfg.vehicle.lr * xp.sin(v.p)
            return rx, ry, v.p + np.pi
        return v.tx, v.ty, v.tyaw + np.pi

    def _observe(self):
        v = self.vehicle
        k1 = geo.curvature_at(self.xs, self.ys, v.tx, v.ty, self.k1s, self.maxk)
        k2 = geo.curvature_at(self.xs, self.ys, v.tx, v.ty, self.k2s, self.maxk)
        e_y, e_psi = geo.path_errors(self.xs, self.ys, v.x, v.y, v.p,
                                     reverse=self.reverse, error_theta_scale=self.scale,
                                     tangent_offset=self.tan_off)
        if self.tractor_only:
            cols = [v.s, e_y, e_psi, k1, k2]
            errors = dict(e_y=e_y, e_psi=e_psi, k1=k1, steer=v.s)
        else:
            hitch = v.p - v.tyaw
            e_y_t, e_psi_t = geo.path_errors(self.xs, self.ys, v.tx, v.ty, v.tyaw,
                                             reverse=self.reverse, error_theta_scale=self.scale,
                                             tangent_offset=self.tan_off)
            cols = [v.s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2]
            errors = dict(hitch=hitch, e_y=e_y, e_psi=e_psi, e_y_t=e_y_t, e_psi_t=e_psi_t,
                          k1=k1, steer=v.s)
        obs = xp.stack(cols, axis=1)
        if self.cfg.obs.lidar_beams > 0:
            lx, ly, lyaw = self._lidar_pose()
            d = raycast(self.xs, self.ys, self.cfg, lx, ly, lyaw)
            obs = xp.concatenate([obs, d], axis=1)
        return obs.astype(xp.float32), errors

    # ------------------------------------------------------------------ term
    def _terminated_masks(self):
        v = self.vehicle
        oob = ~((v.x >= 0) & (v.x <= self.cfg.world.width_m)
                & (v.y >= 0) & (v.y <= self.cfg.world.height_m))
        h = xp.abs((v.p - v.tyaw + np.pi) % (2 * np.pi) - np.pi)
        jack = h > (np.pi / 2)
        coll = cor.collision(self.xs, self.ys, v, self.cfg)
        env_len = self.cfg.world.width_m * 0.85
        succ = (v.x > env_len) | (v.tx > env_len)
        terminated = oob | jack | coll | succ
        return terminated, succ

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._pool_ys is None:
            self._build_pool()
        self._reset_idxs(list(range(self.num_envs)))
        obs, _ = self._observe()
        return obs

    def step(self, actions):
        self.step_count += 1
        mode = self.cfg.action.mode

        if mode.value == "discrete":
            # Discrete(n_steer_bins*2): idx = steer_bin*2 + speed_bin (0=go,1=stop).
            ai = xp.asarray(actions).reshape(-1).astype(xp.int64)
            steer_idx = ai // 2
            speed_idx = ai % 2
            steer_rate = self._steer_bins[steer_idx]
            stop = speed_idx == 1
            velocity_cmd = xp.where(stop, 0.0, self.episode_speed)
        else:
            a = xp.asarray(actions, dtype=xp.float64)
            if a.ndim == 1:
                a = a[:, None]
            steer_rate = a[:, 0]
            if mode.value == "fixed_speed":
                velocity_cmd = self.episode_speed
                stop = xp.zeros(self.num_envs, dtype=bool)
            elif mode.value == "variable_speed":
                velocity_cmd = a[:, 1]
                stop = xp.zeros(self.num_envs, dtype=bool)
            else:  # stop_signal
                stop = a[:, 1] > self.cfg.action.stop_threshold
                velocity_cmd = xp.where(stop, 0.0, self.episode_speed)

        self.vehicle.step(steer_rate, velocity_cmd)
        if self.tractor_only:
            self.vehicle.align_trailer()

        terminated, succ = self._terminated_masks()
        self.success = succ
        truncated = xp.asarray(self.step_count >= self.max_steps)

        obs, errors = self._observe()
        prox = cor.proximity_penalty(self.xs, self.ys, self.vehicle, self.cfg)
        gkw = {}
        if self._guided:
            g = self._pp_gains; s = errors["steer"]; k1 = errors["k1"]
            if self.reverse:
                delta_des = (g["k_hitch"] * errors.get("hitch", 0.0) + g["k_ff"] * k1
                             + g["k_y"] * errors.get("e_y_t", 0.0)
                             + g["k_theta"] * errors.get("e_psi_t", 0.0))
            else:
                wheelbase = self.vehicle.lf + self.vehicle.lr
                delta_des = -(g["k_ff"] * xp.arctan(wheelbase * k1)
                              + g["k_y"] * errors["e_y"] + g["k_theta"] * errors["e_psi"])
            pp_steer = xp.clip((delta_des - s) / self.vehicle.dt, -self._max_rate, self._max_rate)
            gkw["steer_diff"] = (steer_rate - pp_steer) / (2.0 * self._max_rate)
            gkw["alpha"] = max(0.0, 1.0 - self._guide_steps / self._guide_transition)
            self._guide_steps += self.num_envs
        running = rw.running_reward(
            self.cfg, e_y=errors["e_y"], e_psi=errors["e_psi"],
            e_y_t=errors.get("e_y_t", 0.0), e_psi_t=errors.get("e_psi_t", 0.0),
            hitch=errors.get("hitch", 0.0), xd=self.vehicle.xd, proximity=prox, **gkw)
        term_r = rw.terminal_reward(self.cfg, succ)
        reward = xp.where(terminated, term_r, running)
        # stop-signal fires terminal stop penalty and terminates
        stop_pen = -abs(self.cfg.action.stop_penalty)
        terminated = terminated | stop
        reward = xp.where(stop, stop_pen, reward)

        done = terminated | truncated
        final_obs = obs
        done_np = to_numpy(done).astype(bool)
        idxs = np.nonzero(done_np)[0]
        # success mask captured BEFORE auto-reset (auto-reset clears self.success;
        # reading it afterwards is the completion-metric bug we avoid here).
        info = {"success": to_numpy(succ).astype(bool)}
        if len(idxs):
            info["final_observation"] = final_obs
            info["_done_idx"] = idxs
            self._reset_idxs(idxs.tolist())
            obs, _ = self._observe()

        return obs, reward, terminated, truncated, info
