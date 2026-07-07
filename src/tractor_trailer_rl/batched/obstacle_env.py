"""Batched obstacle-avoidance env with the hidden-local-planner reward + stop-gate.

Subclass of ``BatchedLaneFollowingEnv`` (Track D). It reuses the parity-tested
dynamics / geometry / corridor / auto-reset machinery unchanged and only adds the
obstacle layer, so the pure lane-following ablation env (``obstacle=None``) is
completely unaffected -- both run side by side.

What it overrides:
  * ``_reset_idxs``       place circles + build the hidden avoidance path + score
                          layout difficulty for the envs being (re)set.
  * ``_lidar``            min-combine the corridor lidar with obstacle ray hits so
                          the policy can *see* obstacles (its only obstacle signal).
  * ``_terminated_masks`` add obstacle collision to the termination set.
  * ``_compute_reward``   reward tracking the *hidden* local-planner path (agent
                          never observes it), a slow-down-on-difficulty bonus, and
                          the difficulty-gated STOP_SIGNAL reward: stopping pays off
                          only when the layout is hard, and is penalised when easy.
  * ``step``              surface per-episode difficulty / stop in ``info`` for
                          difficulty-stratified evaluation.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .backend import xp, to_numpy
from .env import BatchedLaneFollowingEnv
from . import geometry as geo
from . import corridor as cor
from . import reward as rw
from . import obstacles as obs


class BatchedObstacleAvoidanceEnv(BatchedLaneFollowingEnv):
    def __init__(self, cfg, num_envs, path_pool_size=4096):
        assert cfg.obstacle is not None, "obstacle env requires cfg.obstacle"
        super().__init__(cfg, num_envs, path_pool_size=path_pool_size)
        K = int(cfg.obstacle.max_obstacles)
        N = self.num_envs
        self.ox = xp.zeros((N, K)); self.oy = xp.zeros((N, K)); self.orad = xp.zeros((N, K))
        self.olat = xp.zeros((N, K)); self.ostation = xp.zeros((N, K))
        self.ovalid = xp.zeros((N, K), dtype=bool)
        self.local_ys = None                      # (N,P) hidden avoidance path (lazy)
        self.difficulty = xp.zeros(N)             # (N,) current-episode layout difficulty
        self._last_difficulty = None              # pre-reset snapshot for eval
        self._last_stop = None
        # the hidden-planner reward tracks the local path with the configured base
        # mode; 'guided' (a centreline PP teacher) is meaningless here -> multiplicative.
        rmode = cfg.obstacle.reward_mode
        if rmode == "guided":
            rmode = "multiplicative"
        self._reward_cfg = cfg.evolve(reward=replace(cfg.reward, mode=rmode))

    # ------------------------------------------------------------------ reset
    def _reset_idxs(self, idxs):
        super()._reset_idxs(idxs)
        if len(idxs) == 0:
            return
        if self.local_ys is None or self.local_ys.shape != self.ys.shape:
            self.local_ys = xp.array(self.ys)     # copy; overwritten per-env below
        midx = xp.asarray(np.asarray(idxs, dtype=np.int64))
        ys_np = to_numpy(self.ys)[np.asarray(idxs, dtype=np.int64)]     # (M,P)
        ox, oy, orad, olat, ost, ovalid = obs.place_host(self._rng, self._xs_np, ys_np, self.cfg)
        self.ox[midx] = xp.asarray(ox); self.oy[midx] = xp.asarray(oy)
        self.orad[midx] = xp.asarray(orad); self.olat[midx] = xp.asarray(olat)
        self.ostation[midx] = xp.asarray(ost); self.ovalid[midx] = xp.asarray(ovalid)
        # hidden avoidance path + difficulty for the reset subset
        off = obs.plan_offsets(self.xs, self.ys[midx], self.ox[midx], self.oy[midx],
                               self.orad[midx], self.ovalid[midx], self.cfg)
        self.local_ys[midx] = self.ys[midx] + off
        self.difficulty[midx] = obs.difficulty(self.olat[midx], self.ostation[midx],
                                               self.orad[midx], self.ovalid[midx], self.cfg)

    # ------------------------------------------------------------------ observe
    def _lidar(self, lx, ly, lyaw):
        d_cor = super()._lidar(lx, ly, lyaw)
        d_obs = obs.lidar(self.ox, self.oy, self.orad, self.ovalid, lx, ly, lyaw, self.cfg)
        return xp.minimum(d_cor, d_obs)

    def _bev_obstacles(self):
        """Draw this env's obstacle circles into the BEV image (blocked cells)."""
        return self.ox, self.oy, self.orad, self.ovalid

    # ------------------------------------------------------------------ term
    def _terminated_masks(self):
        terminated, succ = super()._terminated_masks()
        bx, by = cor.body_points(self.vehicle, self.cfg)
        coll = obs.collision(self.ox, self.oy, self.orad, self.ovalid, bx, by, self.cfg)
        return terminated | coll, succ

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, errors, prox, terminated, succ, stop, gkw):
        v = self.vehicle
        oc = self.cfg.obstacle
        # hidden local-planner tracking: errors vs the avoidance path (NOT observed)
        le_y, le_psi = geo.path_errors(self.xs, self.local_ys, v.x, v.y, v.p,
                                       reverse=self.reverse, error_theta_scale=self.scale,
                                       tangent_offset=self.tan_off)
        if self.tractor_only:
            le_y_t = 0.0; le_psi_t = 0.0
        else:
            le_y_t, le_psi_t = geo.path_errors(self.xs, self.local_ys, v.tx, v.ty, v.tyaw,
                                               reverse=self.reverse, error_theta_scale=self.scale,
                                               tangent_offset=self.tan_off)
        bx, by = cor.body_points(v, self.cfg)
        prox_total = prox + obs.proximity(self.ox, self.oy, self.orad, self.ovalid, bx, by, self.cfg)
        running = rw.running_reward(self._reward_cfg, e_y=le_y, e_psi=le_psi,
                                    e_y_t=le_y_t, e_psi_t=le_psi_t,
                                    hitch=errors.get("hitch", 0.0), xd=v.xd, proximity=prox_total)
        # slow down when the layout is hard
        max_speed = max(abs(self.cfg.action.fixed_speed_m_s), 1e-6)
        norm_speed = xp.clip(xp.abs(v.xd) / max_speed, 0.0, 1.0)
        running = running + self.difficulty * (1.0 - norm_speed) * oc.slow_reward_scale
        # success / crash terminals (direction defaults)
        term_r = rw.terminal_reward(self.cfg, succ)
        reward = xp.where(terminated, term_r, running)
        # difficulty-keyed stop-gate: stopping pays on an impassable layout (must beat
        # a crash) and is penalised on an easy one (must lose to driving on). Keyed on
        # difficulty (not progress) so it is robust to where the obstacle sits; a small
        # progress term still credits reaching the obstacle before stopping.
        env_len = self.cfg.world.width_m * 0.85
        progress = xp.clip(xp.maximum(v.x, v.tx) / env_len, 0.0, 1.0)
        exp = obs.exp_scale(self.difficulty, oc.stop_difficulty_k)
        stop_reward = (exp * oc.stop_hard_reward - (1.0 - exp) * oc.stop_easy_penalty
                       + progress * oc.stop_progress_bonus)
        # snapshot pre-reset difficulty/stop for stratified eval (auto-reset overwrites)
        self._last_difficulty = self.difficulty.copy()
        self._last_stop = stop
        terminated = terminated | stop
        reward = xp.where(stop, stop_reward, reward)
        return reward, terminated

    # ------------------------------------------------------------------ gym API
    def step(self, actions):
        obs_, reward, terminated, truncated, info = super().step(actions)
        if self._last_difficulty is not None:
            info["difficulty"] = to_numpy(self._last_difficulty)
            info["stopped"] = to_numpy(self._last_stop).astype(bool)
        return obs_, reward, terminated, truncated, info
