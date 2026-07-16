"""SDF-BEV env variant (Phase-2B vision rescue — NEW, additive).

A subclass of the frozen BatchedLaneFollowingEnv that swaps ONLY the BEV image:
signed-distance (+ optional CoordConv) instead of binary occupancy. The frozen
occupancy path (base env, run_stage1_sweep) is untouched. Mirrors how
obstacle_env.py subclasses the base env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from .env import BatchedLaneFollowingEnv
from .bev_sdf import render_bev_sdf


@dataclass(frozen=True)
class SdfObsCfg:
    """Which SDF-BEV channels to render. sdf is always on; coord adds 2 channels."""
    coord: bool = False


class SdfBevEnv(BatchedLaneFollowingEnv):
    def __init__(self, cfg, num_envs, sdf_cfg: SdfObsCfg = SdfObsCfg(), path_pool_size: int = 4096):
        self._sdf_cfg = sdf_cfg
        super().__init__(cfg, num_envs, path_pool_size=path_pool_size)
        if self._bev_on:
            S = int(cfg.obs.bev_size)
            C = 1 + (2 if sdf_cfg.coord else 0)
            img_space = gym.spaces.Box(-1.0, 1.0, (C, S, S), dtype=np.float32)
            vec_space = self.single_observation_space.spaces["vector"]
            self.single_observation_space = gym.spaces.Dict(
                {"vector": vec_space, "image": img_space})

    def _bev(self):
        lx, ly, lyaw = self._lidar_pose()
        ox, oy, orad, ovalid = self._bev_obstacles()
        # render_bev_sdf returns (N, C, S, S) already (no channel-expand needed).
        return render_bev_sdf(self.xs, self.ys, self.cfg, lx, ly, lyaw,
                              coord=self._sdf_cfg.coord, ox=ox, oy=oy, orad=orad, ovalid=ovalid)
