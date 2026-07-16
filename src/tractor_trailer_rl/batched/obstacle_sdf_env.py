"""Obstacle env with an SDF-BEV image (Phase-3; NEW, additive).

Subclass of the Track-D obstacle env that swaps its BEV image to signed-distance
(+ optional CoordConv), exactly as SdfBevEnv does for the lane env. Obstacles are drawn
into the image automatically (the obstacle env's `_bev_obstacles` supplies the circles,
which `render_bev_sdf` marks as blocked). The frozen occupancy path is untouched.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .obstacle_env import BatchedObstacleAvoidanceEnv
from .bev_sdf import render_bev_sdf
from .bev_sdf_env import SdfObsCfg
from .obstacle_geom import obstacle_range_profile, obstacle_profile_names

OBSTACLE_PROFILE_BINS = 15
OBSTACLE_PROFILE_NAMES = obstacle_profile_names(OBSTACLE_PROFILE_BINS)


class ObstacleSdfBevEnv(BatchedObstacleAvoidanceEnv):
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
        ox, oy, orad, ovalid = self._bev_obstacles()      # obstacle circles -> drawn into the SDF image
        return render_bev_sdf(self.xs, self.ys, self.cfg, lx, ly, lyaw,
                              coord=self._sdf_cfg.coord, ox=ox, oy=oy, orad=orad, ovalid=ovalid)

    def obstacle_profile(self, nbins=OBSTACLE_PROFILE_BINS):
        """(N, nbins) obstacle angular clearance profile — the TRUE obstacle state (privileged
        teacher) and the LABEL the distillation encoder predicts from the image. Order-invariant,
        count-agnostic, continuous. See batched/obstacle_geom.py."""
        lx, ly, lyaw = self._lidar_pose()
        return obstacle_range_profile(self.ox, self.oy, self.orad, self.ovalid,
                                      lx, ly, lyaw, float(self.cfg.obs.bev_range_m), nbins=nbins)
