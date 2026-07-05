"""Observation vector layouts + gym Box bounds.

TRAILER_8:     [s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2]  (+ lidar)
TRACTOR_5:     [s, e_y, e_psi, k1, k2]                          (+ lidar)
"""

from __future__ import annotations

import numpy as np


def state_bounds(cfg) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) for the state portion of the obs (before lidar), per kind."""
    o = cfg.obs
    if cfg.is_tractor_only:
        low = np.array([
            -o.steering_observation,
            -o.cross_track_distance_observation,
            -o.cross_track_angle_observation,
            -o.curvature_observation,
            -o.curvature_observation,
        ], dtype=np.float32)
    else:
        low = np.array([
            -o.steering_observation,
            -o.hitch_angle_observation,
            -o.cross_track_distance_observation,
            -o.cross_track_angle_observation,
            -o.cross_track_distance_observation,
            -o.cross_track_angle_observation,
            -o.curvature_observation,
            -o.curvature_observation,
        ], dtype=np.float32)
    return low, -low


def observation_bounds(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Full (low, high) including lidar beams."""
    low, high = state_bounds(cfg)
    beams = cfg.obs.lidar_beams
    if beams > 0:
        low = np.concatenate([low, np.zeros(beams, dtype=np.float32)])
        high = np.concatenate([high, np.ones(beams, dtype=np.float32)])
    return low, high
