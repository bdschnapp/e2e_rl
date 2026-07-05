"""Pure observation builder — the single code path shared by training and deploy.

`build_observation` takes a centerline, ego/trailer state, and a Config, and
returns the policy observation vector. No pygame, no env instantiation, no global
config. The Gymnasium env's `_get_obs` is a thin shim over this; the ROS bridge
calls it directly (replacing the old ROSLineFollowingAdapter that mutated a full
env).

Direction enters in exactly three physical places (no empirical sign flips):
  1. the `reverse` flag into path_errors (heading wrap, fix A),
  2. the lidar mount point/heading,
  3. (upstream) the caller's frame is +X = direction of travel.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..config import Config, LookaheadMode
from .. import geometry as geo
from .occupancy import build_occupancy_grid, GridMeta
from .lidar import raycast, Pose


@dataclass(frozen=True)
class EgoState:
    x: float
    y: float
    yaw: float
    steer: float
    xd: float


@dataclass(frozen=True)
class TrailerState:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ObsResult:
    vector: np.ndarray
    errors: dict
    nearest_idx: int


def _curvature_steps(cfg: Config, speed: float) -> tuple[int, int]:
    lk = cfg.lookahead
    if lk.mode == LookaheadMode.FIXED_SAMPLES:
        return lk.k1_samples, lk.k2_samples
    k1 = geo.lookahead_steps_for_speed(speed, lk.preview_time_s[0], cfg.path.spacing_m,
                                       lk.dist_min_m, lk.dist_max_m)
    k2 = geo.lookahead_steps_for_speed(speed, lk.preview_time_s[1], cfg.path.spacing_m,
                                       lk.dist_min_m, lk.dist_max_m)
    return k1, k2


def _heading_offset(cfg: Config, speed: float) -> int:
    lk = cfg.lookahead
    if lk.heading_preview_s and lk.heading_preview_s > 0.0:
        return geo.lookahead_steps_for_speed(speed, lk.heading_preview_s, cfg.path.spacing_m,
                                             lk.dist_min_m, lk.dist_max_m)
    return 1  # parity: nearest/nearest+1 segment


def _lidar_pose(cfg: Config, ego: EgoState, trailer: TrailerState | None) -> Pose:
    if not cfg.is_reverse:
        # forward: trailer env and tractor-only both mount on the tractor.
        return Pose(ego.x, ego.y, ego.yaw)
    if cfg.is_tractor_only:
        # reverse tractor-only: tractor rear axle, facing backward.
        rear_x = ego.x - cfg.vehicle.lr * np.cos(ego.yaw)
        rear_y = ego.y - cfg.vehicle.lr * np.sin(ego.yaw)
        return Pose(rear_x, rear_y, ego.yaw + np.pi)
    # reverse trailer: trailer (leads in reverse), facing approach direction.
    return Pose(trailer.x, trailer.y, trailer.yaw + np.pi)


def build_observation(
    xs: np.ndarray,
    ys: np.ndarray,
    ego: EgoState,
    trailer: TrailerState | None,
    cfg: Config,
    *,
    occ_grid: np.ndarray | None = None,
    occ_meta: GridMeta | None = None,
) -> ObsResult:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    reverse = cfg.is_reverse
    scale = cfg.obs.error_theta_scale
    tan_off = _heading_offset(cfg, ego.xd)

    # curvature query point = trailer axle (rigid-aligned behind tractor for
    # tractor-only) — matches e2e_rl compute_curvature(self) which uses trailer.
    curv_pt = trailer if trailer is not None else TrailerState(ego.x, ego.y, ego.yaw)
    k1s, k2s = _curvature_steps(cfg, ego.xd)
    maxk = cfg.obs.curvature_observation
    k1 = geo.curvature_at(xs, ys, curv_pt.x, curv_pt.y, k1s, maxk)
    k2 = geo.curvature_at(xs, ys, curv_pt.x, curv_pt.y, k2s, maxk)

    e_y, e_psi = geo.path_errors(xs, ys, ego.x, ego.y, ego.yaw,
                                 reverse=reverse, error_theta_scale=scale,
                                 tangent_offset=tan_off)
    nearest = geo.nearest_index(xs, ys, ego.x, ego.y)

    if cfg.is_tractor_only:
        vec = [ego.steer, e_y, e_psi, k1, k2]
        errors = dict(e_y=e_y, e_psi=e_psi, k1=k1, k2=k2)
    else:
        hitch = ego.yaw - trailer.yaw
        e_y_t, e_psi_t = geo.path_errors(xs, ys, trailer.x, trailer.y, trailer.yaw,
                                         reverse=reverse, error_theta_scale=scale,
                                         tangent_offset=tan_off)
        vec = [ego.steer, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2]
        errors = dict(hitch=hitch, e_y=e_y, e_psi=e_psi, e_y_t=e_y_t,
                      e_psi_t=e_psi_t, k1=k1, k2=k2)

    vec = np.array(vec, dtype=np.float32)

    if cfg.obs.lidar_beams > 0:
        if occ_grid is None:
            occ_grid, occ_meta = build_occupancy_grid(xs, ys, cfg.world)
        pose = _lidar_pose(cfg, ego, trailer)
        d = raycast(occ_grid, occ_meta, pose,
                    num_sensors=cfg.obs.lidar_beams,
                    fov_deg=cfg.obs.lidar_fov_deg,
                    max_range_m=cfg.obs.lidar_range_m,
                    step_m=cfg.obs.lidar_step_m)
        vec = np.concatenate([vec, d.astype(np.float32)])

    return ObsResult(vector=vec, errors=errors, nearest_idx=nearest)
