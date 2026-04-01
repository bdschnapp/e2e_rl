import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from dataclasses import dataclass
import pygame

import e2erl_utils.config as config

from Environments.LineFollowing import (
    StateObservationLineFollowingEnv,
    ReverseStateObservationLineFollowingEnv,
    BevObservationLineFollowingEnv,
    ReverseBevObservationLineFollowingEnv,
)
from Environments.TractorTrailer import WINDOW_HEIGHT, WINDOW_WIDTH, METERS_PER_PIXEL, TRACTOR_WIDTH, TRAILER_WIDTH


@dataclass
class Pose:
    x: float    # meters
    y: float    # meters
    yaw: float  # radians


def get_obstacle_distances(
    occ_grid: NDArray[np.uint8],
    lidar_pose: Pose,
    *,
    num_sensors: int = 16,
    fov: float = np.deg2rad(120.0),
    max_range_m: float = 20.0,
    step_m: float = 0.05,
) -> NDArray[np.float32]:
    """
    Returns normalized distances [0,1] for each lidar beam.
    1.0 = no obstacle within range
    """

    H, W = occ_grid.shape

    # Convert lidar pose to pixel coordinates
    x0 = int(lidar_pose.x / METERS_PER_PIXEL)
    y0 = int(lidar_pose.y / METERS_PER_PIXEL)

    max_steps = int(max_range_m / step_m)

    # Beam angles in vehicle frame
    angles = np.linspace(
        -fov / 2.0,
        +fov / 2.0,
        num_sensors,
        dtype=np.float32,
    )

    distances = np.ones(num_sensors, dtype=np.float32)

    for i, a in enumerate(angles):
        theta = lidar_pose.yaw + a

        dx = np.cos(theta)
        dy = np.sin(theta)

        for step in range(1, max_steps + 1):
            dist_m = step * step_m
            px = int(x0 + (dist_m / METERS_PER_PIXEL) * dx)
            py = int(y0 + (dist_m / METERS_PER_PIXEL) * dy)

            # Out of bounds → treat as obstacle
            if px < 0 or px >= W or py < 0 or py >= H:
                distances[i] = dist_m / max_range_m
                break

            if occ_grid[py, px] > 0:
                distances[i] = dist_m / max_range_m
                break

    return distances


def generate_obstacles(
    centerline: NDArray[np.float32],
    occ_grid: NDArray[np.uint8],
    num_obstacles: int,
    *,
    lateral_offset_range_m=(1.0, 5.0),
    radius_range_m=(0.8, 1.4),
    longitudinal_jitter_m=0.5,
    rng=None,
):
    """
    Generates circular obstacles near the lane centerline.

    centerline: (N,2) array in meters
    occ_grid: occupancy grid to write into (modified in-place)
    """

    H, W = occ_grid.shape
    n_pts = len(centerline)

    if num_obstacles <= 0 or n_pts < 2:
        return occ_grid, None

    usable = n_pts - 1  # because we access idx+1
    B = min(num_obstacles, usable)  # cannot have more bins than usable indices
    edges = np.linspace(0, usable, B + 1).astype(int)

    # Build bins as (start, end_exclusive) index ranges
    bins = []
    for i in range(B):
        a = edges[i]
        b = edges[i + 1]
        # Ensure each bin has at least one valid index
        if b <= a:
            b = min(a + 1, usable)
        if a < usable:
            bins.append((a, b))

    if not bins:
        return occ_grid, None
    _rng = rng if rng is not None else np.random.default_rng()
    bin_order = _rng.permutation(len(bins))

    obstacles: list[tuple[float, float, float]] = []
    for k in range(num_obstacles):
        bi = bin_order[k % len(bins)]
        a, b = bins[bi]

        # Sample an index within this bin
        idx = int(_rng.integers(a, b))
        p = centerline[idx]
        p_next = centerline[idx + 1]

        # Tangent (lane direction)
        t = p_next - p
        t /= np.linalg.norm(t) + 1e-6

        # Normal (left/right)
        n = np.array([-t[1], t[0]])

        # Random lateral offset (left or right)
        lat_mag = _rng.uniform(*lateral_offset_range_m)
        lat_sign = _rng.choice([-1.0, 1.0])
        lateral_offset = lat_sign * lat_mag

        # Optional longitudinal jitter
        longitudinal_offset = _rng.uniform(
            -longitudinal_jitter_m,
            longitudinal_jitter_m
        )

        # Final obstacle position
        pos = p + lateral_offset * n + longitudinal_offset * t

        # Random obstacle size
        radius_m = _rng.uniform(*radius_range_m)
        radius_px = int(radius_m / METERS_PER_PIXEL)

        obstacles.append((float(pos[0]), float(pos[1]), float(radius_m)))

        # Convert to grid coordinates
        cx = int(pos[0] / METERS_PER_PIXEL)
        cy = int(pos[1] / METERS_PER_PIXEL)

        # Rasterize filled circle
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy > radius_px * radius_px:
                    continue

                x = cx + dx
                y = cy + dy

                if 0 <= x < W and 0 <= y < H:
                    occ_grid[y, x] = 100

    return occ_grid, obstacles


def plan_local_path(
    centerline: NDArray[np.float32],
    obstacles: list[tuple[float, float, float]],
    *,
    lane_half_width_m: float,
    vehicle_width_m: float,
    influence_radius_m: float = 6.0,
    max_lateral_shift_m: float | None = None,
    smoothing_window: int = 11,
) -> NDArray[np.float32]:
    """
    Computes a laterally-shifted path that avoids obstacles while staying in lane.
    Returns a new path with the SAME number of points and spacing as centerline.
    """

    assert centerline.ndim == 2 and centerline.shape[1] == 2
    N = len(centerline)

    if max_lateral_shift_m is None:
        max_lateral_shift_m = lane_half_width_m - vehicle_width_m * 0.5

    inflated_obstacles = [
        (ox, oy, r + vehicle_width_m * 0.75)
        for ox, oy, r in obstacles
    ]

    lateral_offsets = np.zeros(N, dtype=np.float32)

    def tangent_normal(i):
        i0 = min(i, N - 2)
        p = centerline[i0]
        p_next = centerline[i0 + 1]
        t = p_next - p
        t /= np.linalg.norm(t) + 1e-6
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return t, n

    lookahead_pts = 8  # small horizon, keeps this local

    for i in range(N):
        p = centerline[i]
        _, n = tangent_normal(i)

        left_score = 0.0
        right_score = 0.0

        # --- evaluate corridor feasibility ---
        for k in range(i, min(N, i + lookahead_pts)):
            pk = centerline[k]
            _, nk = tangent_normal(k)

            left_pos = pk + max_lateral_shift_m * nk
            right_pos = pk - max_lateral_shift_m * nk

            for ox, oy, r in inflated_obstacles:
                obs = np.array([ox, oy], dtype=np.float32)

                left_clear = np.linalg.norm(obs - left_pos) - r
                right_clear = np.linalg.norm(obs - right_pos) - r

                left_score += min(left_clear, 0.0)
                right_score += min(right_clear, 0.0)

        offset = 0.0

        for ox, oy, r in inflated_obstacles:
            d = np.array([ox, oy], dtype=np.float32) - p
            dist = np.linalg.norm(d)

            if dist > influence_radius_m:
                continue

            lateral_dist = np.dot(d, n)
            penetration = r - abs(lateral_dist)

            if penetration > 0.0:
                # prefer side with better corridor feasibility
                if left_score > right_score:
                    direction = 1.0
                elif right_score > left_score:
                    direction = -1.0
                else:
                    direction = -np.sign(lateral_dist) if lateral_dist != 0 else 1.0

                strength = penetration / r
                offset += direction * strength * (r + 0.2)

        lateral_offsets[i] = offset

    lateral_offsets = np.clip(
        lateral_offsets,
        -max_lateral_shift_m,
        max_lateral_shift_m
    )

    if smoothing_window >= 3 and smoothing_window % 2 == 1:
        kernel = np.ones(smoothing_window, dtype=np.float32)
        kernel /= kernel.sum()
        lateral_offsets = np.convolve(lateral_offsets, kernel, mode="same")

    local_path = np.zeros_like(centerline)
    for i in range(N):
        _, n = tangent_normal(i)
        local_path[i] = centerline[i] + lateral_offsets[i] * n

    return local_path


def compute_path_difficulty(
    obstacles: list[tuple[float, float, float]],
    centerline: NDArray[np.float32],
    lane_half_width_m: float,
    vehicle_width_m: float,
) -> tuple[NDArray[np.float32], bool, "float | None"]:
    """
    Computes a per-point difficulty score along the path based on available corridor width.

    At each sampled cross-section, finds the widest free lateral gap within the lane
    after accounting for nearby obstacle footprints. Difficulty scales linearly from
    0% (free_width = lane_width) to 100% (free_width ≤ vehicle_width).

    Parameters
    ----------
    obstacles : list of (x, y, radius) tuples in metres
    centerline : (N, 2) array of path points in metres
    lane_half_width_m : half the lane width (corridor searched in [-L, +L])
    vehicle_width_m : minimum passage width required for the vehicle

    Returns
    -------
    difficulty_per_point : float32 array of shape (N_samples,), values in [0, 1]
    is_feasible : True if no cross-section is completely blocked (difficulty < 1.0)
    first_blocking_x_m : world-x of the first fully-blocked cross-section, or None
    """
    if not obstacles:
        return np.zeros(1, dtype=np.float32), True, None

    N = len(centerline)
    n_samples = min(200, N)
    sample_indices = np.linspace(0, N - 1, n_samples, dtype=int)

    lane_width = 2.0 * lane_half_width_m
    difficulty_arr = np.zeros(n_samples, dtype=np.float32)
    is_feasible = True
    first_blocking_x: "float | None" = None

    for si, idx in enumerate(sample_indices):
        p = centerline[idx]

        # Path tangent and left normal
        i0 = min(int(idx), N - 2)
        t = centerline[i0 + 1] - centerline[i0]
        norm = np.linalg.norm(t)
        if norm < 1e-6:
            continue
        t = t / norm
        n = np.array([-t[1], t[0]], dtype=np.float32)  # left normal

        # Collect blocked lateral intervals from nearby obstacles
        blocked: list[tuple[float, float]] = []
        for ox, oy, r in obstacles:
            d = np.array([ox, oy], dtype=np.float32) - p
            lon = float(np.dot(d, t))
            if abs(lon) > r + 1.0:
                continue  # not near this cross-section
            lat = float(np.dot(d, n))
            lo = max(lat - r, -lane_half_width_m)
            hi = min(lat + r, lane_half_width_m)
            if hi > lo:
                blocked.append((lo, hi))

        # Find maximum free gap in [-lane_half_width, +lane_half_width]
        if not blocked:
            max_free_gap = lane_width
        else:
            blocked.sort()
            merged: list[list[float]] = []
            for lo, hi in blocked:
                if not merged or lo > merged[-1][1]:
                    merged.append([lo, hi])
                else:
                    merged[-1][1] = max(merged[-1][1], hi)

            cursor = -lane_half_width_m
            max_free_gap = 0.0
            for lo, hi in merged:
                gap = lo - cursor
                if gap > max_free_gap:
                    max_free_gap = gap
                cursor = max(cursor, hi)
            # Trailing gap
            gap = lane_half_width_m - cursor
            if gap > max_free_gap:
                max_free_gap = gap

        # Map free gap to difficulty in [0, 1]
        denom = lane_width - vehicle_width_m
        if denom <= 0.0:
            difficulty = 1.0
        else:
            difficulty = float(np.clip(
                1.0 - (max_free_gap - vehicle_width_m) / denom, 0.0, 1.0
            ))

        difficulty_arr[si] = difficulty

        if difficulty >= 1.0 and is_feasible:
            is_feasible = False
            first_blocking_x = float(p[0])

    return difficulty_arr, is_feasible, first_blocking_x


class ObstacleMixin:
    """
    Mixin providing obstacle generation, local path planning, and obstacle-aware reward.
    Does NOT touch the observation space — compose with a BEV or lidar obs class.

    Reward structure (obstacle environments only)
    ---------------------------------------------
    Terminal failure:
        reward = progress × _MAX_FAIL_REWARD × exp_scale(difficulty)

        exp_scale(d) = (e^(k·d) - 1) / (e^k - 1),  k = _DIFFICULTY_K
        Maps difficulty ∈ [0, 1] → reward scale ∈ [0, 1] with a convex
        (exponential) curve — small bonus for easy obstacles, full bonus for
        impassable ones.

    Deliberate stop (speed < _STOP_SPEED_TOL for _STOP_SECONDS):
        Same formula but multiplied by _STOP_REWARD_BONUS (> 1) to make
        controlled stopping strictly more valuable than crashing.

    Per-step bonus:
        slow_bonus = difficulty × (1 − |speed| / max_speed) × _SLOW_REWARD_SCALE
        Rewards gradual deceleration as obstacles become harder without
        dominating the base path-tracking reward.

    Override _MAX_FAIL_REWARD in subclasses to match the scale of the
    environment's success reward (default 50 ≈ half of forward success=100).
    """

    # --- Tunable reward constants (override in subclasses as needed) ---
    _DIFFICULTY_K: float = 3.0       # exponential steepness; higher = more curved
    _MAX_FAIL_REWARD: float = 50.0   # max failure reward at full difficulty + progress
    _STOP_REWARD_BONUS: float = 1.5  # stop reward = _MAX_FAIL_REWARD × this factor
    _SLOW_REWARD_SCALE: float = 0.1  # per-step slow bonus at max difficulty

    # --- Stop-detection constants ---
    _STOP_SPEED_TOL: float = 0.1     # |xd| below this counts as stopped (m/s)
    _STOP_SECONDS: float = 1.0       # consecutive seconds near-zero to trigger stop

    def __init__(self, render_mode=None, max_episode_steps=1000, **kwargs):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

        if not hasattr(self, "obstacles_low"):
            self.obstacles_low = 0
        if not hasattr(self, "obstacles_high"):
            self.obstacles_high = 0

        self._stopped_steps: int = 0
        self.stopped: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exp_scale(self, difficulty: float) -> float:
        """Maps difficulty ∈ [0, 1] → reward scale ∈ [0, 1] via (e^kd-1)/(e^k-1)."""
        k = self._DIFFICULTY_K
        return float((np.exp(k * difficulty) - 1.0) / (np.exp(k) - 1.0))

    # ------------------------------------------------------------------
    # Gymnasium step override — stop detection
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        difficulty = getattr(self, 'path_difficulty', 0.0)

        if not terminated and not truncated and difficulty > 0.0:
            if abs(self.vehicle.xd) < self._STOP_SPEED_TOL:
                self._stopped_steps += 1
            else:
                self._stopped_steps = 0

            steps_for_stop = max(1, round(self._STOP_SECONDS / self.vehicle.dt))
            if self._stopped_steps >= steps_for_stop:
                self.stopped = True
                terminated = True
                exp = self._exp_scale(difficulty)
                progress = self._get_progress_fraction()
                reward = progress * self._MAX_FAIL_REWARD * self._STOP_REWARD_BONUS * exp

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(self):
        error, error_theta = self.get_vehicle_errors(xx=self.local_path[:, 0], yy=self.local_path[:, 1])
        error_t, error_theta_t = self.get_trailer_errors(xx=self.local_path[:, 0], yy=self.local_path[:, 1])
        base = super().get_reward(error, error_theta, error_t, error_theta_t)

        difficulty = getattr(self, 'path_difficulty', 0.0)

        # --- Terminal: replace failure penalty with exponential difficulty reward ---
        if self._get_term() and not getattr(self, 'success', False) and difficulty > 0.0:
            exp = self._exp_scale(difficulty)
            return self._get_progress_fraction() * self._MAX_FAIL_REWARD * exp

        # --- Step: bonus for slow speed proportional to current difficulty ---
        if not self._get_term() and difficulty > 0.0:
            max_speed = max(abs(config.initial_xd), 1e-6)
            normalized_speed = np.clip(abs(self.vehicle.xd) / max_speed, 0.0, 1.0)
            slow_bonus = difficulty * (1.0 - normalized_speed) * self._SLOW_REWARD_SCALE
            return base + slow_bonus

        return base

    def _render_frame(self, surface=None):
        if surface is None:
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        surface = surface if surface else self.canvas
        super()._render_frame(surface)
        self.render_path(surface)
        self.render_path(surface, xx=self.local_path[:, 0], yy=self.local_path[:, 1], color=(255, 0, 0))
        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def generate_path(self):
        super().generate_path()

        # Reset stop-detection state on every new episode
        self._stopped_steps = 0
        self.stopped = False

        num_obstacles = 0
        if self.obstacles_high >= 1:
            num_obstacles = int(self.np_random.integers(self.obstacles_low, self.obstacles_high))

        centerline = self._sample_centerline()
        self.occ_grid, obstacles = generate_obstacles(
            centerline, self.occ_grid, num_obstacles, rng=self.np_random
        )
        original_path = np.stack([self.xx, self.yy], axis=1)
        lane_hw = (
            getattr(config, "lane_centerline_half_width_m", 1.75)
            + getattr(config, "lane_shoulder_m", 0.50)
        )
        vw = max(TRACTOR_WIDTH, TRAILER_WIDTH)
        if obstacles:
            self.local_path = plan_local_path(
                centerline=original_path,
                obstacles=obstacles,
                lane_half_width_m=lane_hw,
                vehicle_width_m=vw,
            )
            self._difficulty_profile, self.feasible, self.blockage_x = compute_path_difficulty(
                obstacles=obstacles,
                centerline=original_path,
                lane_half_width_m=lane_hw,
                vehicle_width_m=vw,
            )
            self.path_difficulty = float(np.max(self._difficulty_profile))
        else:
            self.local_path = centerline
            self._difficulty_profile = np.zeros(1, dtype=np.float32)
            self.feasible = True
            self.blockage_x = None
            self.path_difficulty = 0.0

        self._occ_dirty = True
        self._ensure_occ_surface_if_needed()
        self.obstacle_mask = pygame.mask.from_surface(self._occ_surface)


class ObstacleAvoidance(ObstacleMixin):
    """
    ObstacleMixin + lidar observation. Appends normalized lidar distances to the
    Box observation produced by the base state env.
    """

    def __init__(self, render_mode=None, max_episode_steps=1000, **kwargs):
        if not hasattr(self, "lidar_beams"):
            self.lidar_beams = 24
        if not hasattr(self, "lidar_range"):
            self.lidar_range = 20.0

        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

        lidar_low = np.zeros(self.lidar_beams, dtype=np.float32)
        lidar_high = np.ones(self.lidar_beams, dtype=np.float32)

        self.obs_low = np.concatenate([self.obs_low, lidar_low])
        self.obs_high = np.concatenate([self.obs_high, lidar_high])

        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32
        )
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_lidar_pose(self):
        """Return Pose(x, y, yaw) for lidar origin."""
        raise NotImplementedError

    def _get_obs(self):
        observation = super()._get_obs()

        lidar_pose = self._get_lidar_pose()
        obstacle_distances = get_obstacle_distances(
            self.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams,
        )
        return np.concatenate((observation, obstacle_distances.astype(np.float32)))


class LidarStateObservationLineFollowingEnv(StateObservationLineFollowingEnv):
    """
    Lane-following env with state + simulated lidar observations.

    Designed for fair Phase 1 comparison against state_only and BEV variants:
      - Extends StateObservationLineFollowingEnv directly (not ObstacleAvoidanceEnv)
      - Same path generation, reward function, and termination as state_only
      - Same 8-dim state vector; lidar distances are appended on top
      - Lidar reads from the lane occupancy grid (detects lane boundaries only)
      - No obstacles generated; no local_path override of reward

    Observation: [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂, d₀, …, d_{N-1}]
    """

    def __init__(self, render_mode="human", max_episode_steps=1000, lidar_beams=16, reward_mode: str = "dense"):
        self.lidar_beams = lidar_beams
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
        )

        lidar_low  = np.zeros(self.lidar_beams, dtype=np.float32)
        lidar_high = np.ones(self.lidar_beams,  dtype=np.float32)

        self.obs_low  = np.concatenate([self.obs_low,  lidar_low])
        self.obs_high = np.concatenate([self.obs_high, lidar_high])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32
        )
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_obs(self):
        state_obs = super()._get_obs()   # 8-dim from StateObservationLineFollowingEnv

        lidar_pose = Pose(
            x=self.vehicle.x,
            y=self.vehicle.y,
            yaw=self.vehicle.p,
        )
        lidar_distances = get_obstacle_distances(
            self.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams,
        )
        self.observation = np.concatenate(
            [state_obs, lidar_distances.astype(np.float32)]
        )
        return self.observation


class ObstacleAvoidanceEnv(ObstacleAvoidance, StateObservationLineFollowingEnv):
    def __init__(self, render_mode=None, max_episode_steps=1000, reward_mode: str = "dense"):
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.lidar_beams = 24
        self.lidar_range = 20.0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=False,
        )

    def _get_lidar_pose(self):
        return Pose(
            x=self.vehicle.x,
            y=self.vehicle.y,
            yaw=self.vehicle.p
        )


class ReverseObstacleAvoidanceEnv(ObstacleAvoidance, ReverseStateObservationLineFollowingEnv):
    _MAX_FAIL_REWARD: float = 100.0  # matches reverse success reward scale (200)

    def __init__(self, render_mode=None, max_episode_steps=1000, reward_mode: str = "dense"):
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.lidar_beams = 24
        self.lidar_range = 20.0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=False,
        )

    def _get_lidar_pose(self):
        return Pose(
            x=self.vehicle.trailer.x,
            y=self.vehicle.trailer.y,
            yaw=self.vehicle.trailer.yaw + np.pi
        )


class BevObstacleAvoidanceEnv(ObstacleMixin, BevObservationLineFollowingEnv):
    """
    Forward lane-following with obstacles + BEV image observation.
    ObstacleMixin handles path generation, local path planning, and reward.
    BevObservationLineFollowingEnv handles the Dict (image + vector) observation.
    """

    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense"):
        self.obstacles_low = 0
        self.obstacles_high = 0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=False,
        )


class ReverseBevObstacleAvoidanceEnv(ObstacleMixin, ReverseBevObservationLineFollowingEnv):
    """
    Reverse lane-following with obstacles + BEV image observation.
    ObstacleMixin handles path generation, local path planning, and reward.
    ReverseBevObservationLineFollowingEnv handles reverse dynamics and Dict obs.
    """

    _MAX_FAIL_REWARD: float = 100.0  # matches reverse success reward scale (200)

    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense"):
        self.obstacles_low = 0
        self.obstacles_high = 0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=False,
        )
