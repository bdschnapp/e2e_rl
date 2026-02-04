import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from dataclasses import dataclass
import pygame

import e2erl_utils.config as config

from Environments.LineFollowing import StateObservationLineFollowingEnv, ReverseStateObservationLineFollowingEnv
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
    bin_order = np.random.permutation(len(bins))

    obstacles: list[tuple[float, float, float]] = []
    for k in range(num_obstacles):
        bi = bin_order[k % len(bins)]
        a, b = bins[bi]

        # Sample an index within this bin
        idx = np.random.randint(a, b)
        p = centerline[idx]
        p_next = centerline[idx + 1]

        # Tangent (lane direction)
        t = p_next - p
        t /= np.linalg.norm(t) + 1e-6

        # Normal (left/right)
        n = np.array([-t[1], t[0]])

        # Random lateral offset (left or right)
        lat_mag = np.random.uniform(*lateral_offset_range_m)
        lat_sign = np.random.choice([-1.0, 1.0])
        lateral_offset = lat_sign * lat_mag

        # Optional longitudinal jitter
        longitudinal_offset = np.random.uniform(
            -longitudinal_jitter_m,
            longitudinal_jitter_m
        )

        # Final obstacle position
        pos = p + lateral_offset * n + longitudinal_offset * t

        # Random obstacle size
        radius_m = np.random.uniform(*radius_range_m)
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


class ObstacleAvoidance:
    def __init__(self, render_mode=None, max_episode_steps=1000, **kwargs):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

        if not hasattr(self, "obstacles_low"):
            self.obstacles_low = 0
        if not hasattr(self, "obstacles_high"):
            self.obstacles_high = 0

        if not hasattr(self, "lidar_beams"):
            self.lidar_beams = 24
        if not hasattr(self, "lidar_range"):
            self.lidar_range = 20.0

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

    def _get_reward(self):
        error, error_theta = self.get_vehicle_errors(xx=self.local_path[:, 0], yy=self.local_path[:, 1])
        error_t, error_theta_t = self.get_trailer_errors(xx=self.local_path[:, 0], yy=self.local_path[:, 1])
        return super().get_reward(error, error_theta, error_t, error_theta_t)

    # def _render_frame(self, surface=None):
    #     # initialize pygame if it hasn't been already
    #     if surface is None:
    #         if self.canvas is None:
    #             pygame.init()
    #             self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    #
    #     surface = surface if surface else self.canvas
    #
    #     # render the vehicle
    #     super()._render_frame(surface)
    #
    #     # render the path on top of the vehicle
    #     self.render_path(surface)
    #     # render the local path
    #     self.render_path(surface, xx=self.local_path[:, 0], yy=self.local_path[:, 1], color=(255, 0, 0))
    #
    #     return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def generate_path(self):
        super().generate_path()

        num_obstacles = 0
        if self.obstacles_high >= 1:
            # NOTE: randint high is exclusive; keep your behavior
            num_obstacles = np.random.randint(self.obstacles_low, self.obstacles_high)

        centerline = self._sample_centerline()
        self.occ_grid, obstacles = generate_obstacles(centerline, self.occ_grid, num_obstacles)
        original_path = np.stack([self.xx, self.yy], axis=1)
        if obstacles:
            self.local_path = plan_local_path(
                centerline=original_path,
                obstacles=obstacles,
                lane_half_width_m= getattr(config, "lane_centerline_half_width_m", 1.75) +
                                   getattr(config, "lane_shoulder_m", 0.50),
                vehicle_width_m=max(TRACTOR_WIDTH, TRAILER_WIDTH)
            )
        else:
            self.local_path = centerline

        self._occ_dirty = True
        self._ensure_occ_surface_if_needed()
        self.obstacle_mask = pygame.mask.from_surface(self._occ_surface)


class ObstacleAvoidanceEnv(ObstacleAvoidance, StateObservationLineFollowingEnv):
    def __init__(self, render_mode=None, max_episode_steps=1000):
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.lidar_beams = 24
        self.lidar_range = 20.0
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)

    def _get_lidar_pose(self):
        return Pose(
            x=self.vehicle.x,
            y=self.vehicle.y,
            yaw=self.vehicle.p
        )


class ReverseObstacleAvoidanceEnv(ObstacleAvoidance, ReverseStateObservationLineFollowingEnv):
    def __init__(self, render_mode=None, max_episode_steps=1000):
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.lidar_beams = 24
        self.lidar_range = 20.0
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)

    def _get_lidar_pose(self):
        return Pose(
            x=self.vehicle.trailer.x,
            y=self.vehicle.trailer.y,
            yaw=self.vehicle.trailer.yaw + np.pi
        )
