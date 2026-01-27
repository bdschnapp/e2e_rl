import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import pygame

from Environments.LineFollowing import StateObservationLineFollowingEnv, ReverseStateObservationLineFollowingEnv
from Environments.TractorTrailer import WINDOW_HEIGHT, WINDOW_WIDTH, METERS_PER_PIXEL


@dataclass
class LidarPose:
    x: float    # meters
    y: float    # meters
    yaw: float  # radians


def get_obstacle_distances(
    occ_grid: NDArray[np.uint8],
    lidar_pose: LidarPose,
    *,
    num_sensors: int = 6,
    fov: float = np.deg2rad(280.0),
    max_range_m: float = 10.0,
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
    lateral_offset_range_m=(0.5, 2.0),
    radius_range_m=(0.25, 0.6),
    longitudinal_jitter_m=0.5,
):
    """
    Generates circular obstacles near the lane centerline.

    centerline: (N,2) array in meters
    occ_grid: occupancy grid to write into (modified in-place)
    """

    H, W = occ_grid.shape
    occ_grid.fill(0)

    for _ in range(num_obstacles):
        # Pick a segment along the centerline
        idx = np.random.randint(0, len(centerline) - 1)
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
        pos = (
            p
            + lateral_offset * n
            + longitudinal_offset * t
        )

        # Random obstacle size
        radius_m = np.random.uniform(*radius_range_m)
        radius_px = int(radius_m / METERS_PER_PIXEL)

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
                    occ_grid[y, x] = 255

    return occ_grid


class ObstacleAvoidanceEnv(StateObservationLineFollowingEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.lidar_beams = 6
        self.lidar_range = 10.0  # meters

    """
    def _get_obs(self):
        # Get the original observation from the parent class
        observation = super()._get_obs()

        # Add obstacle distance sensors to the observation
        lidar_pose = LidarPose(x=self.vehicle.x, y=self.vehicle.y, yaw=self.vehicle.p)
        obstacle_distances = get_obstacle_distances(
            self.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams
        )

        observation = np.concatenate((observation, obstacle_distances))
        return observation
    """

    def generate_path(self):
        super().generate_path()
        num_obstacles = np.random.randint(5, 15)
        centerline = self._sample_centerline()
        self.occ_grid = generate_obstacles(centerline, self.occ_grid, num_obstacles)
        self._occ_dirty = True
        self._ensure_occ_surface_if_needed()
        self.obstacle_mask = pygame.mask.from_surface(self._occ_surface)


class ReverseObstacleAvoidanceEnv(ReverseStateObservationLineFollowingEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_obs(self):
        # Get the original observation from the parent class
        observation = super()._get_obs()

        # Add obstacle distance sensors to the observation
        lidar_pose = LidarPose(x=self.vehicle.trailer.x, y=self.vehicle.trailer.y, yaw=self.vehicle.trailer.yaw)
        obstacle_distances = get_obstacle_distances(self.occ_grid, lidar_pose)

        observation = np.concatenate((observation, obstacle_distances))
        return observation

    def generate_path(self):
        super().generate_path()
        num_obstacles = np.random.randint(5, 15)
        centerline = self._sample_centerline()
        self.occ_grid = generate_obstacles(centerline, self.occ_grid, num_obstacles)
        self._occ_dirty = True
        self._ensure_occ_surface_if_needed()
        self.obstacle_mask = pygame.mask.from_surface(self._occ_surface)
