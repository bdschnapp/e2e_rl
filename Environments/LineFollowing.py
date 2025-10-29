import numpy as np
import pygame
from scipy.interpolate import CubicSpline
from gymnasium import spaces
from dataclasses import dataclass
from scipy.spatial import cKDTree

from Environments.TractorTrailer import TractorTrailerEnv, WINDOW_WIDTH, WINDOW_HEIGHT, METERS_PER_PIXEL, COLOR_WHITE, \
    COLOR_BLACK
import e2erl_utils.config as config

@dataclass
class GridMeta:
    origin_x: float        # world x at grid(0,0)
    origin_y: float        # world y at grid(0,0)
    res_m: float           # meters per cell
    width: int             # cells
    height: int            # cells

class LineFollowingEnv(TractorTrailerEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)

        # path variables
        self.xx = np.array([])
        self.yy = np.array([])
        self.generate_path()  # ensure that a path is always initialized

        # redefine box observation space to add cross track error and angle error
        self.box_observation_space = spaces.Box(low=np.array([-config.steering_observation,
                                                              -config.hitch_angle_observation,
                                                              -config.cross_track_distance_observation,
                                                              -config.cross_track_angle_observation]),
                                                high=np.array([config.steering_observation,
                                                               config.hitch_angle_observation,
                                                               config.cross_track_distance_observation,
                                                               config.cross_track_angle_observation]),
                                                dtype=np.float32)

        self.observation_space = spaces.Dict({
            'vector': self.box_observation_space,
            'image': self.image_observation_space
        })

    def _get_reward(self):
        if self._get_term():
            return -10.0
        error, error_theta = self.get_errors()
        return 5 * np.exp(-abs(error)) * np.exp(-abs(error_theta)) - (self.vehicle.xd / 5)

    def _get_obs(self):
        obs_dict = super()._get_obs()  # handle the image observation
        error, error_theta = self.get_errors()
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        observation = np.array([self.vehicle.s,
                                hitch_angle,
                                error,
                                error_theta], dtype=np.float32)
        obs_dict['vector'] = observation
        return obs_dict

    def _get_trunc(self):
        # check out of bounds
        trunc = super()._get_trunc()
        return trunc

    def _get_term(self):
        # check out of bounds or collision
        term = super()._get_term()

        # check if trailer is jackknifed
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        term = term or (abs(hitch_angle) > np.pi / 2)

        return term

    def get_errors(self):
        # find the closest point on the path to the trailer axle
        errors = np.sqrt((self.xx - self.vehicle.trailer.x) ** 2 + (self.yy - self.vehicle.trailer.y) ** 2)
        nearest_index = np.argmin(errors)
        nearest_y = self.yy[nearest_index]

        # calculate cross track distance error
        error = errors[nearest_index]
        if nearest_y > self.vehicle.trailer.y:
            error = -1 * error

        # calculate cross track angle error
        try:
            theta = np.arctan((self.yy[nearest_index + 1] - self.yy[nearest_index])
                              / (self.xx[nearest_index + 1] - self.xx[nearest_index]))
        except IndexError:
            theta = np.arctan((self.yy[nearest_index] - self.yy[nearest_index - 1])
                              / (self.xx[nearest_index] - self.xx[nearest_index - 1]))
        error_theta = (self.vehicle.trailer.yaw - theta) * config.error_theta_scale

        return error, error_theta

    def generate_path(self):
        x0 = self.vehicle.trailer.x
        x = np.array([x0, 30, 45, 60, 75, 90])

        a = -1
        b = 1
        n = 6

        vert_offset = 45

        y = (a + (b - a) * np.random.rand(n))

        # Have it start so the vehicle isn't immediately on the line, y(0) of vehicle will always be vert_offset
        y[0] = (-0.25 / 50) + np.random.rand() * (0.5 / 50)

        # self.xx is the x values for the cubic spline
        self.xx = np.arange(int(x0), 90, 1)

        cs = CubicSpline(x, y, bc_type=((1, 0.0), 'not-a-knot'))
        # self.yy is the y values for the cubic spline
        self.yy = cs(self.xx) * 5 + vert_offset

        x_g = self.xx[-1]
        y_g = self.yy[-1]
        yaw_g = np.arctan((self.yy[-1] - self.yy[-2]) / (self.xx[-1] - self.xx[-2]))
        self.goal_pose = (x_g, y_g, yaw_g)

    def render_path(self, surface):
        for i in range(len(self.xx) - 1):
            x1 = int(self.xx[i] / METERS_PER_PIXEL)
            y1 = int(WINDOW_HEIGHT - self.yy[i] / METERS_PER_PIXEL)
            x2 = int(self.xx[i + 1] / METERS_PER_PIXEL)
            y2 = int(WINDOW_HEIGHT - self.yy[i + 1] / METERS_PER_PIXEL)
            pygame.draw.line(surface, COLOR_BLACK, (x1, y1), (x2, y2), 2)

    def _render_frame(self, surface=None):
        # initialize pygame if it hasn't been already
        if surface is None:
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        surface = surface if surface else self.canvas

        # fill the background
        surface.fill(COLOR_WHITE)

        # render the vehicle
        super()._render_frame(surface)

        # render the path on top of the vehicle
        self.render_path(surface)

        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def step(self, action):
        # override the action to use a constant speed
        # action = np.array([action[0], -1 * config.initial_xd])
        return super().step(action)

    def reset(self, seed=None, options=None):
        self.vehicle.reset(-1 * config.initial_xd, x=0.1, y=45, p=-np.pi)
        self.generate_path()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


class StateObservationLineFollowingEnv(LineFollowingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)

        # redefine observation space
        self.observation_space = spaces.Box(low=np.array([-config.steering_observation,
                                                          -config.hitch_angle_observation,
                                                          -config.cross_track_distance_observation,
                                                          -config.cross_track_angle_observation]),
                                            high=np.array([config.steering_observation,
                                                           config.hitch_angle_observation,
                                                           config.cross_track_distance_observation,
                                                           config.cross_track_angle_observation]),
                                            dtype=np.float32)
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_obs(self):
        error, error_theta = self.get_errors()
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        self.observation = np.array([self.vehicle.s,
                                     hitch_angle,
                                     error,
                                     error_theta], dtype=np.float32)
        return self.observation


class LaneDrivingEnv(LineFollowingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)
        self.occ_grid = None  # np.uint8 [H,W], 0=free, 100=blocked
        self.occ_meta: GridMeta | None = None
        self._build_occupancy_grid()  # build once on init

    def _world_bounds(self):
        """World bounds derived from the current global window constants."""
        world_w_m = WINDOW_WIDTH * METERS_PER_PIXEL
        world_h_m = WINDOW_HEIGHT * METERS_PER_PIXEL
        # We’ll use a world-aligned grid whose (0,0) is the bottom-left of the window
        x_min, y_min = 0.0, 0.0
        x_max, y_max = world_w_m, world_h_m
        return x_min, y_min, x_max, y_max

    def _sample_centerline(self):
        """
        Resample the existing spline (self.xx, self.yy) at approximately uniform arclength.
        Returns Nx2 array of (x, y) in world meters.
        """
        # compute cumulative arclength on existing piecewise-linear polyline
        dx = np.diff(self.xx)
        dy = np.diff(self.yy)
        seg_len = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = s[-1]
        if total <= 0.0:
            return np.vstack([self.xx, self.yy]).T

        ds = getattr(config, "lane_sample_ds_m", 0.25)
        new_s = np.arange(0.0, total, ds)
        # interpolate x(s), y(s) linearly over the polyline parameter
        xs = np.interp(new_s, s, self.xx)
        ys = np.interp(new_s, s, self.yy)
        return np.stack([xs, ys], axis=1)

    def _grid_from_bounds(self, x_min, y_min, x_max, y_max, res_m):
        width = int(np.ceil((x_max - x_min) / res_m))
        height = int(np.ceil((y_max - y_min) / res_m))
        meta = GridMeta(origin_x=x_min, origin_y=y_min, res_m=res_m, width=width, height=height)
        return meta

    def _cell_centers_world(self, meta: 'GridMeta'):
        """
        Return (Xc, Yc) 2D arrays of cell-center coordinates in world meters for the entire grid.
        Note: y increases upward in world; later we’ll remember that the screen Y is inverted.
        """
        gx = (np.arange(meta.width) + 0.5) * meta.res_m + meta.origin_x
        gy = (np.arange(meta.height) + 0.5) * meta.res_m + meta.origin_y
        Xc, Yc = np.meshgrid(gx, gy, indexing='xy')
        return Xc, Yc

    def _build_occupancy_grid(self):
        """
        Build lane-style occupancy once (or when path changes).
        Strategy: nearest-distance of each grid cell center to the resampled centerline.
        Cells within lane_half_width + shoulder are 'free' (0), else 'blocked' (100).
        """
        x_min, y_min, x_max, y_max = self._world_bounds()
        res = getattr(config, "grid_res_m", 0.10)
        meta = self._grid_from_bounds(x_min, y_min, x_max, y_max, res)

        # Sample centerline points
        centerline = self._sample_centerline()  # [N,2] in meters

        # KDTree for nearest distance queries
        kdt = cKDTree(centerline)

        # Evaluate distance per grid cell (vectorized)
        Xc, Yc = self._cell_centers_world(meta)
        pts = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
        dists, _ = kdt.query(pts, k=1, workers=-1)  # nearest distance in meters

        lane_half = getattr(config, "lane_centerline_half_width_m", 1.75)
        shoulder = getattr(config, "lane_shoulder_m", 0.50)
        lane_radius = lane_half + shoulder

        free_mask = dists.reshape(meta.height, meta.width) <= lane_radius

        # Initialize grid: blocked=100 everywhere, then set free cells to 0
        grid = np.full((meta.height, meta.width), 100, dtype=np.uint8)
        grid[free_mask] = 0

        # Cache
        self.occ_grid = grid
        self.occ_meta = meta

    # Convenience hooks for later phases
    def get_occupancy_grid(self):
        """Return a (grid copy, meta) so callers don't mutate the cache by mistake."""
        return self.occ_grid.copy() if self.occ_grid is not None else None, self.occ_meta

    def world_to_grid(self, x, y):
        """Map world meters → integer grid indices (gx, gy). No bounds checking here."""
        meta = self.occ_meta
        gx = int((x - meta.origin_x) / meta.res_m)
        gy = int((y - meta.origin_y) / meta.res_m)
        return gx, gy



def main():
    from controllers.mpc import TractorTrailerSteeringMPC
    from controllers.mpc_traj_gen import generate_trajectory
    mpc = TractorTrailerSteeringMPC()
    env = LineFollowingEnv(render_mode='human')
    env.reset()
    done = False
    action = env.action_space.sample()

    while not done:
        trajectory = generate_trajectory(env.xx, env.yy, env.vehicle)
        state = (env.vehicle.xd, env.vehicle.s)

        u = -mpc.solve(trajectory, state)  # get target steering angle from MPC
        s = env.vehicle.s  # get current vehicle steering angle
        ds_dt = (u - s) / env.vehicle.dt  # convert steering angle error to steering rate
        action[0] = ds_dt
        action[1] = -1 * config.initial_xd  # constant speed

        print(f'S: {s}, U: {u}')
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        env.render()

    env.close()


if __name__ == "__main__":
    main()
