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
        errors = np.sqrt((self.xx - self.vehicle.x) ** 2 + (self.yy - self.vehicle.y) ** 2)
        nearest_index = np.argmin(errors)
        nearest_y = self.yy[nearest_index]

        # calculate cross track distance error
        error = errors[nearest_index]
        if nearest_y > self.vehicle.y:
            error = -1 * error

        # calculate cross track angle error
        try:
            theta = np.arctan((self.yy[nearest_index + 1] - self.yy[nearest_index])
                              / (self.xx[nearest_index + 1] - self.xx[nearest_index]))
        except IndexError:
            theta = np.arctan((self.yy[nearest_index] - self.yy[nearest_index - 1])
                              / (self.xx[nearest_index] - self.xx[nearest_index - 1]))
        error_theta = (self.vehicle.p - theta) * config.error_theta_scale

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
        # surface.fill(COLOR_WHITE)

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

        # --- Phase 2: render cache ---
        self._occ_surface = None  # pygame.Surface aligned to world extents
        self._occ_dirty = True  # set True whenever grid changes
        self._show_spline_debug = True  # toggle overlay of spline (thin line)

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
        self._occ_dirty = True

    def _grid_to_surface(self):
        """
        Convert self.occ_grid (uint8: 0=free,100=blocked) into a pygame.Surface matching WINDOW size.
        Palette: free (0) -> bright, blocked (100) -> dark.
        Notes:
          - We render the grid world-up, then flip to screen-down (pygame Y+ down).
          - We cache the scaled surface; re-create only when _occ_dirty is True.
        """
        if self.occ_grid is None:
            return None
        grid = self.occ_grid  # H×W, uint8: {0,100}
        gray = (255 - (grid.astype(np.uint16) * 215 // 100)).astype(np.uint8)
        gray_flipped = np.flipud(gray)
        rgb = np.dstack([gray_flipped] * 3)  # make it RGB
        surf_small = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf_small, (WINDOW_WIDTH, WINDOW_HEIGHT))
        return surf

    def _ensure_occ_surface_if_needed(self):
        """Create or refresh the cached occupancy surface if dirty."""
        if self._occ_dirty or (self._occ_surface is None):
            self._occ_surface = self._grid_to_surface()
            self._occ_dirty = False

    def generate_path(self):
        super().generate_path()
        self._build_occupancy_grid()

    def reset(self, seed=None, options=None):
        self.vehicle.reset(config.initial_xd, x=4.1, y=45, p=0)
        self.generate_path()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _render_frame(self, surface=None):
        # Ensure base canvas exists
        if surface is None:
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        world_canvas = surface if surface is not None else self.canvas
        self._ensure_occ_surface_if_needed()
        if self._occ_surface is not None:
            world_canvas.blit(self._occ_surface, (0, 0))
        else:
            world_canvas.fill(COLOR_WHITE)
        super()._render_frame(world_canvas)
        if not getattr(config, "use_bev_render", True):
            return np.transpose(np.array(pygame.surfarray.pixels3d(world_canvas)), axes=(1, 0, 2))
        xA, yA, yaw_cam = self._get_anchor_world()
        anchor_x_pix = xA / METERS_PER_PIXEL
        anchor_y_pix = WINDOW_HEIGHT - (yA / METERS_PER_PIXEL)
        target_cx = WINDOW_WIDTH // 2
        target_cy = WINDOW_HEIGHT // 2
        offset_x = target_cx - anchor_x_pix
        offset_y = target_cy - anchor_y_pix
        shifted_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        shifted_surface.fill((0, 0, 0))
        shifted_surface.blit(
            world_canvas,
            (offset_x, offset_y)
        )
        yaw_cam_deg = np.rad2deg(yaw_cam)
        if getattr(config, "bev_forward_up", True):
            rot_deg = yaw_cam_deg + 90.0
        else:
            rot_deg = yaw_cam_deg
        zoom_scale = getattr(config, "bev_zoom_scale", 1.5)
        bev_rotzoom = pygame.transform.rotozoom(shifted_surface, rot_deg, zoom_scale)
        final_canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        final_canvas.fill((0, 0, 0))
        bev_rect = bev_rotzoom.get_rect()
        bev_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        final_canvas.blit(bev_rotzoom, bev_rect.topleft)
        return np.transpose(np.array(pygame.surfarray.pixels3d(final_canvas)), axes=(1, 0, 2))

    def _get_anchor_world(self):
        """Return anchor point (xA, yA) and yaw psi for camera (tractor yaw)."""
        xA = self.vehicle.x
        yA = self.vehicle.y
        psi = -self.vehicle.p
        return xA, yA, psi


class StateObservationLaneDrivingEnv(LaneDrivingEnv):
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


def compute_curvature(env, lookahead_steps=10, max_curvature=0.3):
    """
    Estimate path curvature ahead of the trailer axle.
    Returns a signed curvature kappa (1/m).
    Positive means 'curve left', negative 'curve right'.
    """
    tx = env.vehicle.trailer.x
    ty = env.vehicle.trailer.y
    dx = env.xx - tx
    dy = env.yy - ty
    dist_sq = dx*dx + dy*dy
    nearest_idx = int(np.argmin(dist_sq))
    i = nearest_idx + lookahead_steps
    i = max(1, min(i, len(env.xx) - 2))
    x_im1, x_i, x_ip1 = env.xx[i-1], env.xx[i], env.xx[i+1]
    y_im1, y_i, y_ip1 = env.yy[i-1], env.yy[i], env.yy[i+1]
    x_p = (x_ip1 - x_im1) * 0.5
    y_p = (y_ip1 - y_im1) * 0.5
    x_pp = (x_ip1 - 2.0 * x_i + x_im1)
    y_pp = (y_ip1 - 2.0 * y_i + y_im1)
    denom = (x_p**2 + y_p**2)**1.5 + 1e-6
    kappa = (x_p * y_pp - y_p * x_pp) / denom  # signed curvature
    kappa = np.clip(kappa, -max_curvature, max_curvature)
    return kappa

def main():
    env = LaneDrivingEnv(render_mode='human')
    env.reset()
    done = False
    action = env.action_space.sample()
    k_y = 0.9
    k_theta = 4.0
    wheelbase = env.vehicle.trailer.L + env.vehicle.lr + env.vehicle.lf
    i = 0
    while not done:
        i += 1
        error, error_theta = env.get_errors()
        u_fb = -k_y * error - k_theta * error_theta
        kappa = compute_curvature(env)
        u_ff = np.arctan(wheelbase * kappa)
        u = u_ff + u_fb

        s = env.vehicle.s  # get current vehicle steering angle
        ds_dt = (u - s) / env.vehicle.dt  # convert steering angle error to steering rate
        action[0] = ds_dt
        action[1] = 3.0 # constant speed

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        if i % 1 == 0:
            print(f'S: {s}, U: {u}')
            env.render()

    env.close()


if __name__ == "__main__":
    main()
