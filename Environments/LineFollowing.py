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


def forward_pure_pursuit(env, render=True):
    env.reset()
    done = False
    action = env.action_space.sample()
    k_y = 1.2
    k_yi = 0.1
    k_theta = 2.4
    wheelbase = env.vehicle.lr + env.vehicle.lf
    MAX_STEER_RATE = 0.5  # rad/s
    i = 0
    i_error = 0.0
    total_reward = 0.0
    steps = 0
    while not done:
        i += 1
        error, error_theta = env.get_vehicle_errors()
        i_error += error * env.vehicle.dt
        i_error = np.clip(i_error, -0.5, 0.5)
        u_fb = -k_y * error - k_yi * i_error - k_theta * error_theta
        kappa = compute_curvature(env)
        u_ff = np.arctan(wheelbase * kappa)
        u = 0.85 * u_ff + 0.15 * u_fb

        s = env.vehicle.s  # get current vehicle steering angle

        # convert steering angle error to steering rate
        ds_dt = np.clip((u - s) / env.vehicle.dt, -MAX_STEER_RATE, MAX_STEER_RATE)
        # ds_dt = (u - s) / env.vehicle.dt
        action[0] = ds_dt
        action[1] = config.initial_xd  # constant speed

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        done = term or trunc
        if render:
            env.render()

    return total_reward, steps


def reverse_pure_pursuit(env, render=False):
    obs, _ = env.reset()
    done = False

    action = env.action_space.sample()

    # --------- Inner loop gains (fast): stabilize phi -> phi_ref ----------
    Kp = 3.0          # articulation angle gain
    Kd = 0.5          # articulation rate damping

    # --------- Outer loop gains (slow): trailer tracking -> phi_ref ----------
    K_y_t = 1.2  # lateral error gain (trailer)
    K_th_t = 2.4  # heading error gain (trailer)
    phi_ref_max = np.deg2rad(15)  # keep small so inner loop can stay stable

    # Conservative reverse speed
    v_rev = -1.0

    MAX_STEER_RATE = 0.25
    MAX_STEER_ANGLE = np.deg2rad(config.steering_action + 15)

    # Jackknife region for extra authority
    JACKKNIFE_SOFT = np.deg2rad(20)
    JACKKNIFE_HARD = np.deg2rad(60)

    prev_phi = 0.0

    total_reward, steps = 0.0, 0

    while not done:
        # --- Get errors (prefer trailer errors for reverse tracking) ---
        # You already have these in your env:
        error_t, error_theta_t = env.get_trailer_errors()

        # --- Outer loop: compute desired hitch angle (small) ---
        # Signs may need flipping depending on your error conventions.
        phi_ref = -(K_y_t * error_t) - (K_th_t * error_theta_t)
        phi_ref = float(np.clip(phi_ref, -phi_ref_max, phi_ref_max))

        # --- Current hitch angle ---
        phi = env.vehicle.p - env.vehicle.trailer.yaw
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        phi = np.clip(phi, -JACKKNIFE_HARD, JACKKNIFE_HARD)

        # Hitch rate
        phi_dot = (phi - prev_phi) / env.vehicle.dt
        prev_phi = phi

        # --- Inner loop: stabilize phi around phi_ref ---
        e_phi = (phi - phi_ref)

        gain_boost = 1.0
        if abs(phi) > JACKKNIFE_SOFT:
            gain_boost = 1.0 + 2.0 * (abs(phi) - JACKKNIFE_SOFT) / (JACKKNIFE_HARD - JACKKNIFE_SOFT)
            gain_boost = np.clip(gain_boost, 1.0, 3.0)

        delta_cmd = gain_boost * (-(Kp * e_phi) - (Kd * phi_dot))
        delta_cmd = np.clip(delta_cmd, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

        # Convert steering angle to steering rate action
        s = env.vehicle.s
        ds_dt = np.clip((delta_cmd - s) / env.vehicle.dt, -MAX_STEER_RATE, MAX_STEER_RATE)

        action[0] = ds_dt
        action[1] = v_rev

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        done = term or trunc

        if render:
            env.render()

    return total_reward, steps


class LineFollowingEnv(TractorTrailerEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)

        # path variables
        self.xx = np.array([])
        self.yy = np.array([])
        self.generate_path()  # ensure that a path is always initialized
        self.success = False

        # max attempts to find collision-free spawn
        self.max_attempts = 100

        # redefine box observation space to add cross track error and angle error
        self.box_observation_space = spaces.Box(low=np.array([-config.steering_observation,
                                                              -config.hitch_angle_observation,
                                                              -config.cross_track_distance_observation,
                                                              -config.cross_track_angle_observation,
                                                              -config.cross_track_distance_observation,
                                                              -config.cross_track_angle_observation
                                                              ]),
                                                high=np.array([config.steering_observation,
                                                               config.hitch_angle_observation,
                                                               config.cross_track_distance_observation,
                                                               config.cross_track_angle_observation,
                                                               config.cross_track_distance_observation,
                                                               config.cross_track_angle_observation
                                                               ]),
                                                dtype=np.float32)

        self.observation_space = spaces.Dict({
            'vector': self.box_observation_space,
            'image': self.image_observation_space
        })

    def _get_reward(self):
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        return self.get_reward(error, error_theta, error_t, error_theta_t)


    def get_reward(self, error, error_theta, error_t, error_theta_t):
        if self._get_term():
            if self.success:
                return 200.0
            return -100.0
        # return 5 * np.exp(-abs(error)) * np.exp(-abs(error_theta)) - (self.vehicle.xd / 5)
        return (0.5 * self.vehicle.xd) - (error ** 2) - (error_theta ** 2) - ((0.5 * error_t) ** 2) - ((0.5 * error_theta_t) ** 2)

    def _get_obs(self):
        obs_dict = super()._get_obs()  # handle the image observation
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        observation = np.array([

            self.vehicle.s,
            hitch_angle,
            error,
            error_theta,
            error_t,
            error_theta_t
        ], dtype=np.float32)
        obs_dict['vector'] = observation
        return obs_dict

    def _get_trunc(self):
        # check out of bounds
        trunc = super()._get_trunc()
        return trunc

    def _get_term(self):
        # check out of bounds or collision
        term = super()._get_term()

        # check collision with obstacles (includes jackknife check)
        term = term or self._check_collision()

        env_len = (WINDOW_WIDTH * METERS_PER_PIXEL) * 0.85
        if self.vehicle.x > env_len or self.vehicle.trailer.x > env_len:
            self.success = True
            term = True

        return term

    def get_vehicle_errors(self, xx=None, yy=None):
        if xx is None or yy is None:
            xx = self.xx
            yy = self.yy
        return self.get_errors(self.vehicle.x, self.vehicle.y, self.vehicle.p, xx, yy)

    def get_trailer_errors(self, xx=None, yy=None):
        if xx is None or yy is None:
            xx = self.xx
            yy = self.yy
        return self.get_errors(self.vehicle.trailer.x, self.vehicle.trailer.y, self.vehicle.trailer.yaw, xx, yy)

    def get_errors(self, x, y, p, xx=None, yy=None):
        if xx is None or yy is None:
            xx = self.xx
            yy = self.yy

        # find the closest point on the path to the trailer axle
        errors = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        nearest_index = np.argmin(errors)
        nearest_y = yy[nearest_index]

        # calculate cross track distance error
        error = errors[nearest_index]
        if nearest_y > y:
            error = -1 * error

        # calculate cross track angle error
        try:
            theta = np.arctan((yy[nearest_index + 1] - yy[nearest_index])
                              / (xx[nearest_index + 1] - xx[nearest_index]))
        except IndexError:
            theta = np.arctan((yy[nearest_index] - yy[nearest_index - 1])
                              / (xx[nearest_index] - xx[nearest_index - 1]))
        error_theta = (p - theta) * config.error_theta_scale

        return error, error_theta

    def generate_path(self):
        # World width in meters (WINDOW_WIDTH * METERS_PER_PIXEL)
        world_width_m = WINDOW_WIDTH * METERS_PER_PIXEL

        # Path spans from near start to near end of world
        x0 = 5.0  # Start path 5m into the world
        x_end = world_width_m - 5.0  # End path 5m before world edge

        # Create evenly spaced control points
        x = np.linspace(x0, x_end, 6)

        a = -1
        b = 1
        n = 6

        vert_offset = 45

        y = (a + (b - a) * np.random.rand(n))

        # Have it start so the vehicle isn't immediately on the line, y(0) of vehicle will always be vert_offset
        y[0] = (-0.25 / 50) + np.random.rand() * (0.5 / 50)

        # self.xx is the x values for the cubic spline
        self.xx = np.arange(int(x0), int(x_end), 1)

        cs = CubicSpline(x, y, bc_type=((1, 0.0), 'not-a-knot'))
        # self.yy is the y values for the cubic spline
        self.yy = cs(self.xx) * 5 + vert_offset

        x_g = self.xx[-1]
        y_g = self.yy[-1]
        yaw_g = np.arctan((self.yy[-1] - self.yy[-2]) / (self.xx[-1] - self.xx[-2]))
        self.goal_pose = (x_g, y_g, yaw_g)

    def get_point_on_path(self, percent):
        """
        Get a point on the path at a given percentage along its length.

        Args:
            percent: Float between 0.0 and 1.0 representing position along path

        Returns:
            Tuple (x, y, yaw) where yaw is the heading angle in radians
        """
        if len(self.xx) == 0:
            raise ValueError("Path has not been generated yet")

        # Clamp percent to valid range
        percent = np.clip(percent, 0.0, 1.0)

        # Find index at this percentage of path length
        idx = int(percent * (len(self.xx) - 1))
        idx = np.clip(idx, 0, len(self.xx) - 2)  # Ensure we can compute derivative

        x = self.xx[idx]
        y = self.yy[idx]

        # Compute yaw from path tangent (using forward difference)
        dx = self.xx[idx + 1] - self.xx[idx]
        dy = self.yy[idx + 1] - self.yy[idx]
        yaw = np.arctan2(dy, dx)

        return x, y, yaw

    def render_path(self, surface, xx=None, yy=None, color=COLOR_BLACK):
        if xx is None or yy is None:
            xx = self.xx
            yy = self.yy
        for i in range(len(xx) - 1):
            x1 = int(xx[i] / METERS_PER_PIXEL)
            y1 = int(WINDOW_HEIGHT - yy[i] / METERS_PER_PIXEL)
            x2 = int(xx[i + 1] / METERS_PER_PIXEL)
            y2 = int(WINDOW_HEIGHT - yy[i + 1] / METERS_PER_PIXEL)
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

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
        # self.render_path(surface)

        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def step(self, action):
        # override the action to use a constant speed
        # action = np.array([action[0], -1 * config.initial_xd])
        return super().step(action)

    def reset(self, seed=None, options=None):
        for attempt in range(self.max_attempts):
            # Generate path first so we can find starting position on it
            self.generate_path()

            # Get position 10% along the path
            x, y, yaw = self.get_point_on_path(0.1)

            # This env drives backward (negative speed, reversed yaw)
            self.vehicle.reset(-1 * config.initial_xd, x=x, y=y, p=yaw + np.pi)

            # Check if spawn is collision-free
            if not self._check_collision():
                break

            if attempt == self.max_attempts - 1:
                print(f"Warning: Could not find collision-free spawn after {self.max_attempts} attempts")
        self.success = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, info


class LaneDrivingEnv(LineFollowingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)
        self.occ_grid = None  # np.uint8 [H,W], 0=free, 100=blocked
        self.occ_meta: GridMeta | None = None
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

        self._ensure_occ_surface_if_needed()
        self.obstacle_mask = pygame.mask.from_surface(self._occ_surface)

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

    def _check_collision(self):
        """
        Checks for collision between the vehicle (tractor and trailer) and obstacles.
        Returns True if a collision occurs.
        Child classes should set self.obstacle_mask to enable collision detection.
        """
        if self.obstacle_mask is None:
            return False

        vehicle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        vehicle_surface.fill((0, 0, 0, 0))
        vehicle_surface = self._render_vehicle(vehicle_surface)
        alpha = pygame.surfarray.pixels_alpha(vehicle_surface).copy().T
        ys, xs = np.nonzero(alpha > 0)
        x_world = xs * METERS_PER_PIXEL
        y_world = (WINDOW_HEIGHT - ys) * METERS_PER_PIXEL
        # --- World → grid ---
        meta = self.occ_meta
        gx = np.floor((x_world - meta.origin_x) / meta.res_m).astype(int)
        gy = np.floor((y_world - meta.origin_y) / meta.res_m).astype(int)

        # --- Out-of-bounds = collision ---
        if (
                np.any(gx < 0) or np.any(gx >= meta.width) or
                np.any(gy < 0) or np.any(gy >= meta.height)
        ):
            return True

        # --- Occupancy check ---
        if np.any(self.occ_grid[gy, gx] == 100):
            return True

        # jack knife collision check
        hitch_angle = abs((self.vehicle.p - self.vehicle.trailer.yaw + np.pi) % (2 * np.pi) - np.pi)
        if hitch_angle > np.deg2rad(90):
            return True

        return False

    def generate_path(self):
        super().generate_path()
        self._build_occupancy_grid()

    def reset(self, seed=None, options=None):
        for attempt in range(self.max_attempts):
            # Generate path first so we can find starting position on it
            self.generate_path()

            # Get position 10% along the path
            x, y, yaw = self.get_point_on_path(0.1)

            # Reset vehicle at this position, facing along the path
            self.vehicle.reset(config.initial_xd, x=x, y=y, p=yaw)

            # Check if spawn is collision-free
            if not self._check_collision():
                break

            if attempt == self.max_attempts - 1:
                print(f"Warning: Could not find collision-free spawn after {self.max_attempts} attempts")

        self.success = False
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


class StateObservationLineFollowingEnv(LaneDrivingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)

        self.obs_low = np.array([
                -config.steering_observation,
                -config.hitch_angle_observation,
                -config.cross_track_distance_observation,
                -config.cross_track_angle_observation,
                -config.cross_track_distance_observation,
                -config.cross_track_angle_observation,
                -config.curvature_observation,
                -config.curvature_observation
            ])
        self.obs_high = np.array([
                config.steering_observation,
                config.hitch_angle_observation,
                config.cross_track_distance_observation,
                config.cross_track_angle_observation,
                config.cross_track_distance_observation,
                config.cross_track_angle_observation,
                config.curvature_observation,
                config.curvature_observation
            ])

        # redefine observation space
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32)
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_obs(self):
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        k1 = compute_curvature(self, lookahead_steps=10, max_curvature=config.curvature_observation)
        k2 = compute_curvature(self, lookahead_steps=20, max_curvature=config.curvature_observation)
        self.observation = np.array([
            self.vehicle.s,
            hitch_angle,
            error,
            error_theta,
            error_t,
            error_theta_t,
            k1,
            k2
        ], dtype=np.float32)
        return self.observation


class ReverseStateObservationLineFollowingEnv(StateObservationLineFollowingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode="human")

        # redefine action space to reverse
        self.action_space = spaces.Box(
            low=np.array([-np.deg2rad(config.steering_action), -config.speed_action_high], dtype=np.float64),
            high=np.array([np.deg2rad(config.steering_action), -config.speed_action_low], dtype=np.float64),
        )

    def _get_reward(self):
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        return self.get_reward(error, error_theta, error_t, error_theta_t)

    def get_reward(self, error, error_theta, error_t, error_theta_t):
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw

        # --- Terminal ---
        if self._get_term():
            if self.success:
                return 200.0
            return -500.0

        # --- Stability penalties ---
        path_penalty = (
                error ** 2 +
                0.5 * error_theta ** 2 +
                0.5 * error_t ** 2 +
                0.25 * error_theta_t ** 2
        )

        # --- Jackknife prevention ---
        jackknife_penalty = 10.0 * max(0.0, abs(hitch_angle) - 0.4) ** 2

        # --- Reward slow, controlled reverse ---
        reverse_reward = 3.0 * np.clip(-self.vehicle.xd, 0.0, 1.0)

        # --- Final reward ---
        return reverse_reward - path_penalty - jackknife_penalty

    def reset(self, seed=None, options=None):
        for attempt in range(self.max_attempts):
            # Generate path first so we can find starting position on it
            self.generate_path()

            # Get position 10% along the path
            x, y, yaw = self.get_point_on_path(0.10)

            # Reset vehicle at this position, facing along the path
            self.vehicle.reset(config.initial_xd, x=x, y=y, p=(yaw + np.pi))

            # Check if spawn is collision-free
            if not self._check_collision():
                break

            if attempt == self.max_attempts - 1:
                print(f"Warning: Could not find collision-free spawn after {self.max_attempts} attempts")

        self.success = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # override the action to use a constant negative speed
        action = np.array([action[0], -1 * config.initial_xd])
        return super().step(action)


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
    env = StateObservationLineFollowingEnv(render_mode='human')
    forward_pure_pursuit(env)
    env.close()


if __name__ == "__main__":
    main()
