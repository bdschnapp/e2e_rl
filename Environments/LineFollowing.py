import numpy as np
import pygame
from scipy.interpolate import CubicSpline
from gymnasium import spaces
from dataclasses import dataclass
from scipy.spatial import cKDTree

from Environments.TractorTrailer import TractorTrailerEnv, WINDOW_WIDTH, WINDOW_HEIGHT, METERS_PER_PIXEL, COLOR_WHITE, \
    COLOR_BLACK, TRACTOR_WIDTH, TRAILER_WIDTH
import e2erl_utils.config as config

FORWARD_REWARD_MODES = ("dense", "tractor_focus", "multiplicative", "guided")
REVERSE_REWARD_MODES = ("dense", "no_hitch", "multiplicative", "guided")


@dataclass
class GridMeta:
    origin_x: float        # world x at grid(0,0)
    origin_y: float        # world y at grid(0,0)
    res_m: float           # meters per cell
    width: int             # cells
    height: int            # cells


def forward_pure_pursuit(env, render=True):
    from controllers.pure_pursuit import PurePursuitController

    env.reset()
    done = False
    controller = PurePursuitController(k_ff=0.85, k_y=0.18, k_theta=0.36)
    max_steer_rate = float(np.deg2rad(config.steering_action))
    wheelbase = float(env.vehicle.lr + env.vehicle.lf)
    total_reward = 0.0
    steps = 0
    while not done:
        error, error_theta = env.get_vehicle_errors()
        kappa = compute_curvature(env)
        action = controller.step(
            e_y=float(error),
            e_theta=float(error_theta),
            kappa=float(kappa),
            wheelbase=wheelbase,
            current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=max_steer_rate,
        )

        obs, reward, term, trunc, info = env.step(env.format_action(action))
        total_reward += reward
        steps += 1
        done = term or trunc
        if render:
            env.render()

    return total_reward, steps


def reverse_pure_pursuit(env, render=False):
    from controllers.pure_pursuit import ReverseHitchPurePursuitController

    obs, _ = env.reset()
    done = False
    controller = ReverseHitchPurePursuitController(k_hitch=1.2, k_y=0.25, k_theta=0.6, k_ff=0.0)
    max_steer_rate = float(np.deg2rad(config.steering_action))

    total_reward, steps = 0.0, 0

    while not done:
        error_t, error_theta_t = env.get_trailer_errors()
        action = controller.step(
            psi2=float(env.vehicle.p - env.vehicle.trailer.yaw),
            e_y_t=float(error_t),
            e_theta_t=float(error_theta_t),
            kappa=float(compute_curvature(env)),
            current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=max_steer_rate,
        )

        obs, reward, term, trunc, info = env.step(env.format_action(action))
        total_reward += reward
        steps += 1
        done = term or trunc

        if render:
            env.render()

    return total_reward, steps


class LineFollowingEnv(TractorTrailerEnv):
    def __init__(self, render_mode="human", reward_mode: str = "dense", fixed_speed: bool = True):
        super().__init__(render_mode=render_mode)
        if reward_mode not in FORWARD_REWARD_MODES:
            raise ValueError(
                f"Unknown forward reward_mode={reward_mode!r}. "
                f"Choose from {FORWARD_REWARD_MODES}."
            )
        self.reward_mode = reward_mode
        self.fixed_speed = bool(fixed_speed)
        self.fixed_speed_command = float(config.initial_xd)

        if self.fixed_speed:
            self.action_space = spaces.Box(
                low=np.array([-np.deg2rad(config.steering_action)], dtype=np.float32),
                high=np.array([np.deg2rad(config.steering_action)], dtype=np.float32),
                dtype=np.float32,
            )

        # guided-reward curriculum state
        self.transition_timesteps = 100_000  # steps over which alpha decays 1→0
        self._guide_steps = 0                # lifetime step counter for alpha
        self._last_action = np.zeros(2, dtype=np.float32)
        self._guide_controller = None        # lazily created on first use
        self._guide_controller_kwargs = {}   # override before training for custom PP gains

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
                                                              ], dtype=np.float32),
                                                high=np.array([config.steering_observation,
                                                               config.hitch_angle_observation,
                                                               config.cross_track_distance_observation,
                                                               config.cross_track_angle_observation,
                                                               config.cross_track_distance_observation,
                                                               config.cross_track_angle_observation
                                                               ], dtype=np.float32),
                                                dtype=np.float32)

        self.observation_space = spaces.Dict({
            'vector': self.box_observation_space,
            'image': self.image_observation_space
        })

    def _get_progress_fraction(self) -> float:
        """Fraction [0, 1] of environment traversed at the current step.

        Uses the furthest x-position reached by either the tractor or trailer,
        normalised between the path start (x≈5 m) and the success threshold.
        Works for both forward driving (tractor leads) and reverse driving
        (trailer leads, but both units end up with large x values).
        """
        env_len = WINDOW_WIDTH * METERS_PER_PIXEL * 0.85  # success threshold
        start_x = 5.0
        current_x = max(self.vehicle.x, self.vehicle.trailer.x)
        return float(np.clip((current_x - start_x) / (env_len - start_x), 0.0, 1.0))

    def _get_reward(self):
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        return self.get_reward(error, error_theta, error_t, error_theta_t)

    def _proximity_penalty(self) -> float:
        return 0.0

    def get_reward(self, error, error_theta, error_t, error_theta_t):
        if self._get_term():
            if self.success:
                return 100.0
            return -100.0

        progress_reward = 0.5 * self.vehicle.xd
        tractor_penalty = (error ** 2) + (error_theta ** 2)
        trailer_penalty = ((0.5 * error_t) ** 2) + ((0.5 * error_theta_t) ** 2)
        hitch_angle = abs(self.vehicle.p - self.vehicle.trailer.yaw)

        if self.reward_mode == "dense":
            return progress_reward - tractor_penalty - trailer_penalty - self._proximity_penalty()

        if self.reward_mode == "tractor_focus":
            return progress_reward - tractor_penalty - self._proximity_penalty()

        if self.reward_mode == "multiplicative":
            # Truly multiplicative: speed is INSIDE the product so a
            # stopped vehicle on a perfect line gets zero reward, mirroring
            # how the reverse multiplicative reward at line ~1011 is
            # structured. The previous additive form
            #     4 · path · hitch  +  0.5 · clip(xd, 0, 1) - P_prox
            # gave the policy a free 4 reward per step for "park on line",
            # which became its preferred behaviour at deployment-class
            # speeds (≤ 2 m/s) where the env-truncation pressure isn't
            # large enough to dominate.
            path_term = np.exp(
                -(
                    abs(error)
                    + 0.5 * abs(error_theta)
                    + 0.75 * abs(error_t)
                    + 0.5 * abs(error_theta_t)
                )
            )
            hitch_term = np.exp(-1.5 * abs(hitch_angle))
            return (
                4.0 * np.clip(self.vehicle.xd, 0.0, 1.0)
                * path_term * hitch_term
                - self._proximity_penalty()
            )

        if self.reward_mode == "guided":
            dense_rew = progress_reward - tractor_penalty - trailer_penalty

            alpha = max(0.0, 1.0 - self._guide_steps / self.transition_timesteps)
            self._guide_steps += 1

            if alpha == 0.0:
                return dense_rew - self._proximity_penalty()

            pp_action = self._compute_guide_pp_action()
            max_steer_rate = np.deg2rad(config.steering_action)
            steer_diff = (self._last_action[0] - pp_action[0]) / (2.0 * max_steer_rate)
            guide_rew = progress_reward - 5.0 * steer_diff ** 2

            return alpha * guide_rew + (1.0 - alpha) * dense_rew - self._proximity_penalty()

        raise ValueError(f"Unsupported forward reward_mode={self.reward_mode!r}")

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

        y = (a + (b - a) * self.np_random.random(n))

        # Have it start so the vehicle isn't immediately on the line, y(0) of vehicle will always be vert_offset
        y[0] = (-0.25 / 50) + self.np_random.random() * (0.5 / 50)

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

    def format_action(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.fixed_speed:
            if action_arr.size == 0:
                raise ValueError("Expected at least one steering command value.")
            return np.array([float(action_arr[0]), self.fixed_speed_command], dtype=np.float32)
        if action_arr.size != 2:
            raise ValueError(
                f"Variable-speed line-following expects 2 action values, got shape {action_arr.shape}."
            )
        return action_arr.astype(np.float32, copy=False)

    def step(self, action):
        formatted_action = self.format_action(action)
        # Cache the action before vehicle physics runs so _get_reward() can access it.
        # For reverse envs this is called with the modified (fixed-speed) action,
        # which is fine since the speed component is the same for both RL and PP.
        self._last_action = np.array(formatted_action, dtype=np.float32)
        return super().step(formatted_action)

    def _create_guide_controller(self):
        """Return the PP controller for forward driving. Override in reverse subclasses."""
        from controllers.pure_pursuit import PurePursuitController
        return PurePursuitController(**self._guide_controller_kwargs)

    def _compute_guide_pp_action(self):
        """Compute the forward pure-pursuit action given the current env state."""
        if self._guide_controller is None:
            self._guide_controller = self._create_guide_controller()
        error, error_theta = self.get_vehicle_errors()
        kappa = compute_curvature(self)
        wheelbase = self.vehicle.lf + self.vehicle.lr
        max_steer_rate = np.deg2rad(config.steering_action)
        return self._guide_controller.step(
            e_y=error,
            e_theta=error_theta,
            kappa=kappa,
            wheelbase=wheelbase,
            current_steer=self.vehicle.s,
            dt=self.vehicle.dt,
            max_steer_rate=max_steer_rate,
        )

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
    # Proximity shaping applies to the lane occupancy grid, so it covers both
    # lane boundaries and obstacle cells inserted by obstacle environments.
    _PROXIMITY_THRESHOLD_M: float = 2.0
    _MAX_PROXIMITY_PENALTY: float = 5.0

    def __init__(self, render_mode="human", reward_mode: str = "dense", fixed_speed: bool = True):
        super().__init__(render_mode=render_mode, reward_mode=reward_mode, fixed_speed=fixed_speed)
        self.occ_grid = None  # np.uint8 [H,W], 0=free, 100=blocked
        self.occ_meta: GridMeta | None = None
        self._occ_surface = None  # pygame.Surface aligned to world extents
        self._occ_dirty = True  # set True whenever grid changes
        self._show_spline_debug = True  # toggle overlay of spline (thin line)

        # Reusable surfaces to avoid memory leaks from repeated allocations
        self._collision_surface = None  # reused in _check_collision
        self._shifted_surface = None    # reused in _render_frame
        self._final_canvas = None       # reused in _render_frame

        # Cache termination result to avoid redundant collision checks
        self._term_cache = None
        self._term_cache_step = -1
        self._step_count = 0

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

    def _min_clearance_to_occ_m(self) -> float:
        """
        Return approximate clearance from the vehicle body to the nearest
        blocked occupancy-grid cell in metres.
        """
        if self.occ_grid is None or self.occ_meta is None:
            return float("inf")

        meta = self.occ_meta
        veh_half_w = max(TRACTOR_WIDTH, TRAILER_WIDTH) / 2.0
        search_m = self._PROXIMITY_THRESHOLD_M + veh_half_w
        r_cells = int(np.ceil(search_m / meta.res_m))

        dy_arr, dx_arr = np.mgrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
        dist_arr = np.hypot(dx_arr, dy_arr) * meta.res_m
        within = dist_arr <= search_m
        dy_w = dy_arr[within]
        dx_w = dx_arr[within]
        d_w = dist_arr[within]

        min_clearance = float("inf")
        for vx, vy in (
            (self.vehicle.x, self.vehicle.y),
            (self.vehicle.trailer.x, self.vehicle.trailer.y),
        ):
            gx0 = int((vx - meta.origin_x) / meta.res_m)
            gy0 = int((vy - meta.origin_y) / meta.res_m)

            gx = gx0 + dx_w
            gy = gy0 + dy_w

            oob = (gx < 0) | (gx >= meta.width) | (gy < 0) | (gy >= meta.height)
            gx_c = np.clip(gx, 0, meta.width - 1)
            gy_c = np.clip(gy, 0, meta.height - 1)
            blocked = oob | (self.occ_grid[gy_c, gx_c] == 100)

            if blocked.any():
                clearance = max(0.0, float(d_w[blocked].min()) - veh_half_w)
                min_clearance = min(min_clearance, clearance)

        return min_clearance

    def _proximity_penalty(self) -> float:
        """
        Per-step penalty for being close to a lane boundary or obstacle.
        The penalty ramps quadratically from zero at the threshold to the
        configured maximum at zero clearance.
        """
        clearance = self._min_clearance_to_occ_m()
        if clearance >= self._PROXIMITY_THRESHOLD_M:
            return 0.0
        t = 1.0 - clearance / self._PROXIMITY_THRESHOLD_M
        return t ** 2 * self._MAX_PROXIMITY_PENALTY

    def _check_collision(self):
        """
        Checks for collision between the vehicle (tractor and trailer) and obstacles.
        Returns True if a collision occurs.
        Child classes should set self.obstacle_mask to enable collision detection.
        """
        if self.obstacle_mask is None:
            return False

        # Reuse collision surface to avoid memory leak from repeated allocations
        if self._collision_surface is None:
            self._collision_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # Clear and reuse existing surface
        self._collision_surface.fill((0, 0, 0, 0))
        self._render_vehicle(self._collision_surface)

        # Get alpha channel - pixels_alpha returns a view, copy to detach from surface
        alpha = pygame.surfarray.pixels_alpha(self._collision_surface)

        # Use numpy operations more efficiently - avoid creating large intermediate arrays
        # Find non-zero indices directly
        nonzero_mask = alpha > 0
        ys, xs = np.where(nonzero_mask.T)  # .T because surfarray is (width, height)

        if len(xs) == 0:
            return False  # No vehicle pixels rendered (shouldn't happen normally)

        x_world = xs * METERS_PER_PIXEL
        y_world = (WINDOW_HEIGHT - ys) * METERS_PER_PIXEL

        # --- World → grid ---
        meta = self.occ_meta
        gx = np.floor((x_world - meta.origin_x) / meta.res_m).astype(np.int32)
        gy = np.floor((y_world - meta.origin_y) / meta.res_m).astype(np.int32)

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

    def step(self, action):
        # Increment step counter and invalidate termination cache
        self._step_count += 1
        self._term_cache = None
        return super().step(action)

    def _get_term(self):
        """Cached termination check to avoid redundant collision detection."""
        if self._term_cache is not None and self._term_cache_step == self._step_count:
            return self._term_cache

        # Compute termination - super()._get_term() already includes collision check
        term = super()._get_term()

        # Cache the result
        self._term_cache = term
        self._term_cache_step = self._step_count
        return term

    def reset(self, seed=None, options=None):
        # Invalidate termination cache on reset
        self._term_cache = None
        self._step_count = 0

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
        if getattr(self, "_show_spline_debug", False):
            self.render_path(world_canvas)
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

        # Reuse shifted_surface to avoid memory leak
        if self._shifted_surface is None:
            self._shifted_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._shifted_surface.fill((0, 0, 0))
        self._shifted_surface.blit(
            world_canvas,
            (offset_x, offset_y)
        )
        yaw_cam_deg = np.rad2deg(yaw_cam)
        if getattr(config, "bev_forward_up", True):
            rot_deg = yaw_cam_deg + 90.0
        else:
            rot_deg = yaw_cam_deg
        zoom_scale = getattr(config, "bev_zoom_scale", 1.5)
        # Note: rotozoom always creates a new surface (unavoidable due to size changes)
        bev_rotzoom = pygame.transform.rotozoom(self._shifted_surface, rot_deg, zoom_scale)

        # Reuse final_canvas to avoid memory leak
        if self._final_canvas is None:
            self._final_canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._final_canvas.fill((0, 0, 0))
        bev_rect = bev_rotzoom.get_rect()
        bev_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self._final_canvas.blit(bev_rotzoom, bev_rect.topleft)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self._final_canvas)), axes=(1, 0, 2))

    def _get_anchor_world(self):
        """Return BEV camera anchor point and yaw using the configured mount."""
        anchor = getattr(config, "bev_anchor", "tractor_rear_axle")
        offset_x = float(getattr(config, "bev_offset_x_m", 0.0))
        offset_y = float(getattr(config, "bev_offset_y_m", 0.0))

        if anchor == "tractor_cg":
            xA = self.vehicle.x
            yA = self.vehicle.y
            yaw = self.vehicle.p
        elif anchor == "trailer_axle":
            xA = self.vehicle.trailer.x
            yA = self.vehicle.trailer.y
            yaw = self.vehicle.trailer.yaw
        else:
            # Approximate the rear axle from the tractor CG and wheelbase split.
            xA = self.vehicle.x - self.vehicle.lr * np.cos(self.vehicle.p)
            yA = self.vehicle.y - self.vehicle.lr * np.sin(self.vehicle.p)
            yaw = self.vehicle.p

        # Apply a configurable camera bias in the anchor body frame.
        xA += offset_x * np.cos(yaw) - offset_y * np.sin(yaw)
        yA += offset_x * np.sin(yaw) + offset_y * np.cos(yaw)
        psi = -yaw
        return xA, yA, psi

    def _get_state_vector_obs(self):
        """Return 8-dim state vector [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂]."""
        error, error_theta = self.get_vehicle_errors()
        error_t, error_theta_t = self.get_trailer_errors()
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        k1 = compute_curvature(self, lookahead_steps=10, max_curvature=config.curvature_observation)
        k2 = compute_curvature(self, lookahead_steps=20, max_curvature=config.curvature_observation)
        return np.array([
            self.vehicle.s, hitch_angle, error, error_theta, error_t, error_theta_t, k1, k2,
        ], dtype=np.float32)

    def _get_bev_image_obs(self):
        """
        Build the CNN image observation from the already-rendered BEV frame.
        The base renderer is left unchanged; this only crops the transformed
        frame before downscaling to the policy input resolution.
        """
        from Environments.TractorTrailer import OBS_WIDTH, OBS_HEIGHT

        full_frame = self._render_frame()

        crop_m = getattr(config, "bev_obs_crop_m", None)
        if crop_m is not None:
            zoom = float(getattr(config, "bev_zoom_scale", 1.0))
            crop_px = int(round(float(crop_m) * zoom / METERS_PER_PIXEL))
            crop_px = max(1, min(crop_px, full_frame.shape[0], full_frame.shape[1]))
            cy = full_frame.shape[0] // 2
            cx = full_frame.shape[1] // 2
            half = crop_px // 2
            y0 = max(0, min(cy - half, full_frame.shape[0] - crop_px))
            if "reverse" in self.__class__.__name__.lower():
                crop_anchor = getattr(
                    config,
                    "bev_obs_crop_anchor_reverse",
                    getattr(config, "bev_obs_crop_anchor", "center"),
                )
            else:
                crop_anchor = getattr(
                    config,
                    "bev_obs_crop_anchor_forward",
                    getattr(config, "bev_obs_crop_anchor", "center"),
                )
            if crop_anchor == "left":
                x0 = 0
            elif crop_anchor == "top":
                y0 = 0
                x0 = max(0, min(cx - half, full_frame.shape[1] - crop_px))
            elif crop_anchor == "right":
                x0 = full_frame.shape[1] - crop_px
            elif crop_anchor == "bottom":
                y0 = full_frame.shape[0] - crop_px
                x0 = max(0, min(cx - half, full_frame.shape[1] - crop_px))
            else:
                x0 = max(0, min(cx - half, full_frame.shape[1] - crop_px))
            full_frame = full_frame[y0:y0 + crop_px, x0:x0 + crop_px]

        surface = pygame.surfarray.make_surface(
            np.transpose(full_frame, axes=(1, 0, 2))
        )
        small_surface = pygame.transform.scale(surface, (OBS_WIDTH, OBS_HEIGHT))
        small_rgb = pygame.surfarray.pixels3d(small_surface)
        small_gray = small_rgb.mean(axis=2)
        return np.expand_dims(small_gray, axis=-1).astype(np.uint8)


def _make_state_obs_bounds():
    """Return (low, high) arrays for the 8-dim state observation space."""
    low = np.array([
        -config.steering_observation,
        -config.hitch_angle_observation,
        -config.cross_track_distance_observation,
        -config.cross_track_angle_observation,
        -config.cross_track_distance_observation,
        -config.cross_track_angle_observation,
        -config.curvature_observation,
        -config.curvature_observation,
    ], dtype=np.float32)
    high = np.array([
        config.steering_observation,
        config.hitch_angle_observation,
        config.cross_track_distance_observation,
        config.cross_track_angle_observation,
        config.cross_track_distance_observation,
        config.cross_track_angle_observation,
        config.curvature_observation,
        config.curvature_observation,
    ], dtype=np.float32)
    return low, high


class StateObservationLineFollowingEnv(LaneDrivingEnv):
    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense",
                 fixed_speed: bool = True):
        super().__init__(render_mode=render_mode, reward_mode=reward_mode, fixed_speed=fixed_speed)

        # Episode timeout to prevent deadlock
        self.max_episode_steps = max_episode_steps

        self.obs_low, self.obs_high = _make_state_obs_bounds()

        # redefine observation space
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32)
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_trunc(self):
        """Episode truncation - timeout if max steps exceeded."""
        if self._step_count >= self.max_episode_steps:
            return True
        return super()._get_trunc()

    def _get_obs(self):
        self.observation = self._get_state_vector_obs()
        return self.observation


class ReverseStateObservationLineFollowingEnv(StateObservationLineFollowingEnv):
    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense",
                 fixed_speed: bool = True):
        if reward_mode not in REVERSE_REWARD_MODES:
            raise ValueError(
                f"Unknown reverse reward_mode={reward_mode!r}. "
                f"Choose from {REVERSE_REWARD_MODES}."
            )
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode="dense",
            fixed_speed=fixed_speed,
        )
        self.reward_mode = reward_mode
        self.fixed_speed_command = -float(config.initial_xd)

        if not self.fixed_speed:
            self.action_space = spaces.Box(
                low=np.array([-np.deg2rad(config.steering_action), -config.speed_action_high], dtype=np.float32),
                high=np.array([np.deg2rad(config.steering_action), -config.speed_action_low], dtype=np.float32),
                dtype=np.float32,
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

        if self.reward_mode == "dense":
            return reverse_reward - path_penalty - jackknife_penalty - self._proximity_penalty()

        if self.reward_mode == "no_hitch":
            return reverse_reward - path_penalty - self._proximity_penalty()

        if self.reward_mode == "multiplicative":
            path_term = np.exp(
                -(
                    abs(error)
                    + 0.5 * abs(error_theta)
                    + 0.75 * abs(error_t)
                    + 0.5 * abs(error_theta_t)
                )
            )
            hitch_term = np.exp(-2.0 * abs(hitch_angle))
            return (
                5.0 * np.clip(-self.vehicle.xd, 0.0, 1.0) * path_term * hitch_term
                - self._proximity_penalty()
            )

        if self.reward_mode == "guided":
            dense_rew = reverse_reward - path_penalty - jackknife_penalty

            alpha = max(0.0, 1.0 - self._guide_steps / self.transition_timesteps)
            self._guide_steps += 1

            if alpha == 0.0:
                return dense_rew - self._proximity_penalty()

            pp_action = self._compute_guide_pp_action()
            max_steer_rate = np.deg2rad(config.steering_action)
            steer_diff = (self._last_action[0] - pp_action[0]) / (2.0 * max_steer_rate)
            guide_rew = reverse_reward - 5.0 * steer_diff ** 2 - jackknife_penalty

            return alpha * guide_rew + (1.0 - alpha) * dense_rew - self._proximity_penalty()

        raise ValueError(f"Unsupported reverse reward_mode={self.reward_mode!r}")

    def _create_guide_controller(self):
        """Return the reverse hitch PP controller for reverse driving."""
        from controllers.pure_pursuit import ReverseHitchPurePursuitController
        return ReverseHitchPurePursuitController(**self._guide_controller_kwargs)

    def _compute_guide_pp_action(self):
        """Compute the reverse pure-pursuit action given the current env state."""
        if self._guide_controller is None:
            self._guide_controller = self._create_guide_controller()
        error_t, error_theta_t = self.get_trailer_errors()
        kappa = compute_curvature(self)
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        max_steer_rate = np.deg2rad(config.steering_action)
        return self._guide_controller.step(
            psi2=hitch_angle,
            e_y_t=error_t,
            e_theta_t=error_theta_t,
            kappa=kappa,
            current_steer=self.vehicle.s,
            dt=self.vehicle.dt,
            max_steer_rate=max_steer_rate,
        )

    def reset(self, seed=None, options=None):
        # Invalidate termination cache on reset
        self._term_cache = None
        self._step_count = 0

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

class BevObservationLineFollowingEnv(LaneDrivingEnv):
    """
    Lane-following env with BOTH state vector AND BEV image observations.

    Observation space (Dict):
      'vector': 8-dim  [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ_10, κ_20]
               (same state as StateObservationLineFollowingEnv)
      'image':  (32, 32, 1) uint8 grayscale BEV from LaneDrivingEnv._render_frame

    This lets image-based agents (CNNFeatureExtractor) access explicit state cues
    alongside the BEV, which is the fair comparison to lidar+state approaches.
    """

    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense",
                 fixed_speed: bool = True):
        super().__init__(render_mode=render_mode, reward_mode=reward_mode, fixed_speed=fixed_speed)
        self.max_episode_steps = max_episode_steps

        vec_low, vec_high = _make_state_obs_bounds()

        self.observation_space = spaces.Dict({
            'vector': spaces.Box(low=vec_low, high=vec_high, dtype=np.float32),
            'image': self.image_observation_space,
        })

    def _get_trunc(self):
        if self._step_count >= self.max_episode_steps:
            return True
        return super()._get_trunc()

    def _get_obs(self):
        return {'image': self._get_bev_image_obs(), 'vector': self._get_state_vector_obs()}


class ReverseLidarStateObservationLineFollowingEnv(ReverseStateObservationLineFollowingEnv):
    """
    Reverse lane-following env with state + simulated lidar observations.

    Mirrors LidarStateObservationLineFollowingEnv but for reverse driving.
    Lidar is mounted on the trailer (which leads in reverse), facing the direction
    of approach (trailer.yaw + π).

    Observation: [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂, d₀, …, d_{N-1}]
    """

    def __init__(self, render_mode="human", max_episode_steps=1000, lidar_beams=16,
                 reward_mode: str = "dense", fixed_speed: bool = True):
        self.lidar_beams = lidar_beams
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=fixed_speed,
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
        from Environments.ObstacleAvoidance import Pose, get_obstacle_distances

        state_obs = super()._get_obs()

        lidar_pose = Pose(
            x=self.vehicle.trailer.x,
            y=self.vehicle.trailer.y,
            yaw=self.vehicle.trailer.yaw + np.pi,
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


class ReverseBevObservationLineFollowingEnv(ReverseStateObservationLineFollowingEnv):
    """
    Reverse lane-following env with state vector + BEV image observations.

    Mirrors BevObservationLineFollowingEnv but for reverse driving.
    Inherits reverse action space, step override, reset, and reward from
    ReverseStateObservationLineFollowingEnv.

    Observation space (Dict):
      'vector': 8-dim [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂]
      'image':  (32, 32, 1) uint8 grayscale BEV
    """

    def __init__(self, render_mode="human", max_episode_steps=1000, reward_mode: str = "dense",
                 fixed_speed: bool = True):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            reward_mode=reward_mode,
            fixed_speed=fixed_speed,
        )

        vec_low, vec_high = _make_state_obs_bounds()
        self.observation_space = spaces.Dict({
            'vector': spaces.Box(low=vec_low, high=vec_high, dtype=np.float32),
            'image': self.image_observation_space,
        })

    def _get_obs(self):
        return {'image': self._get_bev_image_obs(), 'vector': self._get_state_vector_obs()}


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
