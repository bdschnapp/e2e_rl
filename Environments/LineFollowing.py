import numpy as np
import pygame
from scipy.interpolate import CubicSpline
from gymnasium import spaces

from Environments.TractorTrailer import TractorTrailerEnv, WINDOW_WIDTH, WINDOW_HEIGHT, METERS_PER_PIXEL, COLOR_WHITE, \
    COLOR_BLACK
import e2erl_utils.config as config


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
