import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from VehicleModels.tractor_trailer import StateSpaceTractorTrailer

# --- Configuration ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 900
METERS_PER_PIXEL = 0.1

# Vehicle Dimensions (in meters)
TRACTOR_LENGTH = 4.5
TRACTOR_WIDTH = 2.5
TRAILER_LENGTH = 10.0
TRAILER_WIDTH = 2.5

# Colors
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_TRACTOR = (20, 35, 125)
COLOR_TRAILER = (220, 130, 10)

# Observation Dimensions
OBS_WIDTH = 84
OBS_HEIGHT = 84


class TractorTrailerEnv(gym.Env):
    """
    A Gymnasium environment for simulating a tractor-trailer parking task.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode='human'):
        super().__init__()

        self.render_mode = render_mode
        self.window = None  # For displaying to the screen
        self.canvas = None  # For drawing the image
        self.clock = None

        # 1. Define Action Space (MODIFIED)
        # Action: [steering_angle_rate_of_change, target_linear_velocity]
        self.action_space = spaces.Box(
            low=np.array([-np.deg2rad(15), 0], dtype=np.float64),
            high=np.array([np.deg2rad(15), 2], dtype=np.float64),
        )

        # 2. Define Observation Space
        self.box_observation_space = spaces.Box(
            low=np.array([-np.pi / 4, -10], dtype=np.float64),
            high=np.array([np.pi / 4, 10], dtype=np.float64),
        )

        self.image_observation_space = spaces.Box(low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 1), dtype=np.uint8)

        self.observation_space = spaces.Dict({
            'vector': self.box_observation_space,
            'image': self.image_observation_space
        })

        # 3. Initialize the Vehicle Model
        vehicle_params = {
            'm': 1500, 'Iz': 3000, 'Cf': 80000, 'Cr': 80000,
            'lf': 1.2, 'lr': 1.6, 'Cd': 0.208, 'A': 2.4, 'dt': 0.1
        }
        self.vehicle = StateSpaceTractorTrailer(args=vehicle_params, trailer_length=TRAILER_LENGTH)

        self.goal_pose = (0.0, 0.0, 0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_speed = 5.0
        self.vehicle.reset(xd=initial_speed)

        self.vehicle.x = (WINDOW_WIDTH * METERS_PER_PIXEL) / 2
        self.vehicle.y = (WINDOW_HEIGHT * METERS_PER_PIXEL) / 2
        self.vehicle.p = np.deg2rad(0)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.vehicle.loop(action)

        reward = self._get_reward()
        terminated = self._get_term()
        truncated = self._get_trunc()
        observation = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            print("Episode Ended")

        return observation, reward, terminated, truncated, info

    def _render_vehicle(self, surface=None):
        self._draw_rotated_rect(
            surface if surface else self.canvas,
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.p,
            TRACTOR_LENGTH,
            TRACTOR_WIDTH,
            COLOR_TRACTOR,
        )

        # Calculate the trailer's center for rendering
        # The physics model gives us the axle position, but we need to draw
        # the body, which is centered half its length forward of the axle.
        trailer_axle_x = self.vehicle.trailer.x
        trailer_axle_y = self.vehicle.trailer.y
        trailer_yaw = self.vehicle.trailer.yaw

        trailer_center_x = trailer_axle_x + (TRAILER_LENGTH / 2) * np.cos(trailer_yaw)
        trailer_center_y = trailer_axle_y + (TRAILER_LENGTH / 2) * np.sin(trailer_yaw)

        # Draw Trailer using the calculated center point
        self._draw_rotated_rect(
            surface if surface else self.canvas,
            trailer_center_x,  # Use the calculated center X
            trailer_center_y,  # Use the calculated center Y
            trailer_yaw,
            TRAILER_LENGTH,
            TRAILER_WIDTH,
            COLOR_TRAILER,
        )

        steer_angle = self.vehicle.s

        # Position for the indicator: center-top of the tractor
        indicator_length = TRACTOR_LENGTH * 0.2
        indicator_offset = TRACTOR_LENGTH * 0.25  # move toward front
        base_x = self.vehicle.x + indicator_offset * np.cos(self.vehicle.p)
        base_y = self.vehicle.y + indicator_offset * np.sin(self.vehicle.p)
        end_x = base_x + indicator_length * np.cos(self.vehicle.p + steer_angle)
        end_y = base_y + indicator_length * np.sin(self.vehicle.p + steer_angle)

        # Draw the line
        self._draw_rotated_line(
            surface if surface else self.canvas,
            base_x, base_y,
            end_x, end_y,
            color=(255, 0, 0),
            width=2,
        )

        return surface if surface else self.canvas

    def _render_frame(self, surface=None):
        if surface is None:
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        self._render_vehicle(surface if surface else self.canvas)

        # Return the canvas content as a numpy array
        return np.transpose(np.array(pygame.surfarray.pixels3d(surface if surface else self.canvas)), axes=(1, 0, 2))

    def render(self):
        # print("TractorTrailerEnv.render called")
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        frame = self._render_frame()

        # Pygame needs the axes swapped for displaying
        surface = pygame.surfarray.make_surface(np.transpose(frame, axes=(1, 0, 2)))
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def get_goal_errors(self):
        goal_x, goal_y, goal_yaw = self.goal_pose
        trailer_x, trailer_y, trailer_yaw = self.vehicle.trailer.x, self.vehicle.trailer.y, self.vehicle.trailer.yaw
        # bound yaw between 0 and 2π
        trailer_yaw = (trailer_yaw + np.pi) % (2 * np.pi)
        goal_yaw = (goal_yaw + np.pi) % (2 * np.pi)

        distance_to_goal = np.sqrt((trailer_x - goal_x) ** 2 + (trailer_y - goal_y) ** 2)
        yaw_error = np.abs(trailer_yaw - goal_yaw)

        return distance_to_goal, yaw_error

    def _check_in_bounds(self):
        if self.vehicle.x < 0 or self.vehicle.x > WINDOW_WIDTH * METERS_PER_PIXEL:
            return False
        if self.vehicle.y < 0 or self.vehicle.y > WINDOW_HEIGHT * METERS_PER_PIXEL:
            return False
        return True

    def _check_jackknife(self):
        hitch_angle = self.vehicle.p - self.vehicle.trailer.yaw
        return abs(hitch_angle) > np.pi / 2

    def _get_term(self):
        distance_to_goal, yaw_error = self.get_goal_errors()
        if distance_to_goal < 0.1 and yaw_error < np.deg2rad(5):
            return True

        if not self._check_in_bounds():
            return True

        return False

    def _get_trunc(self):
        # TODO: check timeout
        return False

    def _get_obs(self):
        # Render the full-size frame onto the canvas
        full_frame = self._render_frame()

        # --- PREPROCESSING ---
        # 1. Convert the rendered frame (NumPy array) to a Pygame Surface
        full_surface = pygame.surfarray.make_surface(np.transpose(full_frame, axes=(1, 0, 2)))

        # 2. Downscale the surface to the desired observation size
        small_surface = pygame.transform.scale(full_surface, (OBS_WIDTH, OBS_HEIGHT))

        # 3. Convert to a grayscale NumPy array. We take the average across the color channels.
        # The surfarray is in (width, height, channel) format.
        small_array_rgb = pygame.surfarray.pixels3d(small_surface)
        small_array_gray = small_array_rgb.mean(axis=2)

        # 4. Reshape to (height, width, 1) to match the observation space
        image_obs = np.expand_dims(small_array_gray, axis=-1).astype(np.uint8)

        vector_obs = np.array([self.vehicle.s, self.vehicle.xd], dtype=np.float32)

        return {'image': image_obs, 'vector': vector_obs}

    def _get_info(self):
        return {
            "tractor_pos": (self.vehicle.x, self.vehicle.y),
            "tractor_yaw": self.vehicle.p,
            "trailer_pos": (self.vehicle.trailer.x, self.vehicle.trailer.y),
            "trailer_yaw": self.vehicle.trailer.yaw,
        }

    def _get_reward_old(self):
        distance_to_goal, yaw_error = self.get_goal_errors()

        # if close to goal and aligned, give a big reward
        if distance_to_goal < 0.2 and yaw_error < np.deg2rad(5):
            return 100.0 / max(self.vehicle.xd, 0.01)

        # Penalize for yaw error when close to goal
        if distance_to_goal < 10:
            return 10 - yaw_error / distance_to_goal

        return 10 / distance_to_goal

    def _get_reward(self):
        current_distance, angle_diff_goal = self.get_goal_errors()

        if current_distance < 0.2 and angle_diff_goal < np.deg2rad(10):
            return 10

        distance_reward = -0.5 * np.tanh((current_distance - 20) / 20) + 0.5  # ∈ [0, ~1]

        angel_reward = -0.5 * np.tanh((angle_diff_goal - np.deg2rad(15)) / np.deg2rad(7)) + 0.5  # ∈ [0, ~1]

        # Combine the reward components
        return distance_reward + angel_reward

    def _draw_rotated_rect(self, surface, x, y, angle, length, width, color, border_width=0):
        """ Generic helper to draw any rotated rectangle. """
        center_x_pix = x / METERS_PER_PIXEL
        center_y_pix = WINDOW_HEIGHT - (y / METERS_PER_PIXEL)
        length_pix = length / METERS_PER_PIXEL
        width_pix = width / METERS_PER_PIXEL

        points = [
            (-length_pix / 2, -width_pix / 2),
            (length_pix / 2, -width_pix / 2),
            (length_pix / 2, width_pix / 2),
            (-length_pix / 2, width_pix / 2),
        ]

        rotated_points = []
        for x_p, y_p in points:
            x_rot = x_p * np.cos(-angle) - y_p * np.sin(-angle) + center_x_pix
            y_rot = x_p * np.sin(-angle) + y_p * np.cos(-angle) + center_y_pix
            rotated_points.append((x_rot, y_rot))

        pygame.draw.polygon(surface, color, rotated_points, border_width)

    def _draw_rotated_line(self, surface, x0, y0, x1, y1, color, width=2):
        """
        Draw a world-coordinate line from (x0,y0) to (x1,y1),
        converting meters→pixels and flipping Y axis.
        """
        x0_pix = x0 / METERS_PER_PIXEL
        y0_pix = WINDOW_HEIGHT - (y0 / METERS_PER_PIXEL)

        x1_pix = x1 / METERS_PER_PIXEL
        y1_pix = WINDOW_HEIGHT - (y1 / METERS_PER_PIXEL)

        pygame.draw.line(surface, color, (x0_pix, y0_pix), (x1_pix, y1_pix), width)

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = TractorTrailerEnv(render_mode='human')
    obs, info = env.reset()

    # Define a constant action to drive in a circle
    # Action: [steering_rate, target_velocity]
    # - Steering Rate = a constant positive value to turn left
    # - Target Velocity = 5.0 m/s
    circle_action = np.array([np.deg2rad(5.0), 2.0])

    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(circle_action)
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if terminated or truncated:
            print("Episode finished!")
            break

    env.close()
