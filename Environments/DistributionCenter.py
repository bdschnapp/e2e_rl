from Environments.TractorTrailer import *
import random

COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GREY = (30, 30, 30)
COLOR_GOAL = (15, 240, 15)  # Green

PARKING_SPACE_WIDTH = 4.0  # meters
PARKING_SPACE_LENGTH = 12.0  # meters
PARKING_FILL_PERCENTAGE = 0.6  # Fill 60% of parking spots

WAREHOUSE_SIZE = (32.0, 32.0)  # Width, Height in meters


class DistributionCenterEnv(TractorTrailerEnv):
    """
    An environment that generates a distribution center with obstacles
    and a goal parking space.
    """

    def __init__(self, render_mode='human'):
        super().__init__(render_mode)

        # collision checking
        self.obstacles = []

    def reset(self, seed=None, options=None):
        self._generate_layout()
        super().reset(seed=seed)

        world_width = WINDOW_WIDTH * METERS_PER_PIXEL
        world_height = WINDOW_HEIGHT * METERS_PER_PIXEL
        warehouse_width, warehouse_height = WAREHOUSE_SIZE

        margin = 10.0

        horizontal_corridor_height = (world_height - warehouse_height) / 2 - (2 * margin) - PARKING_SPACE_LENGTH
        # horizontal corridor width = world width - 2 * margin
        vertical_corridor_width = (world_width - warehouse_width) / 2 - (2 * margin) - PARKING_SPACE_LENGTH
        # vertical corridor height = world height - 2 * margin

        # --- Randomly Select a Corridor and Position ---
        corridor = random.choice(['top', 'bottom', 'left', 'right'])

        if corridor == 'top':
            start_x = random.uniform(world_width - margin, margin)
            start_y = random.uniform(world_height - margin, world_height - margin - horizontal_corridor_height)
            start_p = random.choice([np.deg2rad(0), np.deg2rad(180)])
        elif corridor == 'bottom':
            start_x = random.uniform(world_width - margin, margin)
            start_y = random.uniform(margin, margin + horizontal_corridor_height)
            start_p = random.choice([np.deg2rad(0), np.deg2rad(180)])
        elif corridor == 'right':
            start_x = random.uniform(world_width - margin - vertical_corridor_width, world_width - margin)
            start_y = random.uniform(world_height - margin, margin)
            start_p = random.choice([np.deg2rad(90), np.deg2rad(270)])
        else:  # 'left'
            start_x = random.uniform(margin, margin + vertical_corridor_width)
            start_y = random.uniform(margin, world_height - margin)
            start_p = random.choice([np.deg2rad(90), np.deg2rad(270)])

        self.vehicle.reset(xd=0, x=start_x, y=start_y, p=start_p)

        observation = self._get_obs()
        info = self._get_info()

        info['goal_pose'] = self.goal_pose
        info['obstacles'] = self.obstacles
        print("DC Reset Complete")
        return observation, info

    def _get_reward(self):
        # default distance based reward
        reward = super()._get_reward()

        # collision checking
        if self._check_collision() or not self._check_in_bounds():
            reward = -10.0

        distance, angle_diff = self.get_goal_errors()
        print(f"reward: {reward}, distance: {distance}")
        return reward

    def _get_term(self):
        # default termination based on distance to goal
        term = super()._get_term()

        # collision checking
        if self._check_collision():
            term = True

        return term

    def _generate_layout(self):
        """
        Generates a layout with a central warehouse surrounded by parking spaces.
        A percentage of spaces are filled, and one empty space is chosen as the goal.
        """
        self.obstacles = []
        self.goal_rect = None
        self.goal_pose = (0.0, 0.0, 0.0)
        all_parking_spots = []

        world_width = WINDOW_WIDTH * METERS_PER_PIXEL
        world_height = WINDOW_HEIGHT * METERS_PER_PIXEL

        # 1. Define the central warehouse
        warehouse_width, warehouse_height = WAREHOUSE_SIZE
        warehouse_x = world_width / 2
        warehouse_y = world_height / 2

        warehouse_rect = pygame.Rect(0, 0, warehouse_width, warehouse_height)
        warehouse_rect.center = (warehouse_x, warehouse_y)
        self.obstacles.append(warehouse_rect)

        # --- Generate parking spots around the warehouse ---

        # Top Edge (pointing up)
        num_spots_top = int((warehouse_width - 2) / PARKING_SPACE_WIDTH)
        for i in range(num_spots_top):
            x = warehouse_rect.left + (i * PARKING_SPACE_WIDTH) + PARKING_SPACE_WIDTH / 2
            y = warehouse_rect.top  - PARKING_SPACE_LENGTH / 2
            all_parking_spots.append({'pos': (x, y), 'angle': np.deg2rad(90)})

        # Bottom Edge (pointing down)
        num_spots_bottom = int((warehouse_width - 2) / PARKING_SPACE_WIDTH)
        for i in range(num_spots_bottom):
            x = warehouse_rect.left + (i * PARKING_SPACE_WIDTH) + PARKING_SPACE_WIDTH / 2
            y = warehouse_rect.bottom + PARKING_SPACE_LENGTH / 2
            all_parking_spots.append({'pos': (x, y), 'angle': np.deg2rad(90)})

        # Right Edge (pointing right)
        num_spots_right = int((warehouse_height - 2) / PARKING_SPACE_WIDTH)
        for i in range(num_spots_right):
            x = warehouse_rect.right + PARKING_SPACE_LENGTH / 2
            y = warehouse_rect.bottom - (i * PARKING_SPACE_WIDTH) - PARKING_SPACE_WIDTH / 2
            all_parking_spots.append({'pos': (x, y), 'angle': np.deg2rad(0)})

        # Left Edge (pointing left)
        num_spots_left = int((warehouse_height - 2) / PARKING_SPACE_WIDTH)
        for i in range(num_spots_left):
            x = warehouse_rect.left - PARKING_SPACE_LENGTH / 2
            y = warehouse_rect.bottom - (i * PARKING_SPACE_WIDTH) - PARKING_SPACE_WIDTH / 2
            all_parking_spots.append({'pos': (x, y), 'angle': np.deg2rad(0)})

        # 3. Randomly fill parking spots and select a goal
        random.shuffle(all_parking_spots)

        num_to_fill = int(len(all_parking_spots) * PARKING_FILL_PERCENTAGE)

        occupied_spots = all_parking_spots[:num_to_fill]
        empty_spots = all_parking_spots[num_to_fill:]

        # Create obstacles for the occupied spots
        for spot in occupied_spots:
            x, y = spot['pos']
            angle = spot['angle']
            # For obstacles, we can use the same dimensions as the parking space
            # We need to create a rotated rect. A simple way is to use a wide trailer body.
            if angle == np.deg2rad(0):
                obstacle = pygame.Rect(0, 0, PARKING_SPACE_LENGTH, PARKING_SPACE_WIDTH)
            elif angle == np.deg2rad(90):
                obstacle = pygame.Rect(0, 0, PARKING_SPACE_WIDTH, PARKING_SPACE_LENGTH)
            obstacle.center = (x, y)
            # Note: Pygame rects are axis-aligned. For true rotated obstacles,
            # we would need more complex collision logic later. For rendering, this is fine.
            # Let's add them as simple rects for now.
            self.obstacles.append(obstacle)

        goal_spot = random.choice(empty_spots)

        if goal_spot['angle'] == np.deg2rad(0):
            self.goal_rect = pygame.Rect(0, 0, PARKING_SPACE_LENGTH, PARKING_SPACE_WIDTH)
        elif goal_spot['angle'] == np.deg2rad(90):
            self.goal_rect = pygame.Rect(0, 0, PARKING_SPACE_WIDTH, PARKING_SPACE_LENGTH)
        self.goal_rect.center = goal_spot['pos']

        self.goal_pose = (goal_spot['pos'][0], goal_spot['pos'][1], goal_spot['angle'])

        # --- Create a mask for all static obstacles ---
        mask_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        mask_surface.fill((0, 0, 0, 0))  # Transparent background
        self.obstacle_mask = pygame.mask.from_surface(self._render_obstacles(mask_surface))

    def _render_obstacles(self, surface=None):
        # Draw Obstacles
        for obs_rect in self.obstacles:
            # Convert from meters to pixels for drawing
            pix_rect = pygame.Rect(
                obs_rect.left / METERS_PER_PIXEL,
                obs_rect.top / METERS_PER_PIXEL,
                obs_rect.width / METERS_PER_PIXEL,
                obs_rect.height / METERS_PER_PIXEL
            )
            if surface is None:
                pygame.draw.rect(self.canvas, COLOR_DARK_GREY, pix_rect)
            else:
                pygame.draw.rect(surface, COLOR_DARK_GREY, pix_rect)
        return surface if surface else self.canvas

    def _render_frame(self, surface=None):
        if surface is None:
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        surface = surface if surface else self.canvas

        surface.fill(COLOR_WHITE)
        self._render_obstacles(surface)

        # Draw the goal parking space
        goal_pix_rect = pygame.Rect(
            self.goal_rect.left / METERS_PER_PIXEL,
            self.goal_rect.top / METERS_PER_PIXEL,
            self.goal_rect.width / METERS_PER_PIXEL,
            self.goal_rect.height / METERS_PER_PIXEL
        )
        pygame.draw.rect(surface, COLOR_GOAL, goal_pix_rect, border_radius=2)

        # --- Draw Vehicle (using the method from the parent class) ---
        super()._render_frame(surface)

        # Return the canvas content as a numpy array
        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the environment
    env = DistributionCenterEnv(render_mode='human')

    # Define training parameters
    total_episodes = 10
    running = True

    # 2. Outer Loop: Iterate through episodes
    for episode in range(total_episodes):
        print(f"--- Starting Episode: {episode + 1} ---")

        # 3. Reset the environment for a new episode
        obs, info = env.reset()

        # 4. Inner Loop: Run the episode until it ends
        while True:
            # drive in a circle
            action = np.array([np.deg2rad(1.0), -2.0])

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the environment (optional)
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    running = False

            # 5. Check if the episode is over
            if terminated or truncated:
                if terminated:
                    print("Episode terminated.")
                if truncated:
                    print("Episode truncated.")

                # The inner loop breaks, and the outer loop will start a new episode
                break

        if not running:
            break

    # Clean up the environment
    env.close()
    print("--- Training finished ---")
