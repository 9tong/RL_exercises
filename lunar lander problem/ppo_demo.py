import gym
import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander
from gym.utils import EzPickle

import pygame
import Box2D

from stable_baselines3 import PPO

# Load ppo model
model = PPO.load("ppo_lunar_lander_standard.zip")

# Initialize Pygame modules
pygame.init()

class CustomLunarLander(LunarLander, EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        super().__init__(**kwargs)
        self.mouse_clicked = False
        self.new_landing_pad_x = None

        # Screen dimensions and scale
        self.W = VIEWPORT_W = 600  # Width of the screen
        self.H = VIEWPORT_H = 400  # Height of the screen
        self.SCALE = SCALE = 30.0  # Scale factor
        self.world_width = VIEWPORT_W / SCALE
        self.world_height = VIEWPORT_H / SCALE

        # Initialize chunk_x and chunk_y as empty lists
        self.chunk_x = []
        self.chunk_y = []

        # Flag position
        self.flag_coords = None

        # Initialize the render mode
        self.render_mode = kwargs.get('render_mode', 'human')
        self.font = pygame.font.SysFont('Arial', 12)

    def render(self):
        result = super().render()

        # Access the Pygame screen
        if self.render_mode == 'human':
            screen = pygame.display.get_surface()

            if self.mouse_clicked and self.flag_coords:
                # Draw the arrow at the selected landing pad position
                arrow_x, arrow_y = self.flag_coords
                arrow_screen_x = int(arrow_x * self.SCALE)
                arrow_screen_y = self.H - int(arrow_y * self.SCALE)

                # Define arrow dimensions
                arrow_width = 10
                arrow_height = 8

                # Define arrow points
                arrow_points = [
                    (arrow_screen_x, arrow_screen_y),  # Tip of the arrow
                    (arrow_screen_x - arrow_width // 2, arrow_screen_y - arrow_height),
                    (arrow_screen_x - arrow_width // 4, arrow_screen_y - arrow_height),
                    (arrow_screen_x - arrow_width // 4, arrow_screen_y - arrow_height * 2),
                    (arrow_screen_x + arrow_width // 4, arrow_screen_y - arrow_height * 2),
                    (arrow_screen_x + arrow_width // 4, arrow_screen_y - arrow_height),
                    (arrow_screen_x + arrow_width // 2, arrow_screen_y - arrow_height),
                ]

                # Draw the arrow
                pygame.draw.polygon(
                    screen,
                    (255, 0, 0),  # Red color
                    arrow_points
                )

                # Draw the env arg, like wind
                message = f'wind:{self.wind_power}'
                text_surface = self.font.render(message, True, (0, 255, 255))
                text_rect = text_surface.get_rect(center=(self.W // 2, 50))
                screen.blit(text_surface, text_rect)

            # Update the display
            pygame.display.flip()

        if not self.mouse_clicked:
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        x_screen, y_screen = event.pos
                        x_world = x_screen / self.SCALE
                        y_world = (self.H - y_screen) / self.SCALE
                        self.new_landing_pad_x = x_world
                        self.mouse_clicked = True
                        print(f"New landing pad x-position set to: {self.new_landing_pad_x:.2f}")

                        # Set flag coordinates
                        self.flag_coords = (x_world, self.helipad_y)

        return result

    def reset(self, **kwargs):
        # Ensure terrain is generated before accessing chunk_x and chunk_y
        super().reset(**kwargs)
        self.mouse_clicked = False
        self.new_landing_pad_x = None
        self.flag_coords = None  # Reset flag coordinates

        # Wait until the mouse is clicked
        while not self.mouse_clicked:
            self.render()
            pygame.time.wait(10)  # Small delay to prevent high CPU usage

        # Set the new landing pad position
        self._set_landing_pad_position(self.new_landing_pad_x)

        # Rebuild the terrain with the new landing pad position
        # self._regenerate_terrain()

        # Reset the environment to apply changes
        observation, info = super().reset(**kwargs)
        return observation, info

    def _set_landing_pad_position(self, x_position):
        pad_width = self.helipad_x2 - self.helipad_x1
        self.helipad_x1 = x_position - pad_width / 2
        self.helipad_x2 = x_position + pad_width / 2

    def _generate_terrain(self):
        # Copy the original _generate_terrain method and store chunk_x and chunk_y
        CHUNKS = 11
        height = self.np_random.uniform(0, self.H / self.SCALE / 2, size=(CHUNKS + 1,))
        self.chunk_x = [i * self.W / self.SCALE / (CHUNKS - 1) for i in range(CHUNKS)]
        self.chunk_y = list(height)
        self.helipad_x1 = self.W / self.SCALE / 2 - 1
        self.helipad_x2 = self.W / self.SCALE / 2 + 1
        self.helipad_y = self.H / self.SCALE / 4

        # Make the ground near the landing pad flat
        for i in range(CHUNKS - 1):
            x1, x2 = self.chunk_x[i], self.chunk_x[i + 1]
            if self.helipad_x1 <= x1 <= self.helipad_x2:
                self.chunk_y[i] = self.helipad_y
                self.chunk_y[i + 1] = self.helipad_y

        # Create terrain
        terrain_poly = [(self.chunk_x[i], self.chunk_y[i]) for i in range(CHUNKS)]
        self.terrain_poly = terrain_poly

        # Build the terrain
        if hasattr(self, 'moon') and self.moon is not None:
            self.world.DestroyBody(self.moon)
        self.moon = self.world.CreateStaticBody(
            shapes=Box2D.b2ChainShape(vertices=terrain_poly),
        )
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    def _regenerate_terrain(self):
        # Ensure that chunk_x and chunk_y are available
        # if not self.chunk_x or not self.chunk_y:
        #     self._generate_terrain()

        # Adjust the terrain points near the helipad to ensure a flat landing area
        for idx, x in enumerate(self.chunk_x):
            if self.helipad_x1 <= x <= self.helipad_x2:
                self.chunk_y[idx] = self.helipad_y

        # Rebuild the terrain with the updated points
        terrain_poly = [(self.chunk_x[i], self.chunk_y[i]) for i in range(len(self.chunk_x))]
        self.terrain_poly = terrain_poly

        # Destroy and recreate the moon body
        if hasattr(self, 'moon') and self.moon is not None:
            self.world.DestroyBody(self.moon)
        self.moon = self.world.CreateStaticBody(
            shapes=Box2D.b2ChainShape(vertices=terrain_poly),
        )
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    # Override the step method to handle Pygame events
    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        # Process Pygame events to handle window closure during simulation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        return observation, reward, done, truncated, info

# Create the custom environment
env = CustomLunarLander(render_mode='human', enable_wind=True, wind_power=8)

# Use the environment as usual
observation, info = env.reset()
done = False
while not done:
    #action = env.action_space.sample()  # Replace with your agent's action
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, truncated, info = env.step(action)
    env.render()
env.close()