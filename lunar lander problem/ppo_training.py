from stable_baselines3 import PPO
import gym
import torch

# Create the standard LunarLander environment
env = gym.make('LunarLander-v2')

# Define the policy network architecture
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])

# Initialize the PPO agent
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

# Train the agent
# Note: Training may take some time depending on your hardware
model.learn(total_timesteps=1000000)

# Save the trained agent
model.save("ppo_lunar_lander_standard_2")