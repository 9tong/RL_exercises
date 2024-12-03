import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load the CartPole environment
env = gym.make("CartPole-v1")

# Set the PPO Agent policy (MLP)
ppo_model = PPO("MlpPolicy", env, verbose=1)

# Train PPO models for evaluation
ppo_model.learn(total_timesteps=10000)

# Evaluate the trained agent
print("Evaluating the PPO model...")
mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Dev: {std_reward}")

# Save the model
ppo_model.save("ppo_cartpole_model")
print("Model saved as 'ppo_cartpole_model'.")