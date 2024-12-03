import gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Load the CartPole environment
env = gym.make("CartPole-v1")

# Set the A2C Agent policy (MLP)
a2c_model = A2C("MlpPolicy", env, verbose=1)

# Train A2C models for evaluation
a2c_model.learn(total_timesteps=10000)

# Evaluate the trained agent
print("Evaluating the A2C model...")
mean_reward, std_reward = evaluate_policy(a2c_model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Dev: {std_reward}")

# Save the model
a2c_model.save("a2c_cartpole_model")
print("Model saved as 'a2c_cartpole_model'.")