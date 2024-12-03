import sys
import time
import gym
from stable_baselines3 import DQN, PPO, A2C
import numpy as np
import matplotlib.pyplot as plt

# Load the CartPole environment
env = gym.make("CartPole-v1")

# Load the trained models
dqn_model = DQN.load("dqn_cartpole_model")
ppo_model = PPO.load("ppo_cartpole_model.zip")
a2c_model = A2C.load("a2c_cartpole_model.zip")

# Function to evaluate a model
def evaluate_model(model, env, num_episodes=5):
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards, np.mean(rewards), np.std(rewards)

# Evaluate each model
dqn_cumulative_rewards, dqn_mean_reward, dqn_std_reward = evaluate_model(dqn_model, env)
ppo_cumulative_rewards, ppo_mean_reward, ppo_std_reward = evaluate_model(ppo_model, env)
a2c_cumulative_rewards, a2c_mean_reward, a2c_std_reward = evaluate_model(a2c_model, env)

# Print evaluation results
print(f"DQN: Mean Reward = {dqn_mean_reward}, Std Reward = {dqn_std_reward}")
print(f"PPO: Mean Reward = {ppo_mean_reward}, Std Reward = {ppo_std_reward}")
print(f"A2C: Mean Reward = {a2c_mean_reward}, Std Reward = {a2c_std_reward}")

# Plot the results
labels = ['DQN', 'PPO', 'A2C']
mean_rewards = [dqn_mean_reward, ppo_mean_reward, a2c_mean_reward]
std_rewards = [dqn_std_reward, ppo_std_reward, a2c_std_reward]

x = np.arange(len(labels))
width = 0.5
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

rects = ax.bar(x, mean_rewards, width, yerr=std_rewards, capsize=5, label='Mean Reward')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Mean Reward')
ax.set_xlabel('Algorithm')
ax.set_title('Comparison of DQN, PPO, and A2C on CartPole')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels to bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)


# Line plot for cumulative rewards
ax2.plot(range(1, len(dqn_cumulative_rewards) + 1), np.cumsum(dqn_cumulative_rewards), label='DQN', marker='o')
ax2.plot(range(1, len(ppo_cumulative_rewards) + 1), np.cumsum(ppo_cumulative_rewards), label='PPO', marker='o')
ax2.plot(range(1, len(a2c_cumulative_rewards) + 1), np.cumsum(a2c_cumulative_rewards), label='A2C', marker='o')

# Add labels, title, and legend for cumulative rewards
ax2.set_ylabel('Cumulative Reward')
ax2.set_xlabel('Episode')
ax2.set_title('DQN/PPO/A2C Cumulative Reward')
ax2.legend()

fig.tight_layout()
plt.show()

env.close()
