import numpy as np
import matplotlib.pyplot as plt

# Multi-armed bandit simulation parameters
num_arms = 10  # Number of arms
num_steps = 1000  # Number of time steps
true_rewards = np.random.normal(0, 1, num_arms)  # True reward for each arm (stationary)

# Methods: UCB and epsilon-greedy
epsilon = 0.1  # Exploration rate for epsilon-greedy
c = 2  # Exploration parameter for UCB

# Initialize values for UCB
ucb_estimates = np.zeros(num_arms)  # Estimated values for each arm
ucb_counts = np.zeros(num_arms)  # Count of times each arm is pulled

# Initialize values for epsilon-greedy
eg_estimates = np.zeros(num_arms)  # Estimated values for each arm
eg_counts = np.zeros(num_arms)  # Count of times each arm is pulled

# Rewards tracking
ucb_rewards = []
eg_rewards = []

# Simulation loop
for t in range(1, num_steps + 1):
    # UCB action selection
    ucb_upper_bounds = ucb_estimates + c * np.sqrt(np.log(t) / (ucb_counts + 1e-9))  # Add a small value to avoid div by 0
    ucb_action = np.argmax(ucb_upper_bounds)
    ucb_reward = np.random.normal(true_rewards[ucb_action], 1)  # Sample reward for selected arm
    ucb_rewards.append(ucb_reward)
    ucb_counts[ucb_action] += 1
    ucb_estimates[ucb_action] += (ucb_reward - ucb_estimates[ucb_action]) / ucb_counts[ucb_action]

    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        eg_action = np.random.randint(num_arms)  # Explore: random action
    else:
        eg_action = np.argmax(eg_estimates)  # Exploit: select best estimate
    eg_reward = np.random.normal(true_rewards[eg_action], 1)  # Sample reward for selected arm
    eg_rewards.append(eg_reward)
    eg_counts[eg_action] += 1
    eg_estimates[eg_action] += (eg_reward - eg_estimates[eg_action]) / eg_counts[eg_action]

# Calculate average rewards
ucb_average_rewards = np.cumsum(ucb_rewards) / np.arange(1, num_steps + 1)
eg_average_rewards = np.cumsum(eg_rewards) / np.arange(1, num_steps + 1)

# Plotting the average rewards
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_steps + 1), ucb_average_rewards, label="UCB", linewidth=2)
plt.plot(range(1, num_steps + 1), eg_average_rewards, label="Epsilon-Greedy", linewidth=2)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.title("Average Reward: UCB vs Epsilon-Greedy", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()