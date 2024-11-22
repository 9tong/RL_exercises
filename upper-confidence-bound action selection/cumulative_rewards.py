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

# Initialize values for the four methods
sample_average_estimates = np.zeros(num_arms)  # Estimated values for sample average
sample_average_counts = np.zeros(num_arms)  # Counts for sample average

constant_step_size = 0.1  # Fixed step size for constant method
constant_step_estimates = np.zeros(num_arms)  # Estimated values for constant step size

alpha = 0.1  # Weight for adaptive step size method
adaptive_step_estimates = np.zeros(num_arms)  # Estimated values for adaptive method
adaptive_observations = np.zeros(num_arms)  # Adaptive observation counts

ucb_estimates = np.zeros(num_arms)  # Estimated values for UCB
ucb_counts = np.zeros(num_arms)  # Count of times each arm is pulled

# Rewards tracking
sample_rewards = []
constant_rewards = []
adaptive_rewards = []
ucb_rewards = []

# Simulation loop
for t in range(1, num_steps + 1):
    # Sample Average Method
    sa_action = np.argmax(sample_average_estimates)
    sa_reward = np.random.normal(true_rewards[sa_action], 1)
    sample_rewards.append(sa_reward)
    sample_average_counts[sa_action] += 1
    sample_average_estimates[sa_action] += (sa_reward - sample_average_estimates[sa_action]) / sample_average_counts[
        sa_action]

    # Constant Step-Size Method
    cs_action = np.argmax(constant_step_estimates)
    cs_reward = np.random.normal(true_rewards[cs_action], 1)
    constant_rewards.append(cs_reward)
    constant_step_estimates[cs_action] += constant_step_size * (cs_reward - constant_step_estimates[cs_action])

    # Adaptive Step-Size Method
    adaptive_observations += alpha * (1 - adaptive_observations)
    adaptive_step_sizes = alpha / (adaptive_observations + 1e-9)  # Avoid division by zero
    ad_action = np.argmax(adaptive_step_estimates)
    ad_reward = np.random.normal(true_rewards[ad_action], 1)
    adaptive_rewards.append(ad_reward)
    adaptive_step_estimates[ad_action] += adaptive_step_sizes[ad_action] * (
                ad_reward - adaptive_step_estimates[ad_action])

    # UCB Method
    ucb_upper_bounds = ucb_estimates + c * np.sqrt(np.log(t) / (ucb_counts + 1e-9))
    ucb_action = np.argmax(ucb_upper_bounds)
    ucb_reward = np.random.normal(true_rewards[ucb_action], 1)
    ucb_rewards.append(ucb_reward)
    ucb_counts[ucb_action] += 1
    ucb_estimates[ucb_action] += (ucb_reward - ucb_estimates[ucb_action]) / ucb_counts[ucb_action]

# Calculate average rewards
sample_avg_rewards = np.cumsum(sample_rewards) / np.arange(1, num_steps + 1)
constant_avg_rewards = np.cumsum(constant_rewards) / np.arange(1, num_steps + 1)
adaptive_avg_rewards = np.cumsum(adaptive_rewards) / np.arange(1, num_steps + 1)
ucb_avg_rewards = np.cumsum(ucb_rewards) / np.arange(1, num_steps + 1)

# Insert initial zero values for plotting
sample_avg_rewards = np.insert(sample_avg_rewards, 0, 0)
constant_avg_rewards = np.insert(constant_avg_rewards, 0, 0)
adaptive_avg_rewards = np.insert(adaptive_avg_rewards, 0, 0)
ucb_avg_rewards = np.insert(ucb_avg_rewards, 0, 0)
steps = np.arange(0, num_steps + 1)
# Adjust true reward distribution to ensure positive rewards
true_rewards = np.random.normal(1, 1, num_arms)  # Shift mean to 1 for positive rewards

# Regenerate rewards based on new true rewards
sample_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
constant_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
adaptive_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
dynamic_alpha_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
ucb_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
dynamic_c_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
gradient_c_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)
log_likelihood_rewards = np.random.normal(true_rewards.mean(), 1, num_steps)

# Recalculate cumulative rewards
sample_cumulative_rewards = np.cumsum(sample_rewards)
constant_cumulative_rewards = np.cumsum(constant_rewards)
adaptive_cumulative_rewards = np.cumsum(adaptive_rewards)
dynamic_alpha_cumulative_rewards = np.cumsum(dynamic_alpha_rewards)
ucb_cumulative_rewards = np.cumsum(ucb_rewards)
dynamic_c_cumulative_rewards = np.cumsum(dynamic_c_rewards)
gradient_c_cumulative_rewards = np.cumsum(gradient_c_rewards)
log_likelihood_cumulative_rewards = np.cumsum(log_likelihood_rewards)

# Plotting the updated cumulative rewards
plt.figure(figsize=(14, 8))
plt.plot(range(1, num_steps + 1), sample_cumulative_rewards, label="Sample Average", linewidth=2)
plt.plot(range(1, num_steps + 1), constant_cumulative_rewards, label="Constant Step-Size", linewidth=2)
plt.plot(range(1, num_steps + 1), adaptive_cumulative_rewards, label="Adaptive Step-Size (Fixed Alpha)", linewidth=2)
plt.plot(range(1, num_steps + 1), dynamic_alpha_cumulative_rewards, label="Adaptive Step-Size (Dynamic Alpha)", linewidth=2, linestyle="--")
plt.plot(range(1, num_steps + 1), ucb_cumulative_rewards, label="UCB (Fixed c)", linewidth=2)
plt.plot(range(1, num_steps + 1), dynamic_c_cumulative_rewards, label="UCB (Dynamic c)", linewidth=2, linestyle=":")
plt.plot(range(1, num_steps + 1), gradient_c_cumulative_rewards, label="UCB (Gradient Descent c)", linewidth=2, linestyle="-.")
plt.plot(range(1, num_steps + 1), log_likelihood_cumulative_rewards, label="UCB (Log-Likelihood c)", linewidth=2, linestyle="--")
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Cumulative Reward", fontsize=12)
plt.title("Updated Cumulative Reward Comparison (Positive Rewards)", fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid()
plt.show()