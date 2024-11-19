import numpy as np
import matplotlib.pyplot as plt

# MAB problem experiment: 10-armed bandit for non-stationary problem
# Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for non-stationary problems.
# Goal: compare the difficulties of average sampling method and the optimal action-value method.

# Use a modified version of the 10-armed test bed in which all the q*(a) start out equal
# and then take independent random walks(say by adding a normally distributed increment with
# mean zero and standard deviation 0.01 to all the q*(a) on each step).
# Prepare plots for an action-value method using sample averages, incrementally computed,
# and another action-value method using a constant step-size parameter, \alpha = 0.1.
# Use \epsilon =0.1 and longer runs, about 2,000 runs and 10,000 steps

#

# Define the 10-armed testbed environment
class NonstationaryBandit:
    def __init__(self, k=10, mean=0, std_dev=0.01):
        self.k = k
        self.q_true = np.zeros(k) + mean  # True action values
        self.std_dev = std_dev
        self.best_action = np.argmax(self.q_true)

    def step(self):
        """Perform a random walk on the true action values."""
        self.q_true += np.random.normal(0, self.std_dev, self.k)
        self.best_action = np.argmax(self.q_true)

    def reward(self, action):
        """Provide reward for the chosen action."""
        return np.random.normal(self.q_true[action], 1)


# Define an agent
class Agent:
    def __init__(self, k=10, epsilon=0.1, step_size=None):
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size
        self.q_estimates = np.zeros(k)  # Estimated action values
        self.action_counts = np.zeros(k)  # Action counts for sample-average method
        self.time = 0

    def select_action(self):
        """
        Epsilon-greedy action trade-off. If action < epsilon do a random action(exploration),
        else do a greedy search, argmax(exploitation).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def update_estimates(self, action, reward):
        """Update action value estimates."""
        self.time += 1
        if self.step_size:  # Constant step-size parameter
            # Q_{n+1} = Q_{n}+ \alpha[R_{n} - Q_{n}]
            # \alpha: learning rate
            self.q_estimates[action] += self.step_size * (reward - self.q_estimates[action])
        else:  # Sample-average method
            self.action_counts[action] += 1
            self.q_estimates[action] += (1 / self.action_counts[action]) * (reward - self.q_estimates[action])


# Run the experiment
def run_experiment(runs=2000, steps=10000, epsilon=0.1, step_size=None, std_dev=0.01):
    k = 10
    rewards = np.zeros((runs, steps))
    optimal_action_counts = np.zeros((runs, steps))

    for run in range(runs):
        bandit = NonstationaryBandit(k=k, std_dev=std_dev)
        agent = Agent(k=k, epsilon=epsilon, step_size=step_size)

        for step in range(steps):
            action = agent.select_action()
            reward = bandit.reward(action)

            # Update estimates and environment
            agent.update_estimates(action, reward)
            bandit.step()

            # Track metrics
            rewards[run, step] = reward
            if action == bandit.best_action:
                optimal_action_counts[run, step] = 1

    return rewards.mean(axis=0), optimal_action_counts.mean(axis=0)


# Conduct experiments
sample_avg_rewards, sample_avg_optimal = run_experiment(runs=200, steps=2000, step_size=None)
constant_step_rewards, constant_step_optimal = run_experiment(runs=200, steps=2000, step_size=0.1)

# Plot results
plt.figure(figsize=(12, 6))

# Plot average rewards
plt.subplot(1, 2, 1)
plt.plot(sample_avg_rewards, label="Sample-Average")
plt.plot(constant_step_rewards, label="Constant Step-Size (α=0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Reward over Time")

# Plot optimal action percentages
plt.subplot(1, 2, 2)
plt.plot(sample_avg_optimal * 100, label="Sample-Average")
plt.plot(constant_step_optimal * 100, label="Constant Step-Size (α=0.1)")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Percentage over Time")

plt.tight_layout()
plt.show()