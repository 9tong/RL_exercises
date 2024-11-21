import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
num_arms = 10  # Number of arms
num_steps = 1000  # Number of steps
true_rewards = np.random.normal(0, 1, num_arms)  # True reward for each arm
steps = np.arange(0, num_steps + 1)  # Steps including initial zero

# Methods: Sample Average, Constant Step-Size, Adaptive Step-Size
constant_step_size = 0.1
alpha = 0.1  # Adaptive step-size parameter
dynamic_alpha_learning_rate = 0.01
sigma = 1.0  # Reward variance (for log-likelihood UCB)

# Initialize Sample Average
sample_average_estimates = np.zeros(num_arms)
sample_average_counts = np.zeros(num_arms)
sample_rewards = []

# Initialize Constant Step-Size
constant_step_estimates = np.zeros(num_arms)
constant_rewards = []

# Initialize Adaptive Step-Size (Fixed Alpha)
adaptive_step_estimates = np.zeros(num_arms)
adaptive_observations = np.zeros(num_arms)
adaptive_rewards = []

# Adaptive Step-Size (Dynamic Alpha)
class DynamicAlpha:
    def __init__(self, initial_alpha=0.1, learning_rate=0.01):
        self.alpha = initial_alpha
        self.learning_rate = learning_rate

    def update(self, reward, estimate):
        error = abs(reward - estimate)
        self.alpha += self.learning_rate * (error - self.alpha)
        self.alpha = max(0.01, min(self.alpha, 1.0))
        return self.alpha

dynamic_alpha = DynamicAlpha()
dynamic_alpha_estimates = np.zeros(num_arms)
dynamic_alpha_rewards = []

# Initialize UCB (Fixed c)
c_fixed = 2
ucb_estimates = np.zeros(num_arms)
ucb_counts = np.zeros(num_arms)
ucb_rewards = []

# UCB (Dynamic c)
class DynamicC:
    def __init__(self, initial_c=2, learning_rate=0.01):
        self.c = initial_c
        self.learning_rate = learning_rate

    def update(self, reward, estimate):
        error = abs(reward - estimate)
        self.c += self.learning_rate * (error - self.c)
        self.c = max(0.01, min(self.c, 5.0))
        return self.c

dynamic_c = DynamicC()
dynamic_c_estimates = np.zeros(num_arms)
dynamic_c_counts = np.zeros(num_arms)
dynamic_c_rewards = []

# UCB (Gradient Descent c)
class GradientDescentC:
    def __init__(self, initial_c=2, learning_rate=0.01):
        self.c = initial_c
        self.learning_rate = learning_rate

    def update(self, reward, estimate, gradient):
        self.c -= self.learning_rate * gradient
        self.c = max(0.01, min(self.c, 5.0))
        return self.c

gradient_c = GradientDescentC()
gradient_c_estimates = np.zeros(num_arms)
gradient_c_counts = np.zeros(num_arms)
gradient_c_rewards = []

# UCB (Log-Likelihood c)
class LogLikelihoodC:
    def __init__(self, initial_c=2, learning_rate=0.01, sigma=1.0):
        self.c = initial_c
        self.learning_rate = learning_rate
        self.sigma = sigma

    def update(self, reward, estimate, exploration_bonus):
        error = reward - (estimate + exploration_bonus)
        gradient = (1 / self.sigma**2) * error * exploration_bonus
        self.c += self.learning_rate * gradient
        self.c = max(0.01, min(self.c, 5.0))
        return self.c

log_likelihood_c = LogLikelihoodC()
log_likelihood_estimates = np.zeros(num_arms)
log_likelihood_counts = np.zeros(num_arms)
log_likelihood_rewards = []

# Simulation Loop
for t in range(1, num_steps + 1):
    # Sample Average
    sa_action = np.argmax(sample_average_estimates)
    sa_reward = np.random.normal(true_rewards[sa_action], sigma)
    sample_rewards.append(sa_reward)
    sample_average_counts[sa_action] += 1
    sample_average_estimates[sa_action] += (sa_reward - sample_average_estimates[sa_action]) / sample_average_counts[sa_action]

    # Constant Step-Size
    cs_action = np.argmax(constant_step_estimates)
    cs_reward = np.random.normal(true_rewards[cs_action], sigma)
    constant_rewards.append(cs_reward)
    constant_step_estimates[cs_action] += constant_step_size * (cs_reward - constant_step_estimates[cs_action])

    # Adaptive Step-Size (Fixed Alpha)
    adaptive_observations += alpha * (1 - adaptive_observations)
    adaptive_step_sizes = alpha / (adaptive_observations + 1e-9)
    ad_action = np.argmax(adaptive_step_estimates)
    ad_reward = np.random.normal(true_rewards[ad_action], sigma)
    adaptive_rewards.append(ad_reward)
    adaptive_step_estimates[ad_action] += adaptive_step_sizes[ad_action] * (ad_reward - adaptive_step_estimates[ad_action])

    # Adaptive Step-Size (Dynamic Alpha)
    dynamic_alpha_observations = adaptive_observations.copy()
    da_action = np.argmax(dynamic_alpha_estimates)
    da_reward = np.random.normal(true_rewards[da_action], sigma)
    dynamic_alpha_rewards.append(da_reward)
    dynamic_alpha.alpha = dynamic_alpha.update(da_reward, dynamic_alpha_estimates[da_action])
    dynamic_alpha_step_sizes = dynamic_alpha.alpha / (dynamic_alpha_observations + 1e-9)
    dynamic_alpha_estimates[da_action] += dynamic_alpha_step_sizes[da_action] * (da_reward - dynamic_alpha_estimates[da_action])

    # UCB (Fixed c)
    ucb_upper_bounds = ucb_estimates + c_fixed * np.sqrt(np.log(t) / (ucb_counts + 1e-9))
    ucb_action = np.argmax(ucb_upper_bounds)
    ucb_reward = np.random.normal(true_rewards[ucb_action], sigma)
    ucb_rewards.append(ucb_reward)
    ucb_counts[ucb_action] += 1
    ucb_estimates[ucb_action] += (ucb_reward - ucb_estimates[ucb_action]) / ucb_counts[ucb_action]

    # UCB (Dynamic c)
    dynamic_c_upper_bounds = dynamic_c_estimates + dynamic_c.c * np.sqrt(np.log(t) / (dynamic_c_counts + 1e-9))
    dynamic_c_action = np.argmax(dynamic_c_upper_bounds)
    dynamic_c_reward = np.random.normal(true_rewards[dynamic_c_action], sigma)
    dynamic_c_rewards.append(dynamic_c_reward)
    dynamic_c_counts[dynamic_c_action] += 1
    dynamic_c_estimates[dynamic_c_action] += (
        dynamic_c_reward - dynamic_c_estimates[dynamic_c_action]
    ) / dynamic_c_counts[dynamic_c_action]
    dynamic_c.c = dynamic_c.update(dynamic_c_reward, dynamic_c_estimates[dynamic_c_action])

    # UCB (Gradient Descent c)
    gradient_c_upper_bounds = gradient_c_estimates + gradient_c.c * np.sqrt(np.log(t) / (gradient_c_counts + 1e-9))
    gradient_c_action = np.argmax(gradient_c_upper_bounds)
    gradient_c_reward = np.random.normal(true_rewards[gradient_c_action], sigma)
    gradient_c_rewards.append(gradient_c_reward)
    gradient_c_counts[gradient_c_action] += 1
    gradient_c_estimates[gradient_c_action] += (
        gradient_c_reward - gradient_c_estimates[gradient_c_action]
    ) / gradient_c_counts[gradient_c_action]
    gradient = abs(gradient_c_reward - gradient_c_estimates[gradient_c_action])
    gradient_c.c = gradient_c.update(gradient_c_reward, gradient_c_estimates[gradient_c_action], gradient)

    # UCB (Log-Likelihood c)
    log_likelihood_upper_bounds = log_likelihood_estimates + log_likelihood_c.c * np.sqrt(np.log(t) / (log_likelihood_counts + 1e-9))
    log_likelihood_action = np.argmax(log_likelihood_upper_bounds)
    log_likelihood_reward = np.random.normal(true_rewards[log_likelihood_action], sigma)
    log_likelihood_rewards.append(log_likelihood_reward)
    log_likelihood_counts[log_likelihood_action] += 1
    log_likelihood_estimates[log_likelihood_action] += (
        log_likelihood_reward - log_likelihood_estimates[log_likelihood_action]
    ) / log_likelihood_counts[log_likelihood_action]
    exploration_bonus = log_likelihood_c.c * np.sqrt(np.log(t) / (log_likelihood_counts[log_likelihood_action] + 1e-9))
    log_likelihood_c.c = log_likelihood_c.update(
        log_likelihood_reward, log_likelihood_estimates[log_likelihood_action], exploration_bonus
    )

# Calculate Average Rewards
def calculate_average_rewards(rewards):
    avg_rewards = np.cumsum(rewards) / np.arange(1, num_steps + 1)
    return np.insert(avg_rewards, 0, 0)

sample_avg_rewards = calculate_average_rewards(sample_rewards)
constant_avg_rewards = calculate_average_rewards(constant_rewards)
adaptive_avg_rewards = calculate_average_rewards(adaptive_rewards)
dynamic_alpha_avg_rewards = calculate_average_rewards(dynamic_alpha_rewards)
ucb_avg_rewards = calculate_average_rewards(ucb_rewards)
dynamic_c_avg_rewards = calculate_average_rewards(dynamic_c_rewards)
gradient_c_avg_rewards = calculate_average_rewards(gradient_c_rewards)
log_likelihood_avg_rewards = calculate_average_rewards(log_likelihood_rewards)

# Plotting the results
plt.figure(figsize=(14, 8))
plt.plot(steps, sample_avg_rewards, label="Sample Average", linewidth=2)
plt.plot(steps, constant_avg_rewards, label="Constant Step-Size", linewidth=2)
plt.plot(steps, adaptive_avg_rewards, label="Adaptive Step-Size (Fixed Alpha)", linewidth=2)
plt.plot(steps, dynamic_alpha_avg_rewards, label="Adaptive Step-Size (Dynamic Alpha)", linewidth=2, linestyle="--")
plt.plot(steps, ucb_avg_rewards, label="UCB (Fixed c)", linewidth=2)
plt.plot(steps, dynamic_c_avg_rewards, label="UCB (Dynamic c)", linewidth=2, linestyle=":")
plt.plot(steps, gradient_c_avg_rewards, label="UCB (Gradient Descent c)", linewidth=2, linestyle="-.")
plt.plot(steps, log_likelihood_avg_rewards, label="UCB (Log-Likelihood c)", linewidth=2, linestyle="--")
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.title("Comparison of All Methods", fontsize=14)
plt.legend(fontsize=12, loc='lower right')
plt.grid()
plt.show()