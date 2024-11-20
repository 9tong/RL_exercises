import numpy as np
import matplotlib.pyplot as plt

# Parameters
true_value = 1.0  # The true underlying value we are estimating
high_initial_bias = 10.0  # High biased initial estimate
num_steps = 100  # Number of steps to simulate
constant_step_size = 0.1  # Step size for constant step-size method

# Generate random observations around the true value
np.random.seed(42)  # For reproducibility
observations = np.random.normal(loc=true_value, scale=1.0, size=num_steps)

# Initialize estimates
sample_average_estimates = [high_initial_bias]
constant_step_estimates = [high_initial_bias]

# Simulation
for t in range(1, num_steps + 1):
    # Sample Average Method
    sample_avg = sample_average_estimates[-1] + (1 / t) * (observations[t - 1] - sample_average_estimates[-1])
    sample_average_estimates.append(sample_avg)

    # Constant Step-Size Method
    constant_step = constant_step_estimates[-1] + constant_step_size * (
                observations[t - 1] - constant_step_estimates[-1])
    constant_step_estimates.append(constant_step)

# Parameters for the new adaptive method
alpha = 0.1  # Weight for the step size and observation adjustment
adaptive_estimates = [high_initial_bias]
average_observations = 0  # Initial average observation count

# Simulation for the new adaptive step size method
for t in range(1, num_steps + 1):
    # Update the average number of observations
    average_observations = average_observations + alpha * (1 - average_observations)
    # Calculate the adaptive step size
    adaptive_step_size = alpha / average_observations
    # Update the estimate using the adaptive step size
    adaptive_estimate = adaptive_estimates[-1] + adaptive_step_size * (observations[t - 1] - adaptive_estimates[-1])
    adaptive_estimates.append(adaptive_estimate)

# Plotting the results with the new adaptive method
plt.figure(figsize=(12, 6))
plt.plot(range(num_steps + 1), sample_average_estimates, label="Sample Average Method", linewidth=2)
plt.plot(range(num_steps + 1), constant_step_estimates, label="Constant Step-Size Method", linewidth=2)
plt.plot(range(num_steps + 1), adaptive_estimates, label="Adaptive Step-Size Method", linewidth=2, linestyle="--")
plt.axhline(y=true_value, color="gray", linestyle="--", label="True Value (1.0)")
plt.axhline(y=high_initial_bias, color="red", linestyle="--", label="Initial Bias (10.0)")
plt.xlabel("Step", fontsize=12)
plt.ylabel("Estimate", fontsize=12)
plt.title("Comparison of Sample Average, Constant, and Adaptive Step-Size Methods", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()