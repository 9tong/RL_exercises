import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Number of random points to sample
num_samples = 50000

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title("Monte Carlo Simulation for Estimating Pi")

# Draw axes crossing at the origin
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Draw unit circle
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
ax.add_artist(circle)

# Initialize point plots
#inside_points, = ax.plot([], [], 'go', markersize=2, label='Inside Circle')
#outside_points, = ax.plot([], [], 'ro', markersize=2, label='Outside Circle')

inside_points, = ax.plot([], [], 'go', markersize=2)
outside_points, = ax.plot([], [], 'ro', markersize=2)

# Data lists
x_inside, y_inside = [], []
x_outside, y_outside = [], []

# Generate random points
x_points = np.random.uniform(-1, 1, num_samples)
y_points = np.random.uniform(-1, 1, num_samples)


# Animation function
def update(i):
    if i >= num_samples:
        ani.event_source.stop()
        return

    x, y = x_points[i], y_points[i]
    if x ** 2 + y ** 2 <= 1:
        x_inside.append(x)
        y_inside.append(y)
    else:
        x_outside.append(x)
        y_outside.append(y)

    # Update the scatter plots
    inside_points.set_data(x_inside, y_inside)
    outside_points.set_data(x_outside, y_outside)

    # Update pi estimate and title
    total_points = len(x_inside) + len(x_outside)
    pi_estimate = 4 * len(x_inside) / total_points
    ax.set_title(f"Monte Carlo Simulation for Estimating Pi\nEstimated Pi: {pi_estimate:.5f}")


# Create animation
ani = FuncAnimation(fig, update, frames=num_samples, interval=10, repeat=False)

# Add legend
ax.legend()

plt.show()
