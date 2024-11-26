import numpy as np
import matplotlib.pyplot as plt

# Define the environment
grid_size = 5
target = (4, 4)
gamma = 0.99
alpha = 0.1
num_actions = 4

# Initialize policy parameters
theta = np.random.randn(grid_size, grid_size, num_actions)

# Softmax policy
def softmax_policy(state, theta):
    logits = theta[state[0], state[1]]
    exp_logits = np.exp(logits - np.max(logits))  # Stability
    return exp_logits / np.sum(exp_logits)

# Sample an action
def sample_action(state, theta):
    probs = softmax_policy(state, theta)
    return np.random.choice(len(probs), p=probs)

# Define environment dynamics
def transition(state, action):
    x, y = state
    if action == 0:  # Up
        x = max(0, x - 1)
    elif action == 1:  # Down
        x = min(grid_size - 1, x + 1)
    elif action == 2:  # Left
        y = max(0, y - 1)
    elif action == 3:  # Right
        y = min(grid_size - 1, y + 1)

    new_state = (x, y)
    reward = 10 if new_state == target else -0.1
    return new_state, reward

# Generate a trajectory
def generate_trajectory(theta):
    trajectory = []
    state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    total_reward = 0
    for _ in range(50):  # Maximum steps per episode
        action = sample_action(state, theta)
        next_state, reward = transition(state, action)
        trajectory.append((state, action, reward))
        state = next_state
        total_reward += reward
        if state == target:
            break
    return trajectory, total_reward

# Update policy
def update_policy(theta, trajectory):
    for t, (state, action, reward) in enumerate(trajectory):
        G_t = sum([gamma ** k * step[2] for k, step in enumerate(trajectory[t:])])  # Cumulative reward
        probs = softmax_policy(state, theta)
        grad = np.zeros_like(theta)
        grad[state[0], state[1], :] = -probs
        grad[state[0], state[1], action] += 1  # Policy gradient
        theta += alpha * grad * G_t

# Training loop
rewards = []
for episode in range(1000):
    trajectory, total_reward = generate_trajectory(theta)
    update_policy(theta, trajectory)
    rewards.append(total_reward)

# Plot cumulative rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Policy Gradient Performance in Grid World')
plt.show()