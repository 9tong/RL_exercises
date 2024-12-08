import numpy as np
import random


class GridworldEnv:
    def __init__(self, width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2)], step_cost=-0.01,
                 goal_reward=1.0):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.step_cost = step_cost
        self.goal_reward = goal_reward

        # Define action space: up, down, left, right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        # State: represented by (row, col)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        # Compute next position
        dr, dc = self.actions[action]
        next_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        # Check boundaries and obstacles
        if (0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width and next_pos not in self.obstacles):
            self.agent_pos = next_pos

        # Compute reward
        if self.agent_pos == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_cost
            done = False

        return self.agent_pos, reward, done

    def render_1(self):
        grid = np.full((self.height, self.width), ' ')
        for r, c in self.obstacles:
            grid[r, c] = 'X'
        grid[self.goal[0], self.goal[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'

        print("\n".join(["".join(row) for row in grid]))
        print()

    def render(self):
        # Prepare a character matrix for rendering
        grid = np.full((self.height, self.width), ' ')
        for r, c in self.obstacles:
            grid[r, c] = 'X'
        grid[self.goal[0], self.goal[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'

        # Print the top boundary
        top_boundary = "+" + "---+" * self.width
        print(top_boundary)

        # Print each row
        for r in range(self.height):
            # Row of cells
            row_str = "|"
            for c in range(self.width):
                row_str += " {} |".format(grid[r, c])
            print(row_str)
            # Bottom boundary of the row
            print(top_boundary)

# Simple Q-Learning Implementation
def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q-table: rows = height, columns = width, actions = 4
    Q = np.zeros((env.height, env.width, len(env.actions)))

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                # Explore
                action = random.choice(list(env.actions.keys()))
            else:
                # Exploit
                action = np.argmax(Q[state[0], state[1], :])

            next_state, reward, done = env.step(action)

            # Q-learning update
            best_next = np.max(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (reward + gamma * best_next - Q[state[0], state[1], action])

            state = next_state

    return Q


# Example usage
if __name__ == "__main__":
    # Create environment
    env = GridworldEnv(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 1), (2, 3), (2, 4), (3, 1)])

    # Learn the policy
    Q = q_learning(env, episodes=1000)

    # Test the learned policy
    state = env.reset()
    done = False
    print("Final Policy Demonstration:")
    while not done:
        env.render()
        print("Next move ------------------------")
        action = np.argmax(Q[state[0], state[1], :])
        state, reward, done = env.step(action)
    env.render()
    print("Goal reached with reward:", reward)