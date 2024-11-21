import numpy as np
import matplotlib.pyplot as plt

# GridWorld environment
class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.agent_pos = (np.random.randint(size), np.random.randint(size))
        self.reward_pos = (np.random.randint(size), np.random.randint(size))
        self.steps = 0

    def reset_reward(self):
        self.reward_pos = (np.random.randint(self.size), np.random.randint(self.size))
        self.steps = 0

    def step(self, action):
        new_pos = list(self.agent_pos)
        if action == 0:  # up
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif action == 1:  # down
            new_pos[0] = min(new_pos[0] + 1, self.size - 1)
        elif action == 2:  # left
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif action == 3:  # right
            new_pos[1] = min(new_pos[1] + 1, self.size - 1)
        self.agent_pos = tuple(new_pos)
        if self.agent_pos == self.reward_pos:
            reward = 10
        else:
            reward = -1
        self.steps += 1
        if self.steps % 500 == 0:
            self.reset_reward()
        return self.agent_pos, reward

# Base Q-learning agent
class QLearningAgent:
    def __init__(self, learning_rate, size=10, actions=4):
        self.Q = np.ones((size, size, actions)) * 10  # Optimistic initialization
        self.learning_rate = learning_rate
        self.size = size
        self.actions = actions
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.001
        self.N = np.zeros((size, size, actions))

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.Q[state])

    def update_eps(self):
        self.eps = self.eps_min + (self.eps - self.eps_min) * np.exp(-self.eps_decay)

# Subclasses for learning rate strategies
class QLearningAgentCLR(QLearningAgent):
    def __init__(self, size=10, actions=4):
        super().__init__(learning_rate=0.1, size=size, actions=actions)

    def update_Q(self, state, action, reward, next_state):
        self.Q[state][action] += self.learning_rate * (reward + np.max(self.Q[next_state]) - self.Q[state][action])

class QLearningAgentALR(QLearningAgent):
    def __init__(self, size=10, actions=4):
        super().__init__(learning_rate=1.0, size=size, actions=actions)
        self.initial_learning_rate = 1.0
        self.decay_rate = 0.001
        self.t = 0

    def update_Q(self, state, action, reward, next_state):
        alpha = self.initial_learning_rate / (1 + self.decay_rate * self.t)
        self.Q[state][action] += alpha * (reward + np.max(self.Q[next_state]) - self.Q[state][action])
        self.t += 1

class QLearningAgentSAM(QLearningAgent):
    def __init__(self, size=10, actions=4):
        super().__init__(learning_rate=1.0, size=size, actions=actions)

    def update_Q(self, state, action, reward, next_state):
        alpha = 1 / (1 + self.N[state][action])
        self.Q[state][action] += alpha * (reward + np.max(self.Q[next_state]) - self.Q[state][action])
        self.N[state][action] += 1

# Main function to run the experiment
def main():
    env = GridWorld()
    agents = {
        'CLR': QLearningAgentCLR(),
        'ALR': QLearningAgentALR(),
        'SAM': QLearningAgentSAM()
    }
    total_steps = 10000
    moving_average_window = 100
    cumulative_rewards = {agent: [] for agent in agents}

    for step in range(total_steps):
        for agent_name, agent in agents.items():
            state = env.agent_pos
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update_Q(state, action, reward, next_state)
            agent.update_eps()
            cumulative_rewards[agent_name].append(reward)

    # Plotting cumulative rewards
    fig, ax = plt.subplots()
    for agent_name, rewards in cumulative_rewards.items():
        moving_avg = np.convolve(rewards, np.ones(moving_average_window)/moving_average_window, mode='valid')
        ax.plot(range(len(moving_avg)), moving_avg, label=agent_name)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Comparison of Learning Rate Strategies in Nonstationary Environment')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()