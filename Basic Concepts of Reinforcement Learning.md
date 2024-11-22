
>[!tip]
>RL is the study of agents and how they learn by trial and error. It formalizes the idea that rewarding or punishing an agent for its behavior makes it more likely to repeat or forego that behavior in the future.

The main idea of RL is an AGENT is trying to learn how to **BEHAVIOR** in a **GIVEN ENVIRONMENT** for receiving a maximum **CUMULATIVE** **REWARD**, by interacting with it and adjusting its actions based on feedback.
![Pasted image 20241101112551](https://github.com/user-attachments/assets/584f1548-c2f3-4bb6-aca9-d49863d2ab64)

# Terminology
## States & Observations
A state in RL represents the complete description of the environment which the agent lives in. There is no information hidden.
An agent's observation could be total completely or only partially, in RL, observations are defined as :
- Fully-observation: agent can get the complete state of the environment;
- Partially-observation: agent can not get the complete state of the environment;

## Action spaces
Action spaces define the set of all possible **actions** an agent can take in a given environment. They can be **discrete**, where the agent chooses from a finite set of actions, or **continuous**, where the agent selects from a range of possible actions.
- A discrete action space may looks like: {move left, move right, jump, crouch};
- A continuous action space may look like: selecting a num between {min=0, max=100};

## Policies
A policy in RL defines the strategy that the agent uses to determine its actions at each state. It can be **deterministic**, where a specific action is chosen for each state, or **stochastic**, where actions are chosen according to a probability distribution.
- A deterministic policy means: the agent always selects the same action for a given state. In normal, notation in deterministic policy will be represented as  $a = \mu(s)$， $\mu(s)$ is a value which means the only action will be taken by the agent.
- A stochastic policy means: the agent selects actions based on a probability distribution, so the same state may lead to different actions. In normal, notation in continuous policy will be represented as: $a \sim \pi(a|s)$, $\pi(a | s)$ is a probability distribution which means where the action will be sampled from. Usually, categorical policies and diagonal Gaussian policies will be taken in the stochastic scene: 
	- Categorical policy, used in **discrete action spaces**, samples actions from a probability distribution over a finite set of actions.
	- Diagonal Gaussian policies, used in **continuous action spaces**, sample actions from a multivariate normal distribution where the covariance matrix is diagonal.

## Trajectories
A trajectory in RL refers to the sequence of states, actions, and rewards that an agent experiences as it interacts with the environment over time. It represents the path taken by the agent from an initial state to a terminal state or over a fixed time horizon.
Trajectories notation as $\tau$, which looks like:  $\tau$ = $(s_0, a_0, s_1, a_1, \dots)$, they mean every states in this world, and we take what actions for every states. The very start of the $\tau$ collection, which is state $s_0$, is where the agent first observes the environment and selects its initial action, is usually denoted as: $\rho_0$, and $s_0$ is sampled from  $\rho_0$, $s_{0} \sim \rho_0(\cdot)$.
The state transition is represented by:
- Stochastic: a probability distribution $s_{t+1} \sim P(\cdot |s_t, a_t)$, which defines the **likelihood** of transitioning to state $s_{t+1}$ after taking action $a$ in state $s$ at moment $t$;
- Deterministic: a fixed function $s_{t+1} = f(s_t, a_t)$, where the next state $s_{t+1}$ is uniquely determined by the current state $s$ and action $a$ at moment $t$.
## Reward
Reward function which is the most critical important in RL. A reward in RL is a **scalar value** that the agent receives after taking an action in a particular state. It serves as feedback, indicating how good or bad the action was in terms of achieving the agent's goal. The agent's objective is to maximize the **cumulative reward** over time.
A reward function is notated as: $r_{t} = R(s_t, a_{t,} s_{t+1})$, which means a reward is decided by the state now, the action will be taken now, and the state transition after the action taken. And remember, RL need to calculate the **CUMULATIVE** **REWARD** of the agent decision. Normally, we will notated as $R(\tau)$ for the global return, also represent as $R(\tau) = \sum_{t=0}^{T} r_t$. This cumulative reward is what the agent seeks to maximize over its interactions with the environment, but it has a problem, $R(\tau)$ will never converge in this equation, the reward value will go to infinite which is not what we want. To address this, we introduce a discount factor $\gamma \in [0, 1]$ to ensure that future rewards are weighted less, preventing the sum from diverging.
So, the true reward functions should looks like: $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$, this discount factor $\gamma$ ensures that the agent prioritizes immediate rewards while still considering future rewards, allowing for more stable and convergent learning. This is make sense on intuitive level.
## Value functions
Value functions estimate how good it is for the agent to be in a particular state (or to perform a particular action in that state) in terms of expected cumulative reward. There are two main types of value functions: the **state-value function** $V(s)$, which gives the expected return starting from state $s$, and the **action-value function** $Q(s, a)$, which gives the expected return starting from state $s$ and taking action $a$.
State-value function $V(s)$ means: the expected cumulative reward the agent will receive starting from state $s$ and following a particular policy. This function only consider the relation between state and the policy. 
Similarly, the action-value function $Q(s, a)$ represents the expected cumulative reward starting from state $s$, taking action $a$, and then following a particular policy. This function considers both the state and the action, making it more informative for decision-making compared to the state-value function.
With a given policy $\pi$ : the state-value function $V^\pi(s)$ and the action-value function $Q^\pi(s, a)$ can be computed as the expected cumulative reward when following policy $\pi$, and the optimal value of the optimal policy could be denoted as: $`V^*(s)`$ and $`Q^*(s, a)`$, while,
- The equation for $V^\pi(s)$ expresses the recursive relationship between the value of a state and the values of subsequent states: $V^{\pi}(s) = \mathbb{E}{\tau \sim \pi}[{R(\tau)\left| s_0 = s\right.}]$. Similarly, the  equation for $Q^\pi(s, a)$ is $`Q^{\pi}(s,a) = \mathbb{E}{\tau \sim \pi}[{R(\tau)\left| s_0 = s, a_0 = a\right.}]`$.
- The optimality equation for $V^*(s)$ is $`V^*(s) = \max_{\pi} \mathbb{E}_{\tau \sim \pi}[R(\tau) | s_{0}= s]`$. Similarly, the optimality equation for $`Q^*(s, a)$ is $Q^*(s, a) = \max_{\pi} \mathbb{E}_{\tau \sim \pi}[R(\tau) | s_{0}= s, a_{0}= a]`$.
>[!tip]
>What's the optimal eq for the $`Q^*(s, a)`$ means? 
>$`Q^*(s, a)`$  tell you if you are in state $`s`$ , and you take whatever action  $`a`$ right now, **how much return** can you expect if you act perfectly from this point on.
>$`Q^*`$ function is the key point to find the best policy. If agent continue to take actions in the enviroment by following the policy here,  it will maximize the expected cumulative reward.

Another approach is these explaination of $Q$ and $V$ reveal the relationship between  $`V^\pi(s)`$ and  $`Q^\pi(s, a)`$:  $`V^\pi(s) = \mathbb{E}_{a \sim \pi(s)}[Q^\pi(s, a)]`$. Similarly, the relationship between the optimal state-value function and the optimal action-value function is:
$`V^*(s) = \max_a Q^*(s, a)`$
Eq1 indicate the expected value of the action-value function under policy $`\pi`$ , which equals the state-value function for policy $`\pi$`$.
Eq2 represent the optimal value of a state is the maximum action-value across all actions, which forms the basis for optimal decision-making (action-taking).

This brings us to the **Bellman equation**, which is a key concept in reinforcement learning: 
**The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.**
Formally, the Bellman equation for the state-value function is written as:
$`V^\pi(s) = \mathbb{E}_{a \sim \pi(s)}\left[r(s, a) + \gamma V^\pi(s')\right]`$
And for the action-value function:
$`Q^\pi(s, a) = \mathbb{E}\left[r(s, a) + \gamma Q^\pi(s', a')\right]`$
## Advantage Functions
Advantage functions measure how much better taking a specific action is compared to the average action in a given state. Formally, the advantage function is defined as $`A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)`$. It helps in determining whether an action is better or worse than the expected value (average value) of being in that state under policy $\pi$. This is a simple definition.


