
The LunarLander problem is a classic benchmark problem in reinforcement learning (RL), often used to demonstrate the application of RL algorithms to control tasks.
The goal is to control a lander to safely touch down on a designated landing pad on the surface of the moon. The environment simulates basic physics, such as gravity and inertia. It also includes stochastic elements like wind and requires precise control to balance fuel efficiency and landing accuracy.

<img width="360" alt="Pasted image 20241204084335" src="https://github.com/user-attachments/assets/317590d0-1429-4fae-97d3-988199344b90">

# The State Space 
The state is represented by an 8-dimensional vector:
1. x (horizontal position of the lander)
2.	y (vertical position of the lander)
3.	vx (horizontal velocity)
4.	vy (vertical velocity)
5.	θ (orientation of the lander)
6.	vθ (angular velocity)
7.	leg1_contact (whether the left leg is touching the ground: binary)
8.	leg2_contact (whether the right leg is touching the ground: binary)

# The Action Space
 The action space consists of discrete actions: 
 1. do nothing;
 2. fire the main engine;
 3. fire the left orientation engine;
 4. fire the right orientation engine.

# The Reward(positive and negative)

| <center>Positive</center>                      | <center>Negative</center>             |
| ---------------------------------------------- | ------------------------------------- |
| Moving closer to the landing pad               | Moving away from the pad              |
| Touching down softly                           | Firing engines(excessive use of fuel) |
| Landing upright with both legs in contract     | Crashing or tipping over              |
| Resist the wind and self-refine the env change | Ending up far from the landing place  |

# The Solver
A common approach to solving the LunarLander problem is to use reinforcement learning algorithms such as Deep Q-Learning (DQN) or Proximal Policy Optimization (PPO). These algorithms learn a policy by interacting with the environment, optimizing rewards, and improving control over time.
Note that, the lunar-landing experiment of OpenAI Gym has 2 versions: **discrete** and **continuous** versions.  The main difference lies in the **action space**, where the **discrete version has a finite set of actions**, while the **continuous version allows for a range of continuous control inputs**.
## Discrete action space
Each action corresponds to a fixed thrust $\rho$ value, $\forall \mathbf{a} \ \ \epsilon \ \ \mathbf{A}[none, a_{main}, a_{left}, a_{right}]$,  which makes the problem much easier to solve.
## Continuous action space
Each action corresponds to a scalar value which is sampled from a continuous range, typically between 0 and 1 or -1 and 1, depending on the environment's configuration. Can be represented as: 
$\mathbf{a} = \mathbf{A}[a_{main}, a_{left}, a_{right}]$, where $a_{main} \sim [0, 1]$, $a_{left}, a_{right} \sim [-1, 1]$

There should be different strategies for solving 2 kinds of action spaces.  For the discrete version, algorithms like DQN are often preferred, while for the continuous version, PPO or other policy-gradient methods are more suitable. 
