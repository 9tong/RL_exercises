<img width="320" alt="image" src="https://github.com/user-attachments/assets/ea48f573-e8ca-4a55-9a64-b07dee36dac1">

The **pole balancing problem**, also known as the **cart-pole problem**, is a classic control problem in reinforcement learning and control theory. It involves balancing a pole upright on a moving cart by applying forces to the cart. The goal is to keep the pole balanced upright for as long as possible by applying forces to the cart.
If we see the pole balancing problem with RL it can be framed as follows:
- **State Space:** 
	- Cart's position and velocity.  
	- Pole's angle and angular velocity.  
- **Action space:**
	- Apply force to the left or right of the cart.  
- **Reward:**
	- A positive reward is provided for each timestep the pole remains upright.
	- The episode terminates when:
		- The pole angle exceeds a threshold.
		- The cart moves out of bounds on the track.

The cart-pole system is a classic example of an **unstable dynamic system**, meaning small deviations from equilibrium can quickly grow, making it difficult to control.
To solve this, the agent must learn to:
1. Balance the pole by applying corrective forces.
2. Anticipate the pole’s dynamics (e.g., counteract angular velocity before the pole tilts too far).

## DQN Solver on OpenAI gym
Using openai gym and DQN to solve the CartPole problem, the agent learns to balance the pole by approximating the Q-values for each state-action pair and updating its policy through experience replay and target network stabilization.
- Using CartPole-v1
- Agent setup by MLPpolicy
- Steps 10,000

The training code: https://github.com/9tong/basics_of_reinforcement_learning/blob/main/pole%20balancing/train_dqn.py

## Comparing DQN/PPO/A2C
In the CartPole balancing task, Proximal Policy Optimization (PPO) often outperforms Deep Q-Networks (DQN) and Advantage Actor-Critic (A2C) in terms of learning efficiency and stability. PPO's design includes mechanisms that limit the extent of policy updates during training, which helps maintain stability and prevent performance collapse.

Comparative studies have shown that PPO **achieves higher rewards more consistently and converges faster** than DQN and A2C in the CartPole environment. For instance, an empirical study comparing these algorithms found that PPO not only reached optimal performance more rapidly but also maintained it more reliably across different training runs.

>[!tip]
>Note that while PPO generally performs better in cart pole balancing tasks.

**Sample Efficiency**: PPO tends to learn optimal policies faster than DQN, requiring fewer training samples to achieve comparable performance.
**Effective Exploration and Exploitation**: By integrating both policy and value networks (actor-critic architecture), PPO effectively balances exploring new actions and exploiting known rewarding actions. This balance is crucial in environments like CartPole, where timely and appropriate actions are necessary to maintain balance
**Robustness**: When the CartPole environment is subjected to disturbances, such as varying initial pole angles or external forces, PPO demonstrates a higher tolerance and maintains stability better than DQN.
**Stability and Convergence**: PPO's design inherently limits the magnitude of policy updates, which contributes to more stable and reliable learning compared to DQN.

![dqn_ppo_a2c](https://github.com/user-attachments/assets/abfb4179-4736-43a2-b624-7937dd323823)
