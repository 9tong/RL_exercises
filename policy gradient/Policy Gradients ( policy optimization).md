# Mathematical foundation of policy gradients
## The gradient
Policy gradients are a class of reinforcement learning algorithms that optimize policies by directly computing gradients of expected rewards with respect to policy parameters.  
1st, we define a **simple objective function representing the expected reward**: 
$$`J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]`$$

$`J(\pi_\theta)`$ is the reward returned by the policy $`\pi_\theta`$ also denoted as $`\pi_\theta(a | s)`$, and $`\tau`$ represents a trajectory(pair of states and actions) sampled from the policy $`\pi_\theta`$.  $`\theta`$ represents the parameters of the policy. The estimate of the expected reward $`R`$ is cumulative rewards over multiple sampled trajectories. 

2nd, focus on the core idea in RL: let the $`J(\pi_\theta)`$ moves toward direction to the **maximize cumulative rewards**.  
Think, if we compute the gradient of $`J(\pi_\theta)`$ with respect to $`\theta`$, we can update the policy parameters with **gradient ascent** to improve performance.  
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\pi_\theta)$$

Here, $`\nabla_\theta J(\pi_\theta)`$ represents the direction in which the policy parameters should be adjusted to increase the expected reward, also called the **policy gradient**. $`\alpha`$ is the discount parameter (aka learning rate).

## How to calc this gradient
## Unroll these expressions
1. $`J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]`$
2. $`R(\tau) =  \sum_{t=0}^\infty \gamma^t r_t`$
3. $`r_t`$is not directly dependent on  $`\theta`$ , it is a function of the trajectory $`\tau`$ , which is sampled from  $`\pi_\theta`$ . The dependence of $`\nabla_\theta J(\theta)`$  on $`\theta`$  is through the policy  $`\pi_\theta`$
4. The trajectory  $`\tau`$  is sampled with the probability:
  $$P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t)$$
, where $`\rho_0(s_0)`$is the initial state of the distribution, $`\pi`$ is the policy of choosing action $`a_t`$ in state $`s_t`$.
5. Because $`J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]`$, so,   $J(\pi_\theta) = \int P(\tau | \theta) R(\tau) d\tau$
>[!Tip]
>This integral representation stems from the **definition of expected value** in probability theory. The expectation of a random variable  X  under a probability distribution  P(x)  is: **$$\mathbb{E}[X] = \int P(x) X(x) dx$$**
6. And, $\nabla_\theta J(\pi_\theta) = \nabla_{\theta}\int P(\tau | \theta) R(\tau) \, d\tau$, by using the trick of log-likelihood derivative , $\nabla_\theta P(\tau | \theta) = P(\tau | \theta) \nabla_\theta \log P(\tau | \theta)$, so, we can rewrite $\nabla_\theta J(\pi_\theta)$ as $\int P(\tau | \theta) R(\tau) \nabla_\theta \log P(\tau | \theta) d\tau$.
>[!Tip]
>The log-likelihood trick simplifies gradient computation by converting the derivative of a probability into the product of the probability and the derivative of its log. This is particularly useful in reinforcement learning, as it allows us to express gradients in terms of expectations, which can be estimated using sampled trajectories.
> 
>where $N$ is the number of sampled trajectories $\tau_i$. From calculus, the derivative of the natural logarithm is:
>
>$$\frac{d}{dx} \log x = \frac{1}{x}$$
>
>Using the chain rule, we can generalize this for any function  $f(x)$:
>
>$$\frac{d}{dx} \log f(x) = \frac{1}{f(x)} \cdot \frac{df(x)}{dx}$$
>
>Rearranging:
>
>$$\frac{df(x)}{dx} = f(x) \cdot \frac{d}{dx} \log f(x)$$
>
7. Then, return to represent the expression into expected format:
   
$$`\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [ R(\tau) \nabla_\theta \log P(\tau | \theta) ]`$$

8. Then, decompose $P(\tau | \theta)$ into its components, specifically $\pi_\theta(a_t | s_t)$, and rewrite the gradient as
    
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T R(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

## As a result
This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories from the policy, we can approximate the gradient as the average of $R(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t)$ over those trajectories. We find a way to calculate the policy gradient $\nabla_\theta J(\pi_\theta)$ by the computable value  $\nabla_{\theta} \log \pi_{\theta}(a|s)$.

# Experiment
To validate the policy gradient method, we can design an experiment where an agent learns to maximize rewards in a simple environment, such as a grid world or cart-pole balancing task. We initialize the policy parameters $\theta$ randomly and iteratively update them using the gradient ascent formula $\theta \leftarrow \theta + \alpha \nabla_\theta J(\pi_\theta)$. By sampling trajectories, estimating rewards, and computing gradients, we can observe how the agent's performance improves over time. The results can be visualized by plotting cumulative rewards against training iterations.

## Experiment Design: the Grid World
To validate the policy gradient method, we can design a simple yet effective experiment where an agent learns to maximize rewards in a **grid world** or the **cart-pole balancing task**. Here’s how we can structure the experiment:
### Setup
• A 5x5 grid where the agent starts at a random position.
• The goal is to reach a target cell with maximum reward  $R = +10$ 
• Stepping into non-target cells incurs a small penalty $R = -0.1$
### Actions
• The agent can move in four directions: up, down, left, and right.
### Dynamics
• Actions succeed with 80% probability.
• With 20% probability, a random action is taken instead.
### Policy Parameterization
• Define a stochastic policy$\pi_\theta(a | s)$ using a softmax function:$\pi_\theta(a | s) = \frac{\exp(\theta_{s,a})}{\sum_{a{\prime}} \exp(\theta_{s,a{\prime}})}$
where $\theta$ represents the policy parameters for all state-action pairs.
### Experiment Steps
#### 1. Initialize
Randomize $\theta$ (small values, e.g., Gaussian noise).
#### 2. Trajectory Sampling
For each episode, sample a trajectory  $\tau = \{(s_0, a_0), (s_1, a_1), \dots\}$ using the current policy $\pi_\theta$ .
#### 3. Reward Estimation
Compute the cumulative discounted reward for each state-action pair along the trajectory   $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$
#### 4. Policy Gradient Update
Use the policy gradient formula to compute the update:  $\nabla_\theta J(\theta) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t | s_t) G_t$.
Update using gradient ascent:  $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$,
where $\alpha$ is the learning rate.
#### 5. Repeat
Continue sampling trajectories, computing rewards, and updating $\theta$ for a fixed number of episodes.
![|400](pg.png)
