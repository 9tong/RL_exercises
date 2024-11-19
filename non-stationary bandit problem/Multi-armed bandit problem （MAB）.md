# Introduction of MAB
The multi-armed bandit problem is a classic problem in reinforcement learning. Essentially, it is about predicting **how to achieve the maximum reward from multiple systems with different initial states**.
Assuming you can predict the next state of a particular machine, once you discover the most likely path for a single machine to achieve the maximum reward, how do you balance the maximum reward of a single machine with that of all the slot machines as a whole? This problem then becomes a trade-off between local optimization and global optimization.

# Balancing Exploitation and Exploration in MAB
In the MAB problem, the focus is on discussing the two concepts of **exploitation** and **exploration** in reinforcement learning strategies:
- Exploitation refers to when to tend towards a **local optimum**, that is, to further "drill down" on a single observation of reward of status now; This often involves selecting the current **best-known options** (those believed to be close to a **local optimum**) rather than exploring other possibilities,
- Exploration which seeks to investigate and gather more information about potentially better solutions, refers to "expanding" more possibilities, with a macro tendency towards "exhaustiveness" to achieve a greater possibility of obtaining the maximum reward;
Therefore, finding a **balance** between exploitation and exploration is key to obtaining the optimal reward strategy in the multi-armed bandit problem.

# Stationary environment & nonstationary environment
A **Stationary Bandit problem** assumes that the reward distributions of the arms remain constant over time. In this case, the optimal strategy can be learned and exploited without needing to adjust for changes in the environment.
This makes it easier to learn and exploit the best action without needing to adapt to changes.
A **Nonstationary Bandit problem** is a variant of the classic **Multi-Armed Bandit (MAB)** problem where the reward distributions of the arms (or actions) change over time. 
Unlike the stationary MAB problem, where the rewards for each arm are drawn from a fixed distribution, the Nonstationary Bandit problem involves an evolving environment, making it more challenging to identify and exploit the best actions over time.

# Value estimation for the two situations
In stationary environments, one common method to estimate the value of an action is the **Sample-Average Method**, where the value of an action is updated by averaging all observed rewards for that action. This method works well when the environment does not change over time. The update rule of sample average method is:
$`Q_{n+1} = Q_{n} + \frac{1}{n} \left( R_n - Q_n \right)`$
However, in nonstationary environments, this method may not perform well because it gives equal weight to all past observations, even if they are outdated. We often use the **Constant Step-Size Method**, which updates the value estimate by giving more weight to recent rewards, allowing it to adapt more quickly to changes in the environment. The update rule of constant step size method is:
$Q_{n+1} = Q_n + \alpha \left( R_n - Q_n \right)$
while $\alpha$ is the step size, aka learning rate.

# Tracking the nonstationary problem
In nonstationary environments, the reward probabilities can change over time, so it is important to adjust the strategy dynamically. One way to do this is by using a weighted average that gives more importance to recent observations. So the equation may look like: $Q_{t+1} = (1-\alpha)Q_t + \alpha R_t$, also the same meaning of  $Q_{n+1} = Q_{n}+ \alpha[R_{n} - Q_{n}]$.
The $\alpha$ parameter  is called **a constant step size**, aka the **learning rate**.
This ensures that older observations have exponentially **decreasing influence**, while more recent observations are **prioritized**. And the total weight sums to 1:$`(1-\alpha)^{n} + \Sigma^{n}_{i=1} \alpha(1-\alpha)^{n-i} = 1`$
This equation represents a **weighted average** where recent observations are given more weight, which is useful in nonstationary environments where the reward probabilities change over time. 
If $\alpha$ goes to 0, **the exponential moving average collapses into a markov process**.

$\epsilon-greedy$ is a common strategy used to balance exploration and exploitation, the mechanism is at time $t$, a random action $a_t$ which is sampled from the action set using a uniform distribution of $a_t \sim U\left(0,1\right)$ , traded off by the probability $\epsilon$: 
```math
a_t =\begin{cases}\text{random action}, & \text{with probability } \epsilon \\\arg\max_{a} Q(s_t, a), & \text{with probability } 1 - \epsilon\end{cases}
```
>[!info]
> If $a_t$ $\lt$  $\epsilon$ do  a random action, so called  exploration
> if $a_t$ $\geq$  $\epsilon$ do a greedy search, which is called  exploitation
> While $\epsilon$ is larger, policy will going to  explore more,  while a smaller $\epsilon$ will lead to more exploitation. $\epsilon$ will starts with high value in normal cases, then  decay over time to encourage more exploitation linearly or exponentially.

