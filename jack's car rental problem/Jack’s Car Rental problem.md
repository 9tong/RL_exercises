# The Problem
**Jack’s Car Rental problem** is a classic example used in Reinforcement Learning, particularly in the context of dynamic programming and value iteration. It appears in Sutton and Barto’s “Reinforcement Learning: An Introduction” (Chapter 4).
Jack’s Car Rental problem is a well-known benchmark that shows how RL methods can handle decision-making over time in uncertain, stochastic environments.

# Scenario
Jack manages two car rental locations. Each day, customers arrive at both locations wanting to rent cars. Overnight, Jack has the option to move cars between the two locations to better meet the next day’s demand. However, moving cars is costly. The problem is to determine how many cars, if any, Jack should transfer each night to maximize his long-term expected profit.

# Key Points
1. **State Representation:**
A state in this problem is typically defined by the number of cars at each of the two locations at the end of the day (after rentals and returns). For example, a state might be represented as (i, j) where i is the number of cars at the first location and j is the number at the second.
2. **Actions:**
An action is how many cars to move from one location to the other overnight. This could be zero (no move) or a positive or negative number, indicating direction and quantity of cars moved. There’s a cost associated with each car moved.
3. **Transitions (Dynamics):**
Each day, a certain number of customers arrive wanting to rent cars at each location. This demand is modeled by a probability distribution (often Poisson). Similarly, cars are returned to each location overnight according to another Poisson distribution. Thus, the next day’s starting state depends stochastically on the current state, the action taken (transfer of cars), and these random rental and return events.
4. **Rewards:**
The reward for a given day usually comes from the number of cars rented (each car rented yields a fixed profit) minus any transfer cost for moving cars. If demand exceeds the available cars at a location, only the available number can be rented out, limiting that day’s profit.
5. **Goal:**
The objective is to find a policy—a rule for how many cars to move each night—maximizing the expected long-term profit. This is usually solved using dynamic programming or other RL techniques like policy iteration or value iteration.

Jack’s Car Rental is a pedagogical example that:
• Illustrates how to formulate a complex decision-making problem under uncertainty as a Markov Decision Process (MDP).
• Demonstrates the use of Bellman equations, policy evaluation, and improvement.
• Serves as a concrete example where a solution involves balancing immediate profit (renting cars) with long-term gains (strategically positioning cars).


# Setups
X-axis: Number of cars at Location A.
Y-axis: Number of cars at Location B.
Z-axis: The action (number of cars moved) for each state.
Policy is : for each combination of cars at Locations A and B (a state $`(x, y)`$), the policy suggests how many cars should be moved overnight between the two locations.
Z value represents the number of cars to be moved overnight between the two locations for the given state. 
- Positive value means moving cars from Location B to Location A. 
- Negative value means moving cars from Location A to Location B.
- Zero value means no cars are moved between the two locations.
Rental demand follows Poisson distribution. Means of 3 for location A, means of 4 for location B.
Cars returned according to a Poisson distribution. Means of 3 for location A, means of 2 for location B.
Cost: moving cars costs 2$
income: 10$
discount factor: 0.9


# The math steps
The "Jack’s Car Rental" problem, as described in Sutton and Barto’s "Reinforcement Learning: An Introduction," is a classic example of a finite Markov decision process (MDP) that can be approached using dynamic programming techniques such as policy iteration or value iteration. The mathematical formulation involves modeling the probability distributions of customer demand, the state transitions, and the expected rewards, then iteratively applying the Bellman equations. Below is a detailed breakdown of the math and the steps involved.

## Problem Setup
- **State Representation:**  
  The state $s = (i, j)$ represents the number of cars at location 1 and location 2 at the end of the day (after returns, but before any overnight transfers). Typical constraints might be  $i, j \in {0, 1, 2, \ldots, n}$, for some maximum inventory $n$.

- **Actions:**  
  An action $a$ represents how many cars to move overnight from location 1 to location 2. If $a$ is positive, you move $a$  cars from location 1 to location 2; if $a$ is negative, you move $-a$ cars from location 2 to location 1. There is usually a limit on how many cars can be moved (e.g., $|a| \leq 5$). Each car moved incurs a fixed cost (e.g., \$2 per car).

- **Rewards:**  
  Each car rented out yields a certain reward (e.g., \$10 per rental). The expected number of rentals depends on the Poisson-distributed customer demands at each location.

- **Transition Probabilities:**  
  The number of requests at each location is Poisson-distributed:
  $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
  Here, $\lambda$ is the parameter of the Poisson distribution. For location 1, let the request rate be $\lambda_1$, and for location 2, $\lambda_2$. Similarly, the number of returned cars at each location follows its own Poisson distribution with given parameters (e.g., $\lambda_{1r}$ and $\lambda_{2r}$ for returns). 

## The Bellman Expectation and Optimality Equations
For a given policy $\pi$, the state-value function $V^\pi(s)$ must satisfy the Bellman expectation equation:
$`V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]`$
For the *optimal* policy $`\pi^*`$ and *optimal* value function $`V^*`$, we have the Bellman optimality equation:
$`V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]`$
## Computing the Expected Returns and Transitions
To solve the problem, you need to carefully compute the right-hand side of these equations. Consider a single state-action pair $(s, a)$:
1. **Adjust State for Action:**  
   After choosing action $a$ , suppose the initial state is $s=(i,j)$. The new intermediate state after the move is:
	1. $`i' = i - a \quad\text{(cars left at location 1 after move)}`$
	2. $`j' = j + a \quad\text{(cars left at location 2 after move)}`$
	3. We must ensure $0 \leq i' \leq n$ and $0 \leq j' \leq n$, suppose $|n| = 5$

2. **Calculate Immediate Costs/Rewards for the Action:**
   - **Movement cost:** If moving $|a|$ cars, immediate cost is $2|a|$ (assuming \$2 per moved car).
   - This will reduce the net reward obtained that day.

3. **Expected Rentals (Revenue) from the Next Morning:**
   The number of cars requested at location 1 is a Poisson random variable, say $`X_1 \sim \text{Poisson}(\lambda_1)`$. The number of cars requested at location 2 is $`X_2 \sim \text{Poisson}(\lambda_2)`$.
   The expected number of rentals at location 1 is:$`E[\text{rentals}_1] = \sum_{k=0}^{\infty} \min(i', k) \frac{\lambda_1^k e^{-\lambda_1}}{k!}`$
   Similarly, for location 2: $`E[\text{rentals}_2] = \sum_{k=0}^{\infty} \min(j', k) \frac{\lambda_2^k e^{-\lambda_2}}{k!}`$

   Each rental yields a reward (e.g., \$10 per rental), so:
   $E[\text{rental reward}] = 10 (E[\text{rentals}_1] + E[\text{rentals}_2])$

   This summation can be computed once and stored, or truncated at a certain number since the Poisson probabilities become negligible beyond a certain point.

4. **Expected Next State Distribution (After Returns):**
   After fulfilling rentals, returns occur. Let $Y_1 \sim \text{Poisson}(\lambda_{1r})$ and $Y_2 \sim \text{Poisson}(\lambda_{2r})$ be the random variables for returns at locations 1 and 2, respectively.
   The next day’s starting state $s' = (i'', j'')$ results from:
   - Decreasing $i'$ and $j'$ by the rentals that actually took place.
   - Adding returned cars according to $Y_1$ and $Y_2$.
   
   More precisely, the distribution of $s'$ is:$`P(i'', j'' | s,a) = \sum_{r_1=0}^{\infty}\sum_{r_2=0}^{\infty}\sum_{q_1=0}^{\infty}\sum_{q_2=0}^{\infty} P(X_1=q_1)P(X_2=q_2)P(Y_1=r_1)P(Y_2=r_2) \cdot \mathbb{I}[i'' = i' - \min(i',q_1) + r_1]\mathbb{I}[j'' = j' - \min(j',q_2) + r_2]`$
   While this looks complicated, many terms vanish because the indicator functions ensure that $i''$ and $j''$ match the computed outcome. In practice, computations are done by enumerating feasible ranges and using efficient precomputed Poisson probabilities. Often, the state space and Poisson probabilities are truncated to avoid infinite sums.
>[!tip]
>In Detail:
> Let $`Q1 \sim Poisson(\lambda_1), Q2 \sim Poisson(\lambda_2)`$, so $Q1$ and $Q2$ are represented the request for customer rentals at location 1 and location 2.
> Let  $`R1 \sim Poisson(\lambda_{1r}), R2 \sim Poisson(\lambda_{2r})`$, so $R1$ and $R2$ represent the returns of cars at location 1 and location 2.
> 
> Assuming, $Q1,Q2,R1,R2$ are independent variables.
> 
> If next state is : $i'' = i' - \min(i', Q1) + R1$, and  $j'' = j' - \min(j', Q2) + R2$, note that, $\min(j', Q2)$ means the max cars available are the $min$ of $j'$(car right here) and the $Q$(customer demand)
> 
> Considering next state is $s' = (i'',j'')$,  we should find the $P(s' | s, a)$ and the initial $s$ is given , $a$ can be sampled, so,
> 	$`P(i'', j'' | s, a) =  \sum_{Q_1=0}^{\infty} \sum_{Q_2=0}^{\infty} \sum_{R_1=0}^{\infty} \sum_{R_2=0}^{\infty} P(Q_1) P(Q_2) P(R_1) P(R_2)`$
> 
> There exists some paticular state that will never happen,  such as states where the number of cars at either location exceeds the maximum capacity or becomes negative, so add indicator  functions which  enforce next states of $i''$ and $j''$ must follow:  $`[i' - \min(i', Q1) + R1] \in [0, 20]`$ and  $`[j' - \min(j', Q2) + R2] \in [0, 20]`$ and $`\mathbb{E}[i] + \mathbb{E}[j]=20`$
> so, the equation at last looks like a little bit complicated:
> 	$`P(i'', j'' | s,a) = \sum_{r_1=0}^{\infty}\sum_{r_2=0}^{\infty}\sum_{q_1=0}^{\infty}\sum_{q_2=0}^{\infty} P(X_1=q_1)P(X_2=q_2)P(Y_1=r_1)P(Y_2=r_2) \cdot \mathbb{I}[i'' = i' - \min(i',q_1) + r_1]\mathbb{I}[j'' = j' - \min(j',q_2) + r_2]`$
> 
  
5. **Putting it All Together:**
   For each state $s$  and action $a$:
   - Compute the immediate (expected) reward:  
     $R(s,a) = E[\text{rental reward}] - 2|a|$
   - Compute the next-state value as:
     $Q(s,a) = \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V(s')]$

   During **policy evaluation**, given a fixed policy $\pi$, you would do:
   $`V^{(k+1)}(s) = \sum_{a} \pi(a|s) \left\{\sum_{s'} P(s'|s,a) [R(s,a) + \gamma V^{(k)}(s')]\right\}`$

   During **policy improvement**, given the newly computed values $V^{(k+1)}$:
   $`\pi_{\text{new}}(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V^{(k+1)}(s')]`$

   Iterating these two steps (policy evaluation and policy improvement) leads to convergence to the optimal policy $`\pi^*`$ and optimal value function $`V^*`$.

## Computational Techniques
- **Truncation of Poisson Distributions:**  
  Since Poisson probabilities become very small as you move far from $\lambda$, you truncate sums at a reasonable number (e.g., up to around $\lambda + 10$ or so) to keep computations manageable.

- **Caching Poisson Probabilities and Expected Rentals:**  
  Precompute factorial terms, Poisson probabilities, and partial sums for expected rentals to speed up each iteration.

- **Value Iteration Instead of Policy Iteration:**  
  Instead of policy iteration (which alternates evaluation and improvement), one can use value iteration directly to solve the Bellman optimality equation until convergence.

## Summary of the Math Steps
1. Define state, action, and reward structure.
2. Use Poisson distributions to compute expected rentals (and thus expected immediate rewards).
3. Incorporate action costs (moving cars).
4. Compute transition probabilities for the next states based on rental requests and returns.
5. Apply the Bellman equations:
   - For policy evaluation: solve for $V^\pi$.
   - For policy improvement: find $\arg\max_a$ of the action-value functions.
6. Iterate until policy stability or convergence.

Through these steps, formulating the MDP, computing expected returns with Poisson distributions, and applying dynamic programming techniques, you obtain the optimal policy for the car rental problem.



