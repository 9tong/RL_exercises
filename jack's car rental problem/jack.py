import numpy as np
import itertools
from multiprocessing import Pool

# Parameters
MAX_CARS = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9

# Poisson distribution parameters for requests and returns
REQUEST_RATE_A = 3
REQUEST_RATE_B = 4
RETURN_RATE_A = 3
RETURN_RATE_B = 2


# Precompute Poisson probabilities
def poisson_prob(lmbda, n):
    return (lmbda ** n) * np.exp(-lmbda) / np.math.factorial(n)


def poisson_distribution(lmbda, max_n=MAX_CARS):
    return [poisson_prob(lmbda, n) for n in range(max_n + 1)]


# Compute Poisson probabilities for requests and returns at both locations
request_prob_a = poisson_distribution(REQUEST_RATE_A)
request_prob_b = poisson_distribution(REQUEST_RATE_B)
return_prob_a = poisson_distribution(RETURN_RATE_A)
return_prob_b = poisson_distribution(RETURN_RATE_B)

# All possible states
states = [(a, b) for a in range(MAX_CARS + 1) for b in range(MAX_CARS + 1)]

# Initialize value function and policy
value_function = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)


def expected_return(state, action, value_function):
    """Compute the expected return for a given state and action."""
    cars_a, cars_b = state
    cars_a = min(cars_a - action, MAX_CARS)
    cars_b = min(cars_b + action, MAX_CARS)

    reward = -MOVE_COST * abs(action)

    expected_value = reward
    for rental_a in range(MAX_CARS + 1):
        for rental_b in range(MAX_CARS + 1):
            prob_rental_a = request_prob_a[rental_a]
            prob_rental_b = request_prob_b[rental_b]

            actual_rental_a = min(cars_a, rental_a)
            actual_rental_b = min(cars_b, rental_b)
            reward_rental = (actual_rental_a + actual_rental_b) * RENTAL_REWARD

            remaining_cars_a = cars_a - actual_rental_a
            remaining_cars_b = cars_b - actual_rental_b

            for return_a in range(MAX_CARS + 1):
                for return_b in range(MAX_CARS + 1):
                    prob_return_a = return_prob_a[return_a]
                    prob_return_b = return_prob_b[return_b]

                    next_cars_a = min(remaining_cars_a + return_a, MAX_CARS)
                    next_cars_b = min(remaining_cars_b + return_b, MAX_CARS)

                    prob = prob_rental_a * prob_rental_b * prob_return_a * prob_return_b
                    expected_value += prob * (reward_rental + DISCOUNT * value_function[next_cars_a, next_cars_b])

    return expected_value


def evaluate_state(state):
    """Evaluate the best action for a given state."""
    actions = range(-MAX_MOVE, MAX_MOVE + 1)
    cars_a, cars_b = state
    actions = [action for action in actions if 0 <= cars_a - action <= MAX_CARS and 0 <= cars_b + action <= MAX_CARS]
    action_returns = [(action, expected_return(state, action, value_function)) for action in actions]
    best_action, best_return = max(action_returns, key=lambda x: x[1])
    return state, best_action, best_return


def policy_iteration():
    """Main policy iteration loop with multiprocessing."""
    global value_function, policy

    iteration = 0
    while True:
        # Policy Evaluation
        print(f"Iteration {iteration}: Policy Evaluation")
        while True:
            delta = 0
            with Pool() as pool:
                results = pool.map(evaluate_state, states)

            new_value_function = np.zeros_like(value_function)
            for (state, _, best_return) in results:
                new_value_function[state] = best_return
                delta = max(delta, abs(new_value_function[state] - value_function[state]))

            value_function = new_value_function
            print(f"Delta: {delta}")
            if delta < 1e-4:
                break

        # Policy Improvement
        print(f"Iteration {iteration}: Policy Improvement")
        policy_stable = True
        with Pool() as pool:
            results = pool.map(evaluate_state, states)

        for (state, best_action, _) in results:
            if policy[state] != best_action:
                policy_stable = False
            policy[state] = best_action

        if policy_stable:
            print("Policy stable. Iteration complete.")
            break

        iteration += 1


if __name__ == "__main__":
    policy_iteration()
    print("Optimal Policy:")
    print(policy)