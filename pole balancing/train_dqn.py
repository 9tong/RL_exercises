# Import necessary libraries
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import shimmy


# Step 1: Setup the environment
env = gym.make("CartPole-v1")  # Create the CartPole environment

# Step 2: Initialize the DQN agent
model = DQN("MlpPolicy", env, verbose=1)  # Use Multi-Layer Perceptron policy

# Step 3: Train the agent
print("Training the model...")
model.learn(total_timesteps=10000)  # Train for 10,000 timesteps

# Step 4: Evaluate the trained agent
print("Evaluating the trained model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Dev: {std_reward}")

# Step 5: Visualize the trained agent's performance
print("Testing the trained model...")
obs, _ = env.reset()  # Reset the environment
for _ in range(1000):  # Test for 1000 steps
    env.render()  # Render the environment (visualization)
    action, _states = model.predict(obs, deterministic=True)  # Get action from the model
    obs, reward, done, info, _ = env.step(action)  # Take the action
    if done:  # If the episode is done, reset the environment
        obs, _ = env.reset()

env.close()  # Close the environment

# Step 6: Save the trained model for future use
model.save("dqn_cartpole_model")
print("Model saved as 'dqn_cartpole_model'.")

# Optional: Reload the model
# model = DQN.load("dqn_cartpole_model")