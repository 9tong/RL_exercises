import sys
import time

import gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt


# Load the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load the trained model
model = DQN.load("dqn_cartpole_model")

# Initialize plot
fig, ax = plt.subplots()
img = None
action = 1 # Default action (push cart to the left)

# Key event handler
def on_key(event):
    global action
    if event.key == "left":
        action = 0  # Push cart to the left
        print("push left")
    elif event.key == "right":
        action = 1  # Push cart to the right
        print("push right")


# Connect the key press event to the Matplotlib figure
fig.canvas.mpl_connect("key_press_event", on_key)

# Test the trained agent and capture frames
obs, _ = env.reset()
done = False

while True :
    #action, _ = model.predict(obs, deterministic=False)  # Predict action
    time.sleep(0.03)

    obs, reward, done, info, _ = env.step(action)  # Take action

    print(obs, reward, done, info, action)

    # Render the frame as a numpy array
    frame = env.render()  # No mode argument here; uses default rendering

    # If needed, display the frame using matplotlib
    if frame is not None:
        # Initialize or update the plot
        if img is None:
            img = ax.imshow(frame)  # Create the plot for the first time
            plt.axis('off')  # Turn off the axes
        else:
            img.set_data(frame)  # Update the image data
        plt.pause(0.03)  # Pause to make the image visible

env.close()
plt.show()  # Ensure the final frame remains displayed