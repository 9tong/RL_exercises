import time
import gym
from matplotlib.backend_bases import Event, KeyEvent
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# Load the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load the trained model
model = DQN.load("dqn_cartpole_model")

# Initialize plot
fig, ax = plt.subplots()
img = None
action = 1  # Default action (push cart to the right)
action_taken = False  # Track if an action has been taken

# Listen to the key event
def on_key(event: KeyEvent):
    global action, action_taken
    if event.key == "left":
        action = 0  # Push cart to the left
        action_taken = True
        print("push left")
    elif event.key == "right":
        action = 1  # Push cart to the right
        action_taken = True
        print("push right")

# Connect the key press event to the Matplotlib figure
fig.canvas.mpl_connect("key_press_event", on_key)

# Test the trained agent and capture frames
obs, _ = env.reset()
done = False

# Render the initial frame before starting any action
initial_frame = env.render()
if initial_frame is not None:
    img = ax.imshow(initial_frame)  # Create the plot for the first time with the initial frame
    plt.axis('off')  # Turn off the axes
    plt.pause(0.05)  # Pause to make the image visible

# Wait for the user to take an action before proceeding
while not action_taken:
    plt.pause(0.1)  # Keep the initial frame displayed until an action is taken

while True:
    time.sleep(0.05)  # Slow down the loop for better user control

    # Take the action determined by the user
    obs, reward, done, info, _ = env.step(action)
    print(obs, reward, done, info, action)

    # Render the frame as a numpy array
    frame = env.render()

    # If needed, display the frame using matplotlib
    if frame is not None:
        # Initialize or update the plot
        if img is None:
            img = ax.imshow(frame)  # Create the plot for the first time
            plt.axis('off')  # Turn off the axes
        else:
            img.set_data(frame)  # Update the image data
        plt.pause(0.05)  # Pause to make the image visible

    # Handle the 'done' state
    if done:
        obs, _ = env.reset()
        print("Environment reset")

env.close()
#plt.show()  # Ensure the final frame remains displayed
