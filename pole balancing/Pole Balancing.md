<img width="632" alt="image" src="https://github.com/user-attachments/assets/ea48f573-e8ca-4a55-9a64-b07dee36dac1">

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
