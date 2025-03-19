import gym
import numpy as np

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Render enabled
print("Environment Created")

# Display action space and state space
print(f"Action Space: {env.action_space}")  # Discrete(2) → 0 (left), 1 (right)
print(f"State Space: {env.observation_space}")  # Box(4,) → Continuous states

# Define a simple deterministic policy (always move right)
def deterministic_policy(state):
    return 1  # Always take action "1" (move right)

# Generate 30 episodes and print return for each
num_episodes = 30
episode_returns = []

for episode in range(num_episodes):
    state = env.reset()[0]  # Reset environment
    done = False
    total_reward = 0
    
    while not done:
        action = deterministic_policy(state)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state  # Move to the next state

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Return = {total_reward}")

env.close()

# Show state transition probabilities (not explicitly available in CartPole)
print("\nState transition probabilities are not explicitly available for CartPole since it's a continuous environment.")
