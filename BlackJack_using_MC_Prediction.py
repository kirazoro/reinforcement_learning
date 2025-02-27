import pandas as pd
from collections import defaultdict
import gymnasium as gym

# Initialize environment
env = gym.make('Blackjack-v1', render_mode='human')
state, _ = env.reset()
env.render()

# Policy function
def policy(state):
    return 0 if state[0] > 19 else 1

# Function to generate an episode
num_timesteps = 100
def generate_episode(policy):
    episode = []
    state, _ = env.reset()
    for _ in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, _, _ = env.step(action)  # Fixed unpacking
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode

# Monte Carlo evaluation
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 10

for _ in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = sum(rewards[t:])
        total_return[state] += R
        N[state] += 1

# Convert to DataFrame
df_total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
df_N = pd.DataFrame(N.items(), columns=['state', 'N'])
df = pd.merge(df_total_return, df_N, on='state')

# Ensure 'state' column is stored as tuples for correct filtering
df['state'] = df['state'].apply(tuple)

# Calculate value function
df['value'] = df['total_return'] / df['N']

# Display results
print(df.head(10))
print(df.shape)

# Safely query values
state_to_check = (21, 9, False)

# Use string conversion for tuple comparison
if (df['state'].astype(str) == str(state_to_check)).any():
    print(df[df['state'] == state_to_check]['value'].values)

env.close()
