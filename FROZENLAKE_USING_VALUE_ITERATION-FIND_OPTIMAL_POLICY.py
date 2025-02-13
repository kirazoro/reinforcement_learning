# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:54:57 2025

@author: Chitrangi Bhatnagar
"""

# SOLVING FROZEN LAKE USING VALUE ITERATION - FIND AN OPTIMAL POLICY
import numpy as np
import gymnasium as gym

def value_iteration(env):
    num_itns = 1000
    threshold = 1e-20
    gamma = 1.0
    value_table = np.zeros(env.observation_space.n)

    for _ in range(num_itns):
        updated_val_tab = np.copy(value_table)

        for s in range(env.observation_space.n):
            Q_values = [sum([prob * (r + gamma * updated_val_tab[s_]) 
                            for prob, s_, r, _ in env.P[s][a]])
                        for a in range(env.action_space.n)]
            value_table[s] = max(Q_values)

        if np.sum(np.fabs(updated_val_tab - value_table)) <= threshold:
            break

    return value_table

def extract_policy(env, value_table):
    gamma = 1.0
    policy = np.zeros(env.observation_space.n)

    for s in range(env.observation_space.n):
        Q_values = [sum([prob * (r + gamma * value_table[s_]) 
                         for prob, s_, r, _ in env.P[s][a]])
                    for a in range(env.action_space.n)]
        policy[s] = np.argmax(np.array(Q_values))
    
    return policy

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", render_mode="human")

env = env.unwrapped  # Unwrap to access transition probabilities
env.reset()
env.render()
# Run value iteration
optimal_value_function = value_iteration(env)

# Extract optimal policy
optimal_policy = extract_policy(env, optimal_value_function)

print("Optimal Value Function:")
print(optimal_value_function)

print("Optimal Policy:")
print(optimal_policy)