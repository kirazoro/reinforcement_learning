# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:11:42 2025

@author: chitrangi bhatnagar
"""

import gymnasium as gym
import time
env=gym.make('FrozenLake-v1',render_mode='human')
env.reset()
env.render()
env=env.unwrapped
print(env.observation_space)
print(env.action_space)
print(env.P[0][2])
env.reset()

(next_state,reward,done,trans_prob,info)=env.step(2)
env.render()
rnd_action=env.action_space.sample()
env.step(rnd_action)
print("Action taken:",rnd_action)

#generate 1 episode
for e in range(10):
    num_timesteps=20
    ret=0
    env.reset()
    for t in range(num_timesteps):
        rnd_action=env.action_space.sample()
        (next_state,reward,done,trans_prob,info)=env.step(rnd_action)
        ret=ret+reward
        env.render()
        time.sleep(2)
        if done:
            print("The return of this episode is:",ret)
            break
env.close()
