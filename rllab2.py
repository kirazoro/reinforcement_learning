# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 06:55:51 2025

@author: chitrangi bhatnagar
"""

import gymnasium as gym
env=gym.make('CartPole-v1',render_mode='human')
env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
env.reset()
n_episodes=50
n_timesteps=50

for e in range(n_episodes):
    
    ret=0
    for t in range(n_timesteps):
        env.render()
        rnd_action=env.action_space.sample()
        (next_state,reward,done,trans_prob,info)=env.step(rnd_action)
        ret=ret+reward

        if done:
            env.reset()
            break
        if e%10==0:
            print("Episode:{},Return:{}".format(e+1,ret))
env.close()