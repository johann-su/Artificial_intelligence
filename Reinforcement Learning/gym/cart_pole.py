#!/Users/johann/anaconda3/bin/python

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(200):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()