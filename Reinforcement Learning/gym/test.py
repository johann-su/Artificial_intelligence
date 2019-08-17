#!/Users/johann/anaconda3/bin/pytho

import gym
import random

env = gym.make('CartPole-v1')

while True:
    env.reset()
    while True:
        env.render()
        if input('right'):
            action = 0
        elif inpu('left'):
            action = 1
        else:
            action = random.randint(1, 2)
        env.step(action)
        if done:
            break
