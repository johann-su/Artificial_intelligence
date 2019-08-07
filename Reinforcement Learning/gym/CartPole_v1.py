#!/Users/johann/anaconda3/bin/python

# To-do:
# -Experience Replay
# -DQN Algorithm Q(state, action) = r + y * max(Q(state', action'))

# importing the packages
import gym
import numpy as np
from collections import deque
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# creating the environment
env = gym.make('CartPole-v1')

#defining global variables
lr=0.0001
decay=0.001
batch_size=10
Gamma = 0.1

# creating a deep learning model with keras
model = Sequential()

model.add(Dense(64, input_shape=env.OBSERVATION_SPACE_VALUES, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))

model.compile(Adam(lr=lr, decay=decay), loss='mse')
model.summary()

memory = deque()

# running the game
while True:
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # observation = ndarray float64
        # reward = float
        # done = bool
        # action = int
        # info = empty
        # ------------------------------------------------
        #
        # Observation:
        # Type: Box(4)
        # Num	Observation            Min            Max
        # 0	Cart Position             -4.8            4.8
        # 1	Cart Velocity             -Inf            Inf
        # 2	Pole Angle                 -24°           24°
        # 3	Pole Velocity At Tip      -Inf            Inf
        #
        # Action:
        # Type: Discrete(2)
        # Num	Action
        # 0	Push cart to the left
        # 1	Push cart to the right

        observation = np.asarray(observation)
        reward = np.asarray(reward)
        action = np.asarray(action)

        memory.appendleft((observation, reward, action))

        if len(memory) > batch_size:
            rand_sample = random.sample(memory, batch_size)

            model.fit(np.expand_dims(observation, axis=0), np.expand_dims(action, axis=0), verbose=0)
            action = (reward + Gamma + np.max(model.predict(observation, verbose=0)))

        if done:
            break
env.close()
