#!/Users/johann/anaconda3/bin/python

import gym
from collections import deque
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

memory = deque(maxlen=10000)
env = gym.make('CartPole-v1')

model = Sequential()

model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='mse', metrics=['acc'])

X_train = deque()
y_train = deque()
for i_episodes in range(20):
    state = env.reset()
    for t in range(100):
        for i in range(len(memory)):
            X_train.append(memory[i][1])
            X_train.append(memory[i][3])
            y_train.append(memory[i][2])
        model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=32)
        # env.render()
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)
        memory.append((state, observation, action, reward, done))
        if done:
            print(t)
            break
env.close()
