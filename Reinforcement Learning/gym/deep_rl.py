#!/Users/johann/anaconda3/bin/python

import gym
import numpy as np
from collections import deque
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

memory = deque()
epsilon = 1.0
batch_size = 64

def start_game():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            #env.render()
            print(observation)

            action = env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))

            observation, reward, done, info = env.step(action)
            memory.append(observation)
            memory.append(action)
            memory.append(reward)
            memory.append(done)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

def build_model():
    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='mse', metrics=['acc'])

    return model

def run():
    start_game()
    build_model()

    x_batch, y_batch = [], []
    minibatch = random.sample(
        memory, min(len(memory), batch_size))
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

run()
