#!/Users/johann/anaconda3/bin/python

import gym
from collections import deque
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

memory = deque(maxlen=(10000))
env = gym.make('CartPole-v1')

model = Sequential()

model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='mse', metrics=['acc'])
model.summary()

X = deque(maxlen=(10000))
y = deque(maxlen=(10000))
z = deque(maxlen=(10000))

for i_episodes in range(20):
    state = env.reset()
    for t in range(100):
        env.render()
        # print("t: " + str(t))
        # print("memory: " + str(memory))
        if memory:
            # print("training the model")
            for i in range(len(memory)):
                X.append(memory[i][1])
                y.append(memory[i][2])
                z.append(memory[i][3])

            X_train = np.append(X, z)
            y_train = np.asarray(y)

            # print("X_train.shape: " + str(np.asarray(X_train).shape))
            # print("X_train: " + str(np.asarray(X_train)))
            # print("y_train.shape: " + str(np.asarray(y_train).reshape(-1, 1).shape))
            # print("y_train: " + str(np.asarray(y_train)))

            model.fit(X_train.reshape(-1, 5), y_train.reshape(-1, 1), batch_size=32, verbose=0)

            # print(np.asarray(observation).reshape(1, 4).shape)
            # print(np.asarray(observation).reshape(1, 4))
            # print(model.predict(np.asarray(observation).reshape(1, 4)))
            # print("float: " + str(float(model.predict(np.asarray(observation).reshape(1, 4)))))

            observation = np.append(observation, np.random.choice((0, 1), p=[0.1, 0.9]))

            if float(model.predict(observation.reshape(1, 5))) <= 0.5:
                action = 1
            else:
                action = 0

            # print("action: " + str(action))
        else:
            # print("random action")
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        memory.append((state, observation, action, reward, done))
        if done:
            print("done after episode {}".format(t+1))
            break
env.close()
