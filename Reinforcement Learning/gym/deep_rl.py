#!/Users/johann/anaconda3/bin/python

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def start_game(model, env_gym='CartPole-v0', n_episodes=20):

    env = gym.make('env_gym')

    X_train = []
    y_train = []

    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(100):
            #env.render()
            #print(observation)
            if not model:
                action = env.action_space.sample()
            else:
                action = model.predict(state)
            observation, reward, done, info = env.step(action)
            X_train.append(state, observation, reward)
            #training_data_X.append(reward)
            y_train.append(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

# deep learning implementation
def train_model(X_train, y_train, input_dim=4, activation='relu', lr=0.0001, batch_size=32, epochs=5):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

#for _ in range(50):
#    start_game(model, steps=10)
#    train_model(training_data_X, training_data_X)
#    model.save("Reinforcement Learning/gym/model/cart_pole.h5")
