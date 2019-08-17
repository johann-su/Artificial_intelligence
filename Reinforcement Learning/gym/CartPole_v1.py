#!/Users/johann/anaconda3/bin/python

# To-do:
# -TensorBoard implementation
# -find good hyperparameters

# https://github.com/gsurma/cartpole
# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
# https://www.youtube.com/watch?v=t3fbETsIBCY
# https://www.youtube.com/watch?v=qfovbG84EBg&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=6

# importing nessecary libaries
import os
import sys
import time
import random
import gym
import pydot
import numpy as np
from collections import deque
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Pool

# params = {'GAMMA':[0.5, 0.6, 0.7, 0.8, 0.9],
#         'LEARNING_RATE':[0.1, 0.01, 0.001, 0.0001],
#         'BATCH_SIZE':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
#         'EXPLORATION_MIN':[0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005],
#         'EXPLORATION_DECAY':[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]}
#
# rscv = RandomizedSearchCV(estimator=step, param_distributions=params, n_iter=100, scoring=x, n_jobs=-1, verbose=0)

# mode: Train
# a model is trained
#
# mode: Test
# a previusly trained model is used to play the game; No training requiert
#
# mode: Manual
# the user can try to play the game himself
MODE = 'Train'

# defining variables
ENV_NAME = "CartPole-v1" # enviornment

GAMMA = 0.95 # einfluss
LEARNING_RATE = 0.001
# DECAY = 0.000001

MEMORY_SIZE = 1000000 # deque maxlen
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995 # higher is slower

# try:
#     os.remove('Reinforcement Learning/gym/models/')
#     os.remove('Reinforcement Learning/gym/logs/')
# except:
#     pass

# Own Tensorboard class (https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/)
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNSolver:

    def __init__(self, observation_space, action_space):
        #defining the exploration rate
        self.exploration_rate = EXPLORATION_MAX

        # definng the action space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.all_steps = []
        self.all_rewards = []

        # creating the model
        self.model = Sequential()

        self.model.add(Dense(48, input_shape=(observation_space,), activation="elu"))
        self.model.add(Dense(48, activation="elu"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE, amsgrad=True))

        # plot_model(self.model, to_file='Reinforcement Learning/gym/model.png', show_shapes=True)

        # tensorboard for analytics
        self.tensorboard = ModifiedTensorBoard(log_dir="Reinforcement Learning/gym/logs/")


    # adding new values to the memory
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.appendleft((state, action, reward, next_state, done))

    # predicting or choosing a random action
    def act(self, state):
        # random action
        # np.random.rand() ceates a random float
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # predicting an action
        q_values = self.model.predict(state)
        # returning the best q-value
        return np.argmax(q_values[0])

    # choosing a random sample from memory to train the model after a game
    def experience_replay(self):
        # memory too small
        if len(self.memory) < BATCH_SIZE:
            return
        # taking a random sample (how many depends on BATCH_SIZE)
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0, callbacks=[self.tensorboard])
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self, run, step):
        self.all_steps.append(step)
        if run % 10 == 0:
            self.model.save(f'Reinforcement Learning/gym/models/CartPole-v1_{int(np.mean(self.all_steps))}.h5')
        if len(self.all_steps) == 10:
            self.all_steps = []

if MODE == 'Train':
    # the actual game
    def cartpole():
        env = gym.make(ENV_NAME)
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        dqn_solver = DQNSolver(observation_space, action_space)
        # setting episodes to zero
        run = 0
        dqn_solver.tensorboard.step = run
        while True:
            run += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            # setting step to zero
            step = 0
            while True:
                step += 1
                env.render()
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, observation_space])
                dqn_solver.update_memory(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    dqn_solver.save_model(run, step)
                    dqn_solver.tensorboard.update_stats(score=step, reward=np.mean(dqn_solver.all_rewards))
                    break
                dqn_solver.experience_replay()

elif MODE == 'Test':
    # the actual game
    pass

elif MODE == 'Manual':
    pass
    # manual code here

else:
    print('Error: Choose a Mode!')
    sys.exit()

if __name__ == "__main__":
    cartpole()
