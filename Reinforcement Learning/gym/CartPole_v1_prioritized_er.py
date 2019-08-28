# To-do:
# -prioritised experience_replay
# -Manual Mode

# Credit where it's due:
# https://github.com/gsurma/cartpole
# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
# https://www.youtube.com/watch?v=t3fbETsIBCY
# https://www.youtube.com/watch?v=qfovbG84EBg&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=6
# https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/)

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
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from pynput.keyboard import Key, Listener

# choose the mode you want to run the programm in:
#
# mode: hard_policy
# a hard coded policy
#
# mode: Train
# a model is trained
#
# mode: Test
# a previusly trained model is used to play the game; No training
#
# mode: Manual
# play the game yourself

MODE = 'Train'

# defining variables
ENV_NAME = "CartPole-v1" # enviornment

GAMMA = 0.95 # discount factor
LEARNING_RATE = 0.001
DECAY = 0.0001

MEMORY_SIZE = 1000000 # deque maxlen
BATCH_SIZE = 40

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995 # higher is slower

UPDATE_MODEL = 10

# try:
#     os.remove('Reinforcement Learning/gym/models/')
#     os.remove('Reinforcement Learning/gym/logs/')
# except:
#     pass

# Own Tensorboard class
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

# a class that contains all nessecary functions:
# -update_memory
# -act
# -experience_replay
# -save_model
class DQNSolver:
    def __init__(self, observation_space, action_space):
        #defining the exploration rate
        self.exploration_rate = EXPLORATION_MAX

        # definng the action space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.all_steps = []
        self.all_rewards = []
        self.highest_score = 0

        print('Mode = ' + MODE)

        # creating the model
        self.model = Sequential()

        self.model.add(Dense(48, input_shape=(observation_space,), activation="elu"))
        self.model.add(Dense(48, activation="elu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE, decay=DECAY, amsgrad=True))


        self.target_model = Sequential()

        self.target_model.add(Dense(48, input_shape=(observation_space,), activation="elu"))
        self.target_model.add(Dense(48, activation="elu"))
        self.target_model.add(Dropout(0.2))

        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dense(24, activation="elu"))
        self.model.add(Dropout(0.2))

        self.target_model.add(Dense(self.action_space, activation="linear"))
        self.target_model.compile(loss="mse", optimizer=Adam())

        if os.listdir("Reinforcement Learning/gym/models/"):
            self.model.load_weights('Reinforcement Learning/gym/models/CartPole-v1.h5')
            self.model.load_weights('Reinforcement Learning/gym/models/CartPole_target-v1.h5')
            self.exploration_rate = EXPLORATION_MIN

        # tensorboard for analytics
        self.tensorboard = ModifiedTensorBoard(log_dir="Reinforcement Learning/gym/logs/")


    # adding new values to the memory
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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
                q_update = (reward + GAMMA * np.amax(self.target_model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0, callbacks=[self.tensorboard])
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def update_weights(self, run):
        if run % UPDATE_MODEL == 0:
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)

    def save_model(self):
        self.model.save_weights('Reinforcement Learning/gym/models/CartPole-v1.h5')
        self.target_model.save_weights('Reinforcement Learning/gym/models/CartPole_target-v1.h5')

# Train an Agent on the CartPole environment
if MODE == 'Train':
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
                # env.render()
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, observation_space])
                dqn_solver.update_memory(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    dqn_solver.update_weights(run)
                    dqn_solver.save_model()
                    dqn_solver.tensorboard.update_stats(score=step, reward=np.mean(dqn_solver.all_rewards))
                    break
                dqn_solver.experience_replay()
        env.close()

# Test a trained model on the CartPole environment
elif MODE == 'Test':
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
                state = state_next
                if terminal:
                    print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    # dqn_solver.save_model(run, step)
                    dqn_solver.tensorboard.update_stats(score=step, reward=np.mean(dqn_solver.all_rewards))
                    break
        env.close()

# proove your skills by playing the CartPole environment by yourself
elif MODE == 'Manual':
    def cartpole():
        keyboard = Listener()
        env = gym.make(ENV_NAME)
        while True:
            env.reset()
            step = 0
            while True:
                env.render()
                if left:
                    action = 1
                elif right:
                    action = 0
                else:
                    env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print(f'finished after {step} steps')
                    break
        env.close()
        keyboard.stop()

# a hard coded policy to solve the CartPole environment
elif MODE == 'hard_policy':
    def cartpole():
        env = gym.make(ENV_NAME)
        run = 0
        while True:
            run += 1
            observation = env.reset()
            step = 0
            while True:
                step += 1
                env.render()
                position, velocity, angle, angular_velocity = observation
                if angle < 0:
                    action = 0
                else:
                    action = 1

                observation, reward, done, info = env.step(action)
                if done:
                    print(f'finished after {step} steps')
                    break
        env.close()

else:
    print('Error: Choose a Mode!')
    sys.exit()

if __name__ == "__main__":
    cartpole()
