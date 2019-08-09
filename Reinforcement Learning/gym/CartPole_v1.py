#!/Users/johann/anaconda3/bin/python

# To-do:
# -Experience Replay
# -DQN Algorithm Q(state, action) = r + y * max(Q(state', action'))

# https://github.com/gsurma/cartpole
# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
# https://www.youtube.com/watch?v=t3fbETsIBCY
# https://www.youtube.com/watch?v=qfovbG84EBg&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=6

# importing nessecary libaries
import random
import gym
import numpy as np
from collections import deque
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

# defining variables
ENV_NAME = "CartPole-v1" # enviornment

GAMMA = 0.95 # einfluss
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000 # deque maxlen
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

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

        # creating the model
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

        # tensorboard for analytics
        self.tensorboard = ModifiedTensorBoard(log_dir="Reinforcement Learning/gym/logsA")


    # adding new values to the memory
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # predicting or choosing a random action
    def act(self, state):
        # random action
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

# the actual game
def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    # setting episodes to zero
    run = 0
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
                break
            dqn_solver.experience_replay()

    if run % 10 == 0:
        self.model.save(f'Reinforcement Learning/gym/models/CartPole-v1_{step}.h5')


if __name__ == "__main__":
    cartpole()
