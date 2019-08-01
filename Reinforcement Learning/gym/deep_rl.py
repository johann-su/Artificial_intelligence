#!/Users/johann/anaconda3/bin/python

#To-Do:
# -Memory for the prediction (2+ frames for the Prediction)
# -RNN maybe?

from joblib import Parallel
import gym
from collections import deque
import numpy as np

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# creating a memory where previous timesteps are seved
memory = deque(maxlen=(10000))
env = gym.make('CartPole-v1')

# creating a model for evaluating the best strategy to win the game
model = Sequential()

# model.add(Dense(32, input_shape=(5, 2), activation='relu'))
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='mse', metrics=['acc'])
model.summary()

# grid search cv
params = {'lr':[0.1, 0.01, 0.001, 0.0001], 'decay':[0.1, 0.01, 0.001, 0.0001]}
clf = GridSearchCV(Sequential, params, scoring='neg_mean_squared_error', n_jobs=-1)

# splitting the data into three diffrent dics
X = deque(maxlen=(10000))
y = deque(maxlen=(10000))
z = deque(maxlen=(10000))

# starting a game with gym
for i_episodes in range(200):
    state = env.reset()
    for t in range(100):
        # env.render()
        # checking if memory contains more than two entries
        if len(memory) >2:
            # putting the data from memory into the apropreate dict
            for i in range(len(memory)):
                # observation
                X.appendleft(memory[i][1])
                # reward
                y.appendleft(memory[i][2])
                # action
                z.appendleft(memory[i][3])

            # creating a np traing-data-set
            X_train = np.append(X, z)
            y_train = np.asarray(y)

            # fitting the model to the traing-data
            model.fit(X_train.reshape(-1, 5), y_train.reshape(-1, 1), batch_size=32, verbose=0)

            # putting the observation and the action into one dictionary - action being one 90%
            observation = np.append(observation, np.random.choice((0, 1), p=[0.1, 0.9]))

            # the model predicts am action
            if float(model.predict(observation.reshape(1, 5))) <= 0.5:
                action = 1
            else:
                action = 0
        # if no data is in memory make a random choice
        else:
            action = env.action_space.sample()

        # game doing the predicted action
        observation, reward, done, info = env.step(action)
        # appending the all of the data to memory
        memory.appendleft((state, observation, action, reward, done))
        if done:
            print("done after episode {}".format(t+1))
            break
    if i_episodes % 10 == 0:
        model.save('Reinforcement Learning/gym/models/CartPole-v1_{}.h5'.format(i_episodes))
env.close()
