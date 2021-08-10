import gym
import random
import os
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class DQNAgent:
    def __init__(self, obs_n, act_n, gamma = 0.95, 
                    learning_rate=0.001, batch_size=20, memory_size=1000000):
        self.obs_n = obs_n
        self.act_n = act_n

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(obs_n,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(act_n, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

        self.memory = deque(maxlen=memory_size)


    def choose_action(self, S):
        if np.random.rand() <  self.epsilon:
            return random.randrange(self.act_n)
        q_value = self.model.predict(S)
        return np.argmax(q_value[0])

    def add_memory(self, S, action, reward, S_prime, done):
        self.memory.append((S, action, reward, S_prime, done))

    def learn(self):
        if (len(self.memory) < self.batch_size):
            return

        min_batch = random.sample(self.memory, self.batch_size)
        
        for S, action, reward, S_prime, done in min_batch:
            q_update = reward
            if not done:
                q_update = (reward + self.gamma * np.amax(self.model.predict(S_prime)[0]))
            q_values = self.model.predict(S)
            q_values[0][action] = q_update

            self.model.fit(S, q_values, verbose=0)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)