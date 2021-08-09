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

def main():
    max_episodes = 1000

    env = gym.make("CartPole-v0")
    obs_n = env.observation_space.shape[0]

    agent = DQNAgent(
        obs_n = env.observation_space.shape[0], 
        act_n = env.action_space.n)

    run = 0

    scores = [0]

    while (np.mean(scores[-5:]) <= 150 or run <= 30):
        run += 1
        step = 0

        S = env.reset()
        S = np.reshape(S, [1, obs_n])
        
        while True:
            step += 1
            action = agent.choose_action(S)
            S_prime, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            S_prime = np.reshape(S_prime, [1, obs_n])
            agent.add_memory(S, action, reward, S_prime, done)
            S = S_prime

            if done:
                print ("Run: " + str(run) + ", exploration: " + str(agent.epsilon) + ", score: " + str(step))
                scores.append(step)
                break
            agent.learn()

    agent.model.save_weights("./cartpole_dqn.h5")

def test(max_iter=20000, max_episodes=100):
    env = gym.make("CartPole-v0")
    env = env.unwrapped
    obs_n = env.observation_space.shape[0]

    render = False

    agent = DQNAgent(
        obs_n = env.observation_space.shape[0], 
        act_n = env.action_space.n,
        )

    if os.path.exists("./cartpole_dqn.h5"):
        agent.model.load_weights("./cartpole_dqn.h5")

    scores = []

    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, obs_n])
        while score <= max_iter:
            if (render):
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, obs_n])

            score += reward
            state = next_state

            if done or score >= max_iter:
                print("(episode: {}; score: {};)"
                      .format(epoch, score))
                scores.append(score)
                break



if __name__ == '__main__':
    main()
    print("train done")
    test()













