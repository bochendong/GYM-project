import gym
from numpy.random.mtrand import gamma
from gridworld import CliffWalkingWapper
import numpy as np
import time

class Q_LearningAgent(object):
    def __init__(self, obs_n, act_n = 4, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.act_n = act_n
        self.Q = np.zeros((obs_n, act_n))
        
        info = "Q_LearningAgent with state number = " +  str(obs_n) + ", learning rate = " + str(learning_rate) + ", gamma = " + str(gamma) + " created."
        print(info)

    def sample(self, S):
        '''
        Input:
        S -> a number indicate state, in range (0, obs_n)
        Output:
        action -> an action determine by epsilon-greedy strategy.

        epsilon-greedy strategy:

        With epsilon probability, we perform a random action.
        With 1 -  epsilon probability, we perform the action has the maximum Q-value.
        '''

        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(S)
        else:
            action = np.random.choice(self.act_n)

        return action

    def predict(self, S):
        '''
        Input:
        S -> a number indicate state, in range (0, obs_n)
        Output:
        action -> an action which has the maximum Q-value corresponding to the state S.
        '''

        Q_values = self.Q[S]
        maxQ = np.max(self.Q[S])

        action = np.random.choice(np.where(Q_values == maxQ)[0])    # randomly choose an action has the maximum Q-value
        return action

    def learn(self, S, action, reward, S_prime, done):
        predit_Q = self.Q[S, action]

        if (done):
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[S_prime])

        self.Q[S, action] += self.lr * (target_Q - predit_Q)

def run_episode(env, agent, render = False):
    S = env.reset()
    action = agent.sample(S)
    
    (total_step, total_reward) = (0, 0)

    while (True):
        S_prime, reward, done, _ = env.step(action)
        action_prime = agent.sample(S_prime)

        agent.learn(S, action, reward, S_prime, done)

        action = action_prime
        S = S_prime
        total_reward += reward
        total_step += 1

        if render:
            env.render()    # Show grid3
        if done:
            break
    
    return (total_step, total_reward)


def main():
    env = gym.make("CliffWalking-v0")  # 0 left, 1 down, 2 right, 3 up
    env = CliffWalkingWapper(env)

    agent = Q_LearningAgent(
        obs_n=env.observation_space.n, 
        act_n=env.action_space.n, 
        learning_rate=0.1, 
        gamma = 0.9, 
        e_greed=0.2)

    is_render = False

    for episode in range(2000):
        run_episode(env, agent, is_render)

        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    
main()

'''def test_episode(env,agent):
    total_reward = 0
    obs = env.reset()
    
    while (True):
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)

        total_reward += reward
        obs = next_obs

        time.sleep(0.5)
        env.render()
        if done:
            break'''



