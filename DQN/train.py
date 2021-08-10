from dqn import *

def train():
    env = gym.make("CartPole-v0")
    obs_n = env.observation_space.shape[0]

    agent = DQNAgent(
        obs_n = env.observation_space.shape[0], 
        act_n = env.action_space.n)

    run = 0
    scores = [0]

    while (np.mean(scores[-5:]) <= 190 or run <= 30):
        run += 1
        score = 0

        S = env.reset()
        S = np.reshape(S, [1, obs_n])
        
        while True:
            score += 1
            action = agent.choose_action(S)
            S_prime, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            S_prime = np.reshape(S_prime, [1, obs_n])
            agent.add_memory(S, action, reward, S_prime, done)
            S = S_prime

            if done:
                print ("Run: " + str(run) + ", exploration: " + str(agent.epsilon) + ", score: " + str(score))
                scores.append(score)
                break
            agent.learn()

    agent.model.save_weights("./cartpole_dqn.h5")
