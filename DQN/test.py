from dqn import *

def test(max_iter=20000, max_episodes=100):
    env = gym.make("CartPole-v0")
    obs_n = env.observation_space.shape[0]

    render = True

    agent = DQNAgent(
        obs_n = env.observation_space.shape[0], 
        act_n = env.action_space.n,
        )

    if os.path.exists("./cartpole_dqn.h5"):
        agent.model.load_weights("./cartpole_dqn.h5")


    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, obs_n])
        while score <= max_iter:
            if (render):
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, obs_n])

            score += reward
            state = next_state

            if done or score >= max_iter:
                print("(episode: {}; score: {};)".format(epoch, score))
                break


env = gym.make("CartPole-v0")

agent = DQNAgent(
        obs_n = env.observation_space.shape[0], 
        act_n = env.action_space.n)

obs_n = env.observation_space.shape[0]
S = env.reset() 

S = np.reshape(S, [1, obs_n])

print(agent.model.predict(S))