import gym

env = gym.make("CartPole-v1")

while True:
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()
