import gym

env = gym.make("MsPacman-v0")

while True:
    env.reset()
    while True:
        env.render('human')
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
