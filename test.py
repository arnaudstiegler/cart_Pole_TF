import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    out = env.step(0) # take a random action
    print(env.action_space.sample())
    print('obs')
    print(out[0])
    print('reward')
    print(out[1])
    print('done')
    print(out[2])
    print('info')
    print(out[3])