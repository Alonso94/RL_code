import gym
env=gym.make('FetchReach-v1')
env.reset()
print(env.action_space)
print(env.observation_space)