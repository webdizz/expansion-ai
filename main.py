import gym
from gym.envs.registration import register

# register WarLight gym env
register(
    id='WarLight-v0',
    entry_point='warlight_env:WarlightEnv',
)

env = gym.make('WarLight-v0')

print(env.reset())
env.render(mode='sim')
# print(env.observation_space.n)