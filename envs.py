import gym
from gym import wrappers
from gym.envs.registration import register

from config import Params

# register ExpansionAi gym env
register(
    id='ExpansionAi-v0',
    entry_point='expansionai_env:ExpansionAiEnv',
)


def create_expansionai_env(env_id, video=False, params=Params()):
    gym.spec(env_id)._kwargs = {
        'armies': params.armies,
        'board_size': params.board_size,
        'offset_x': 0,
        'offset_y': 0
    }
    env = gym.make(env_id)
    if video:
        env = wrappers.Monitor(env, 'test', force=True, mode='training')
    return env
