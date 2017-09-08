import logging
import os

import torch
import torch.multiprocessing as mp

import optimizer as my_optim
from config import Params
from envs import create_expansionai_env
from model import ActorCritic
from test import test
from train import train

LOGGING_FORMAT = '%(asctime)s - %(name)s - %(thread)d|%(process)d - %(levelname)s - %(message)s'
logging.basicConfig(format=LOGGING_FORMAT)

# logging.getLogger('Model').setLevel(logging.INFO)
# logging.getLogger('WarlightEnv').setLevel(logging.INFO)
logging.getLogger('Train').setLevel(logging.DEBUG)
# logging.getLogger('Test').setLevel(logging.INFO)

# Main run
os.environ['OMP_NUM_THREADS'] = '1'  # 1 thread per core
params = Params()  # creating the params object from the Params class, that sets all the model parameters
params.max_episode_length = 100
params.num_processes = 1

torch.manual_seed(params.seed)  # setting the seed (not essential)
env = create_expansionai_env(params.env_name, params)  # we create an optimized environment thanks to universe

# shared_model is the model shared by the different agents (different threads in different cores)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
# storing the model in the shared memory of the computer, which allows the threads to have access to this shared memory even if they are in different cores
shared_model.share_memory()

# the optimizer is also shared because it acts on the shared model
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
optimizer.share_memory()  # same, we store the optimizer in the shared memory so that all the agents can have access to this shared memory to optimize the model

processes = []  # initializing the processes with an empty list

# making a loop to run all the other processes that will be trained by updating the shared model
for rank in range(0, params.num_processes):
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)

# allowing to create the 'test' process with some arguments 'args' passed to the 'test' target function - the 'test' process doesn't update the shared model but uses it on a part of it - torch.multiprocessing.Process runs a function in an independent thread
p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
p.start()  # starting the created process p
processes.append(p)  # adding the created process p to the list of processes

# creating a pointer that will allow to kill all the threads when at least one of the threads, or main.py will be killed, allowing to stop the program safely
for p in processes:
    p.join()
