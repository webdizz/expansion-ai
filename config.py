class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 3
        self.num_steps = 20
        self.max_episode_length = 1000000
        self.env_name = 'ExpansionAi-v0'
        self.lstm_size = 256
        self.board_size = 20
        self.board_dimension_size = 2
        self.initial_army_size = 5
