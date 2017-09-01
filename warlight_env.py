import gym
import numpy as np
import sys
from six import StringIO


class WarlightEnv(gym.Env):
    metadata = {'render.modes': ['sim', 'live']}

    def __init__(self, board_size=4):
        """ Initializes an env """
        # assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        print('= Init was called with board size "{}"'.format(board_size))

    def _step(self, action):
        """ add step logic here """
        print('= Step was executed with action "{}"'.format(action))

    def _reset(self):
        print('= Reset was executed ')
        self.state = np.zeros((1, self.board_size, self.board_size))
        self.done = False
        # TODO: add place armies
        self.state[0, 2, 3] = 1.0
        # TODO: add first moves
        print('= State shape is "{}" '.format(self.state.shape))
        return self.state

    def _render(self, mode='sim', close=False):
        """ Renders environment """
        print('= Render was executed with mode "{}" and close "{}'.format(mode, close))
        if close:
            return

        # process board
        board = self.state
        out_file = StringIO() if mode == 'live' else sys.stdout
        out_file.write(' ' * 13)
        out_file.write('\t')

        for column in range(board.shape[1]):
            out_file.write('\t' + str(column + 1) + '|')
        out_file.write('\n')

        # underline
        out_file.write('\t')
        out_file.write('-' * (self.board_size * 11 - 2))
        out_file.write('\n')
        # end of header #

        for row in range(board.shape[1]):
            out_file.write('\t')
            out_file.write(str(row + 1) + '\t|')
            for column in range(board.shape[1]):
                out_file.write(str(board[0, row, column]))
                out_file.write('\t|')
            out_file.write('\n')

            # horizontal line
            out_file.write('\t')
            out_file.write('-' * (self.board_size * 11 - 3))
            out_file.write('\n')

        if mode != 'live':
            return out_file
