import logging
import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from six import StringIO

logger = logging.getLogger('ExpansionAiEnv')


class ExpansionAiEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, board_size=4, armies=4, offset_x=0, offset_y=0):
        """ Initializes an env """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        self.armies = armies
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.cell_movements = 8  # 8 movements per cell
        logger.info('Env init was called with board size "%s" and armies "%s"' % (board_size, armies))

        self.board = np.random.randn(self.board_size, self.board_size)
        self.board_flatten = self.board.flatten()

        self.board_action_space = np.random.randn(self.board_size * self.board_size, self.cell_movements)
        self.board_action_space_flatten = self.board_action_space.flatten()

        self.move_down = [5, 4, 3]
        self.move_up = [7, 0, 1]
        self.move_left = [7, 6, 5]
        self.move_right = [1, 2, 3]

        # TODO: add armies into account
        self.action_space = spaces.Discrete(board_size * board_size * self.cell_movements)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))
        logger.info("Env action_space: %s and observation_space: %s" % (self.action_space, self.observation_space))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """ add step logic here """
        self.step_num += 1
        if self.done:
            return self.state, 0., True, {'state': self.state}
        prev_state = self.state
        self.move(action)
        reward = self.game_finished()
        logger.debug(
            'Env movement after step {} for action "{}" for armies {} with state \n{}\n lead to new state \n{}\n'.format(
                self.step_num, action,
                self.armies,
                prev_state[
                    0],
                self.state[
                    0]))

        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    def _reset(self):
        self.step_num = 0
        self.state = np.zeros((2, self.board_size, self.board_size))
        self.done = False
        logger.debug("Env model initial state: \n{}".format(self.state[0]))
        # place armies to initial state
        self.state[0, self.board_size - 1 - self.offset_y, self.offset_x] = self.armies
        # print("= Model with my armies state: {}".format(self.state))
        # TODO: add first moves
        return self.state

    def _render(self, mode='ansi', close=False):
        """ Renders environment """
        logger.debug('Env render was executed with mode "{}" and close "{}'.format(mode, close))
        if close:
            return

        # process board
        board = self.state
        out_file = StringIO() if mode == 'ansi' else sys.stdout
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

    def move(self, action):
        next_coordinates = self.action_to_coordinate(action)
        new_armies = 1
        current_armies_in_cell = self.state[0, next_coordinates[0], next_coordinates[1]]
        self.state[0, next_coordinates[0], next_coordinates[1]] = new_armies + current_armies_in_cell
        # print("New board state {}".format(self.state))

    def action_to_coordinate(self, action):
        action_movement = np.argwhere(self.board_action_space == self.board_action_space_flatten[action])[0]
        current_position_value = self.board_flatten[action_movement[0]]
        current_position = np.argwhere(self.board == current_position_value)[0]

        next_position_row = current_position[0]
        if np.argwhere(self.move_up == action_movement[1]) > -1:
            next_position_row -= 1
        elif np.argwhere(self.move_down == action_movement[1]) > -1:
            next_position_row += 1
        if next_position_row == -1:
            next_position_row = current_position[0]
        if next_position_row > self.board_size - 1:
            next_position_row = current_position[0]

        next_position_col = current_position[1]
        if np.argwhere(self.move_left == action_movement[1]) > -1:
            next_position_col -= 1
        elif np.argwhere(self.move_right == action_movement[1]) > -1:
            next_position_col += 1
        if next_position_col == -1:
            next_position_col = current_position[1]
        if next_position_col > self.board_size - 1:
            next_position_col = current_position[1]

        next_position = [next_position_row, next_position_col]
        return next_position

    def game_finished(self):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        self.armies = current_num_of_armies = np.sum(self.state[0], dtype=np.int32)
        logger.debug("Env current armies num %s" % (current_num_of_armies))
        if 0 not in self.state[0]:
            logger.info("Env wow, is about to get a reward \n{}\n".format((self.state[0])))
            return 1
        elif current_num_of_armies <= 0:
            return -1  # our army was destroyed
        else:
            return 0
