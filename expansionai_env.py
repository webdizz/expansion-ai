import logging
import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from six import StringIO

logger = logging.getLogger('ExpansionAiEnv')

MY_LAYER = 1


class ExpansionAiEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, board_size=4, armies=4, offset_x=0, offset_y=0):
        """ Initializes an env """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        self.armies = armies
        self.initial_armies = armies
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
        self.action_space = spaces.Discrete(self.cell_movements)
        # self.action_space = spaces.Box(0, 8, shape=400)
        # .Discrete(board_size * board_size * self.cell_movements)
        self.observation_space = spaces.Box(-4, 20, shape=400)
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
        # resize actions to have it squared to simplify coordination
        actions_squared = np.resize(action, (20, 20))

        self.move(actions_squared)
        reward = self.game_finished()
        logger.debug(
            'Env movement after step {} for action "{}" for armies {} lead to new state \n{}\n'.format(
                self.step_num, action,
                self.armies,
                self.state[MY_LAYER]))

        self.done = reward >= 1 or reward == -1
        return self.state, reward, self.done, {'state': self.state, 'step': self.step_num, 'armies': self.armies}

    def move(self, actions_squared, new_armies=10):
        occupied_cells = self.resolve_occupied_cells()
        movable_cells = self.resolve_movable_cells()

        logger.debug(("Env occupied cells \n{}\n movable cells \n{}".format(occupied_cells, movable_cells)))

        for cell_to_move in movable_cells:
            movement_action = actions_squared[cell_to_move[0]][cell_to_move[1]]
            to_pos = self.action_to_coordinate(movement_action, cell_to_move)
            previously_armies_in_cell = self.state[MY_LAYER, cell_to_move[0], cell_to_move[1]]
            armies_to_move = previously_armies_in_cell - 1
            # keep at least 1 army in cell
            self.state[MY_LAYER, cell_to_move[0], cell_to_move[1]] = previously_armies_in_cell - armies_to_move

            current_armies_in_cell = self.state[MY_LAYER, to_pos[0], to_pos[1]]
            logger.debug(
                "Env move cell \n{}\n to new position \n{} according to {}".format(cell_to_move, to_pos,
                                                                                   movement_action))
            self.state[MY_LAYER, to_pos[0], to_pos[1]] = new_armies + current_armies_in_cell + armies_to_move

        logger.debug("New board state {}".format(self.state[MY_LAYER]))

    def _reset(self):
        self.step_num = 0
        self.state = np.zeros((2, self.board_size, self.board_size))
        self.done = False
        # place armies to initial state
        self.state[MY_LAYER, :, :] = 0
        self.armies = self.initial_armies
        self.state[MY_LAYER, self.board_size - 2 - self.offset_y, self.offset_x] = self.armies
        self.resolve_occupied_cells()
        self.resolve_movable_cells()
        logger.debug("Env model initial state: \n{}".format(self.state[MY_LAYER]))
        return self.state

    def resolve_movable_cells(self):
        movable_cells = np.argwhere(self.state[MY_LAYER] > 1)
        self.movable_cells_num = movable_cells.size
        return movable_cells

    def resolve_occupied_cells(self):
        occupied_cells = np.argwhere(self.state[MY_LAYER] > 0)
        self.occupied_cells_num = occupied_cells.size
        return occupied_cells

    def action_to_coordinate(self, movement_action, current_position):
        movement_action = np.argwhere(self.board_action_space == self.board_action_space_flatten[movement_action])[0]

        next_position_row = current_position[0]
        if np.argwhere(self.move_up == movement_action[1]) > -1:
            next_position_row -= 1
        elif np.argwhere(self.move_down == movement_action[1]) > -1:
            next_position_row += 1
        if next_position_row == -1:
            next_position_row = current_position[0]
        if next_position_row > self.board_size - 1:
            next_position_row = current_position[0]

        next_position_col = current_position[1]
        if np.argwhere(self.move_left == movement_action[1]) > -1:
            next_position_col -= 1
        elif np.argwhere(self.move_right == movement_action[1]) > -1:
            next_position_col += 1
        if next_position_col == -1:
            next_position_col = current_position[1]
        if next_position_col > self.board_size - 1:
            next_position_col = current_position[1]

        next_position = [next_position_row, next_position_col]
        return next_position

    def game_finished(self):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        self.armies = current_num_of_armies = np.sum(self.state[MY_LAYER], dtype=np.int32)
        logger.debug("Env current armies num %s" % current_num_of_armies)
        occupied_cells_num = self.occupied_cells_num
        self.resolve_occupied_cells()
        current_occupied_cells_num = self.occupied_cells_num

        movable_cells_num = self.movable_cells_num
        self.resolve_movable_cells()
        current_movable_cells = self.movable_cells_num

        if 0 not in self.state[MY_LAYER]:
            logger.info("Env wow, is about to get a reward \n{}\n".format((self.state[MY_LAYER])))
            return 1
        elif (current_num_of_armies <= 0 or current_num_of_armies > 6000) and self.step_num > 0:
            return -1  # our army was destroyed
        elif np.argwhere(self.state[MY_LAYER] < 0).size > 0:
            return -1
        elif self.step_num >= 600:
            return -1
        elif current_occupied_cells_num - occupied_cells_num > 2:
            return 0.01
        elif current_movable_cells - movable_cells_num > 2:
            return 0.05
        elif current_occupied_cells_num > 200:
            return 0.01
        else:
            return 0


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
            out_file.write(str(board[MY_LAYER, row, column]))
            out_file.write('\t|')
        out_file.write('\n')

        # horizontal line
        out_file.write('\t')
        out_file.write('-' * (self.board_size * 11 - 3))
        out_file.write('\n')

    if mode != 'live':
        return out_file
