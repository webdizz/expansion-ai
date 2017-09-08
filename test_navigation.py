import numpy as np

board_size = 4
board_cell_actions = 8

board = np.random.randn(board_size, board_size)
board_flatten = board.flatten()
print("Board: {}".format(board))

board_action_space = np.random.randn(board_size * board_size, board_cell_actions)
board_action_space_flatten = board_action_space.flatten()
print("Board action space: {}".format(board_action_space))

action = 122

move_down = [5, 4, 3]
move_up = [7, 0, 1]
move_left = [7, 6, 5]
move_right = [1, 2, 3]

action_movement = np.argwhere(board_action_space == board_action_space_flatten[action])[0]
current_position_value = board_flatten[action_movement[0]]
current_position = np.argwhere(board == current_position_value)[0]

next_position_row = current_position[0]
if np.argwhere(move_up == action_movement[1]) > -1:
    next_position_row -= 1
elif np.argwhere(move_down == action_movement[1]) > -1:
    next_position_row += 1
if next_position_row == -1:
    next_position_row = current_position[0]
if next_position_row > board_size - 1:
    next_position_row = current_position[0]

next_position_col = current_position[1]
if np.argwhere(move_left == action_movement[1]) > -1:
    next_position_col -= 1
elif np.argwhere(move_right == action_movement[1]) > -1:
    next_position_col += 1
if next_position_col == -1:
    next_position_col = current_position[1]
if next_position_col > board_size - 1:
    next_position_col = current_position[1]

next_position = [next_position_row, next_position_col]
# print("Action movement {} from current position {} to {}".format(action_movement, current_position, next_position))



from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}

# Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
# Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape

x = space.shape
print(x)
observation = np.zeros((2, 4, 4))
observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))
print(np.zeros(observation.shape))