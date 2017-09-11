import numpy as np

board_size = 20
board_cell_actions = 8
armies = 10
MY_LAYER = 1
offset_y = 0
offset_x = 0

board = np.zeros((2, board_size, board_size))
board[MY_LAYER, board_size - 1 - offset_y, offset_x] = armies
board[MY_LAYER, board_size - 1 - offset_y, offset_x + 1] = armies

print("My Board: \n{}".format(board[MY_LAYER]))

action = 122

move_down = [5, 4, 3]
move_up = [7, 0, 1]
move_left = [7, 6, 5]
move_right = [1, 2, 3]

my_cells = np.argwhere(board[MY_LAYER] > 0)
print("My occupation \n{}".format(my_cells[0].size))

action_indices = np.resize(np.arange(0, 400), (400, 1))
actions_squared = np.resize(action_indices, (20, 20))
print("Actions to take \n{}".format(actions_squared))

print("Action 1: {} action 2 {}".format(actions_squared[my_cells[0][0]][my_cells[0][1]], actions_squared[my_cells[1][0]][my_cells[1][1]]))

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
