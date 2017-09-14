import json
import math
import numpy as np


class ExpansionRequest:
    EXIT = [-5, 'E']
    HOLE = [-4, 'O']
    BREAK = [-3, 'B']
    WALL = [-2, '']
    GOLD = [-1, '$']

    PLAYERS = [[-10000, '♥'], [-20000, '♦'], [-30000, '♣'], [-40000, '♠']]

    def __init__(self, raw_request):
        json_content = self._read_as_json(raw_request)
        self.offset_x = json_content['myBase']['x']
        self.offset_y = json_content['myBase']['y']
        self.my_force_id = json_content['myColor']
        self.armies = json_content['available']
        self.tick = json_content['tick']
        self.lobby = json_content['inLobby']

        # construct board with infra and enemy armies
        self.board = np.zeros((20, 20), dtype=np.int)
        self._build_board(json_content)

    def _build_board(self, json_content):
        self._build_playground(json_content['layers'][0])
        self._place_forces_pos(json_content['layers'][1])
        self._place_forces_num(json_content['forces'])

    def _place_forces_num(self, board_source):
        line_length = int(math.sqrt(len(board_source) / 3) * 3)

        row_num = 0
        for row in self._chunk_str(board_source, line_length):
            cell_num = 0
            for cell in self._chunk_str(row, 3):
                cell_value = self.board[row_num][cell_num]
                self.board[row_num][cell_num] = cell_value if cell == '-=#' else int(cell, 36) + cell_value
                cell_num += 1
            row_num += 1

    def _place_forces_pos(self, playground_source):
        line_length = int(math.sqrt(len(playground_source)))

        row_num = 0
        for row in self._chunk_str(playground_source, line_length):
            cell_num = 0
            for cell in self._chunk_str(row, 1):
                self.board[row_num][cell_num] = self._transform_force_cell(cell, self.board[row_num][cell_num])
                cell_num += 1
            row_num += 1

    def _build_playground(self, playground_source):
        line_length = int(math.sqrt(len(playground_source)))

        row_num = 0
        for row in self._chunk_str(playground_source, line_length):
            cell_num = 0
            for cell in self._chunk_str(row, 1):
                self.board[row_num][cell_num] = self._transform_cell_value(cell)
                cell_num += 1
            row_num += 1

    def _transform_force_cell(self, cell, curr_value):
        return {
            self.PLAYERS[0][1]: self.PLAYERS[0][0],
            self.PLAYERS[1][1]: self.PLAYERS[1][0],
            self.PLAYERS[2][1]: self.PLAYERS[2][0],
            self.PLAYERS[3][1]: self.PLAYERS[3][0]
        }.get(cell, curr_value)

    def _transform_cell_value(self, cell):
        return {
            self.EXIT[1]: self.EXIT[0],
            self.BREAK[1]: self.BREAK[0],
            self.GOLD[1]: self.GOLD[0],
            self.HOLE[1]: self.HOLE[0]
        }.get(cell, self.WALL[0])

    def _chunk_str(self, string, chunk_size):
        return [string[i:i + chunk_size] for i in range(0, len(string), chunk_size)]

    def _read_as_json(self, msg):
        return json.loads(msg.replace('board=', ''))
