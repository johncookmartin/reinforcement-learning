import random
from enum import Enum
from util.functions import as_9bit

# represent the game with 9 bits
#
#    For example, if the board is numbered:
#
#    0 | 1 | 2
#    ---------
#    3 | 4 | 5
#    ---------
#    6 | 7 | 8
#
#    The bit array corresponds to the
#    squares like this:
#    0b876543210
#
#    1 represents a move taken by x or o
#    depending on the array
#
#    For example the following position:
#
#    O |   | X
#    ---------
#      | O |
#    ---------
#    X | X |
#
#   would be represented by these two arrays:
#   0b100010000    <-- Os moves
#   0b001000110    <-- Xs moves
#


class TicTacToeGame:
    def __init__(self):
        self.x_moves = 0b000000000
        self.o_moves = 0b000000000

    def check_is_winning(self, check_x=True, state=None):
        if state is None:
            moves = self.x_moves if check_x else self.o_moves
        else:
            moves = as_9bit(state)

        winning_maps = [
            0b111000000,
            0b100100100,
            0b010010010,
            0b001001001,
            0b000111000,
            0b000000111,
            0b100010001,
            0b001010100,
        ]
        for winning_map in winning_maps:
            if (winning_map & moves) == winning_map:
                return True
        return False

    def check_is_draw(self, state=None):
        moves = self.get_full_board() if state is None else as_9bit(state)
        return moves & 0b111111111 == 0b111111111

    def get_full_board(self):
        return as_9bit(self.x_moves ^ self.o_moves)

    def get_options(self, mask=0b111111111):
        return (self.get_full_board() & mask) ^ mask

    def play_move(self, move, is_x=True):
        if is_x:
            self.x_moves |= 1 << move
        else:
            self.o_moves |= 1 << move
