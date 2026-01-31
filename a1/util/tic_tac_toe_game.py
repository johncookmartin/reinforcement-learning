import random
from enum import Enum

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


class OpponentType(Enum):
    RANDOM = 0
    RANDOM_ROW = 1
    RANDOM_COL = 2
    RANDOM_DIAG = 3


class TicTacToeGame:
    def __init__(self):
        self.x_moves = 0b000000000
        self.o_moves = 0b000000000

    def check_is_winning(self, check_x=True):
        moves = self.x_moves if check_x else self.o_moves
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

    def play_move(self, move, is_x=True):
        if is_x:
            self.x_moves = self.x_moves | (1 << move)
        else:
            self.o_moves = self.o_moves | (1 << move)


class TicTacToeOpponent:
    def __init__(self, opponent_type, game, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()
        self.opponent_type = opponent_type
        self.game = game

    def make_move(self, prev_move):
        if self.opponent_type == OpponentType.RANDOM_ROW:
            self.make_row_move(prev_move)
        elif self.opponent_type == OpponentType.RANDOM_COL:
            pass
        elif self.opponent_type == OpponentType.RANDOM_DIAG:
            pass
        else:
            pass

    def make_row_move(self, prev_move):
        if prev_move < 4:
            pass
        elif prev_move < 7:
            pass
        else:
            pass
