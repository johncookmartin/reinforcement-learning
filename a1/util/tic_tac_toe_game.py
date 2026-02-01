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


class OpponentType(Enum):
    RANDOM = 0
    RANDOM_ROW = 1
    RANDOM_COL = 2
    RANDOM_DIAG = 3
    MIRROR = 4


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

    def check_is_draw(self):
        return self.get_full_board() & 0b111111111 == 0b111111111

    def get_full_board(self):
        return as_9bit(self.x_moves ^ self.o_moves)

    def play_move(self, move, is_x=True):
        if is_x:
            self.x_moves = self.x_moves | (1 << move)
        else:
            self.o_moves = self.o_moves | (1 << move)


class TicTacToeOpponent:
    def __init__(self, opponent_type, game, debug=False, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()
        self.opponent_type = opponent_type
        self.game = game
        self.debug = debug

    def make_move(self, prev_move):
        if self.opponent_type == OpponentType.RANDOM_ROW:
            options = self.get_row_options(prev_move)
        elif self.opponent_type == OpponentType.RANDOM_COL:
            options = self.get_column_options(prev_move)
        elif self.opponent_type == OpponentType.RANDOM_DIAG:
            options = self.get_diagonal_options(prev_move)
        elif self.opponent_type == OpponentType.MIRROR:
            options = self.get_mirror_options(prev_move)

        if options.bit_count() == 0:
            options = self.get_options(0b111111111)
        move = self.choose_random_move(options)
        self.game.play_move(move, False)

    def choose_random_move(self, options):
        options_count = options.bit_count()
        move = self.rng.randint(1, options_count)
        i = 0
        for j in range(9):
            bit = (options >> j) & 1
            if bit == 1:
                i += 1
            if i == move:
                return j
        return None

    def get_options(self, mask):
        board = self.game.get_full_board()
        return (board & mask) ^ mask

    def get_row_options(self, prev_move):
        if prev_move < 3:
            return self.get_options(0b000000111)
        elif prev_move < 6:
            return self.get_options(0b000111000)
        else:
            return self.get_options(0b111000000)

    def get_column_options(self, prev_move):
        if prev_move % 3 == 0:
            return self.get_options(0b001001001)
        elif prev_move % 3 == 1:
            return self.get_options(0b010010010)
        else:
            return self.get_options(0b100100100)

    def get_diagonal_options(self, prev_move):
        if prev_move == 0 or prev_move == 8:
            return self.get_options(0b100010001)
        if prev_move == 2 or prev_move == 6:
            return self.get_options(0b001010100)
        if prev_move == 4:
            return self.get_options(0b101010101)
        else:
            return self.get_options(0b111111111)

    def get_mirror_options(self, prev_move):
        if prev_move == 4:
            return self.get_options(0b111111111)
        else:
            i = prev_move
            for j in range(8, -1, -1):
                if i == 0:
                    return self.get_options(0b000000000 | (1 << j))
                else:
                    i -= 1
