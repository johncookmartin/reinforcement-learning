# John's Implementation

import argparse
from enum import Enum
import random
from util.tic_tac_toe_game import TicTacToeGame
from util.functions import as_9bit


class PlayerType(Enum):
    PLAYER = 0
    RANDOM = 1
    RANDOM_ROW = 2
    RANDOM_COL = 3
    RANDOM_DIAG = 4
    MIRROR = 5


class Player:
    def __init__(self, game, player_type, learning_rate=0.1, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()

        self.player_type = player_type
        self.learning_rate = learning_rate
        self.game = game
        self.states = {}

    def play_move(self, prev_move=None):
        if self.player_type == PlayerType.PLAYER:
            move = self.choose_player_move()
        else:
            move = self.choose_opponent_move(prev_move)
        self.game.play_move(move, self.player_type == PlayerType.PLAYER)
        return move

    def choose_player_move(self):
        options = self.game.get_options()
        if self.rng.random() <= 0.1:
            return self.choose_random_move(options)
        else:
            return self.choose_optimal_move(options)

    def choose_opponent_move(self, prev_move):
        if prev_move is None:
            return self.choose_player_move()
        elif self.player_type == PlayerType.RANDOM_ROW:
            options = self.get_row_options(prev_move)
        elif self.player_type == PlayerType.RANDOM_COL:
            options = self.get_column_options(prev_move)
        elif self.player_type == PlayerType.RANDOM_DIAG:
            options = self.get_diagonal_options(prev_move)
        elif self.player_type == PlayerType.MIRROR:
            options = self.get_mirror_options(prev_move)

        if options is None or as_9bit(options).bit_count() == 0:
            options = self.game.get_options()
        return self.choose_random_move(options)

    def choose_random_move(self, options):
        options_count = as_9bit(options).bit_count()
        move = self.rng.randint(1, options_count)
        i = 0
        for j in range(9):
            bit = (options >> j) & 1
            if bit == 1:
                i += 1
            if i == move:
                return j

    def choose_optimal_move(self, options):
        i = 0
        best_move = 0
        best_move_value = 0
        for j in range(9):
            bit = (options >> j) & 1
            if bit == 1:
                i += 1
                option_value = self.get_move_value(j)
                if option_value > best_move_value:
                    best_move_value = option_value
                    best_move = j
        return best_move

    def get_row_options(self, prev_move):
        if prev_move < 3:
            return self.game.get_options(0b000000111)
        elif prev_move < 6:
            return self.game.get_options(0b000111000)
        else:
            return self.game.get_options(0b111000000)

    def get_column_options(self, prev_move):
        if prev_move % 3 == 0:
            return self.game.get_options(0b001001001)
        elif prev_move % 3 == 1:
            return self.game.get_options(0b010010010)
        else:
            return self.game.get_options(0b100100100)

    def get_diagonal_options(self, prev_move):
        if prev_move == 0 or prev_move == 8:
            return self.game.get_options(0b100010001)
        if prev_move == 2 or prev_move == 6:
            return self.game.get_options(0b001010100)
        if prev_move == 4:
            return self.game.get_options(0b101010101)
        else:
            return self.game.get_options(0b111111111)

    def get_mirror_options(self, prev_move):
        if prev_move == 4:
            return self.game.get_options(0b111111111)
        else:
            i = prev_move
            for j in range(8, -1, -1):
                if i == 0:
                    return self.game.get_options(0b000000000 | (1 << j))
                else:
                    i -= 1

    def get_move_value(self, move):
        player_state = (
            self.game.x_moves
            if self.player_type == PlayerType.PLAYER
            else self.game.o_moves
        )
        player_state = player_state | (1 << move)
        opponent_state = (
            self.game.o_moves
            if self.player_type == PlayerType.PLAYER
            else self.game.x_moves
        )
        return self.consider_state(player_state, opponent_state)

    def consider_state(self, player_state, opponent_state):
        state = (as_9bit(player_state) << 9) | as_9bit(opponent_state)
        if state in self.states:
            return self.states[state]
        elif self.game.check_is_winning(player_state):
            self.states[state] = 1
            return 1
        else:
            self.states[state] = 0.5
            return 0.5

    def log_move(self, value):
        current_state = (as_9bit(self.game.x_moves) << 9) | as_9bit(self.game.o_moves)
        current_value = self.consider_state(current_state)
        new_value = current_value + self.learning_rate * (value - current_value)
        self.states[current_state] = new_value


def main(args):
    game = TicTacToeGame()
    opponent = Player(game, PlayerType.RANDOM_ROW)
    player = Player(game, PlayerType.PLAYER)
    keep_playing = True
    while keep_playing:
        print("current board:")
        print(f"{game.get_full_board():09b}")
        print()

        input()
        next_move = player.play_move()
        print("x played:")
        print(f"{game.x_moves:09b}")
        print()
        x_won = game.check_is_winning()
        is_draw = game.check_is_draw()
        if not x_won and not is_draw:
            opponent.play_move(next_move)
            print("o played:")
            print(f"{game.o_moves:09b}")
            print()
            o_won = game.check_is_winning(False)
        if x_won or o_won or is_draw:
            keep_playing = False

    print(f"{game.get_full_board():09b}")
    if x_won:
        print(f"{game.x_moves:09b}")
        print("X won!")
    elif o_won:
        print(f"{game.o_moves:09b}")
        print("O Won!")
    else:
        print(f"{game.x_moves:09b}")
        print("Draw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic Tac Toe Implementation")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
