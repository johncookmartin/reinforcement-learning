# John's Implementation

import argparse
from enum import Enum
import random
from util.tic_tac_toe_game import TicTacToeGame
from util.functions import calculate_running_average, plot_results


class PlayerType(Enum):
    PLAYER = 0
    RANDOM = 1
    RANDOM_ROW = 2
    RANDOM_COL = 3
    RANDOM_DIAG = 4
    MIRROR = 5
    SELF = 6


class Player:
    def __init__(
        self, game, player_type, seed=None, learning_rate=0.1, exploring_rate=0.1
    ):
        self.rng = random.Random(seed) if seed else random.Random()

        self.player_type = player_type
        self.learning_rate = learning_rate
        self.exploring_rate = exploring_rate
        self.game = game

        self.states = {}
        self.prev_state = 0

        self.total_moves = 0
        self.average_change_rate = 0
        self.max_change_rate = 0

    def play_move(self, prev_move=None):
        if self.player_type == PlayerType.PLAYER or self.player_type == PlayerType.SELF:
            move = self.choose_player_move()
        else:
            move = self.choose_opponent_move(prev_move)
        self.game.play_move(move, self.player_type == PlayerType.PLAYER)
        self.log_value()
        self.prev_state = self.get_current_combined_state()
        return move

    def choose_player_move(self):
        options = self.game.get_options()
        if self.rng.random() <= self.exploring_rate:
            return self.choose_random_move(options)
        else:
            return self.choose_optimal_move(options)

    def choose_opponent_move(self, prev_move):
        options = None
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
        else:
            options = self.game.get_options()

        if options is None or options.bit_count() == 0:
            options = self.game.get_options()
        return self.choose_random_move(options)

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

    def choose_optimal_move(self, options):
        i = 0
        best_move = 0
        best_move_value = -1
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
        player_state, opponent_state = self.get_current_states()
        player_state = player_state | (1 << move)
        return self.consider_state(player_state, opponent_state)

    def get_current_states(self):
        player_state = (
            self.game.x_moves
            if self.player_type == PlayerType.PLAYER
            else self.game.o_moves
        )
        opponent_state = (
            self.game.o_moves
            if self.player_type == PlayerType.PLAYER
            else self.game.x_moves
        )
        return player_state, opponent_state

    # def consider_state(self, player_state, opponent_state):
    #     state = (player_state << 9) | opponent_state
    #     if state in self.states:
    #         return self.states[state]["value"]
    #     elif self.game.check_is_winning(
    #         self.player_type == PlayerType.PLAYER, player_state
    #     ):
    #         self.states[state] = {"value": 1, "change_rate": 0, "changes": 0}
    #         return 1
    #     elif self.game.check_is_winning(
    #         self.player_type != PlayerType.PLAYER, opponent_state
    #     ):
    #         self.states[state] = {"value": 0, "change_rate": 0, "changes": 0}
    #         return 0
    #     elif self.game.check_is_draw(state):
    #         self.states[state] = {"value": 0, "change_rate": 0, "changes": 0}
    #         return 0
    #     else:
    #         self.states[state] = {"value": 0.5, "change_rate": 0, "changes": 0}
    #         return 0.5

    def consider_state(self, player_state, opponent_state):
        state = (player_state << 9 | opponent_state)
        if state in self.states:
            return self.states[state]["value"]
        else:
            return 0.5

    def get_current_combined_state(self):
        player_state, opponent_state = self.get_current_states()
        return (player_state << 9) | opponent_state

    def log_value(self):
        player_state, opponent_state = self.get_current_states()
        current_value = self.consider_state(player_state, opponent_state)
        if self.prev_state not in self.states:
            prev_player_state = self.prev_state >> 9
            prev_opponent_state = self.prev_state & 0b111111111
            self.consider_state(prev_player_state, prev_opponent_state)
        new_value = self.states[self.prev_state]["value"] + self.learning_rate * (
            current_value - self.states[self.prev_state]["value"]
        )
        self.total_moves += 1
        self.max_change_rate = abs(
            max(self.max_change_rate, new_value - self.states[self.prev_state]["value"])
        )
        self.average_change_rate = calculate_running_average(
            self.average_change_rate,
            abs(new_value - self.states[self.prev_state]["value"]),
            self.total_moves,
        )
        self.states[self.prev_state]["value"] = new_value


def main(args):
    game = TicTacToeGame()
    opponent = Player(
        game, args.opponent_type, args.seed, args.learning_rate, args.exploring_rate
    )
    player = Player(
        game, PlayerType.PLAYER, args.seed, args.learning_rate, args.exploring_rate
    )

    wins = 0
    record = []

    average_change_rate = 1
    i = 0
    while average_change_rate >= 0.001:
        game_ended = False
        player_move = player.play_move()

        # check if player won
        if game.check_is_winning():
            wins += 1
            game_ended = True
            opponent.log_value()

        # check for draw
        if game.check_is_draw():
            game_ended = True
            player.log_value()
            opponent.log_value()

        if not game_ended:
            opponent.play_move(player_move)

        # check if opponent won
        if game.check_is_winning(False):
            game_ended = True
            player.log_value()

        if game_ended:
            game.clear_board()
            player.prev_state = 0
            opponent.prev_state = 0
            print(f"average change rate: {player.average_change_rate}")
            player.max_change_rate = 0
            i += 1
            win_rate = wins / i
            record.append(win_rate)
            average_change_rate = player.average_change_rate

    plot_results(
        [
            {"record": record, "label": "win rate"},
        ],
        "Win Rate",
        "Game",
        "Percentage",
    )


def player_type_parser(value):
    try:
        return PlayerType[value.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid player type: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic Tac Toe Implementation")
    parser.add_argument("--num_rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--exploring_rate", type=float, default=0.1)
    parser.add_argument(
        "--opponent_type", type=player_type_parser, default=PlayerType.RANDOM
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
