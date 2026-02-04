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
        self.total_games = 0
        self.is_winning = False
        self.win_record = []
        self.win_rate = 0

        self.average_change = 0

    def play_move(self, prev_move=None):
        # check if game is over
        if self.game.check_is_draw() or self.game.check_is_winning(
            self.player_type != PlayerType.PLAYER
        ):
            self.log_no_win()
            return -1

        # choose move
        if self.player_type == PlayerType.PLAYER or self.player_type == PlayerType.SELF:
            move = self.choose_player_move()
        else:
            move = self.choose_opponent_move(prev_move)
        self.game.play_move(move, self.player_type == PlayerType.PLAYER)
        self.total_moves += 1
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
        best_move_arr = []
        best_move_value = -1
        for j in range(9):
            bit = (options >> j) & 1
            if bit == 1:
                i += 1
                option_value = self.get_move_value(j)
                if option_value > best_move_value:
                    best_move_arr = []
                    best_move_value = option_value
                    best_move_arr.append(j)
                elif option_value == best_move_value:
                    best_move_arr.append(j)

        return random.choice(best_move_arr)

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

    def consider_state(self, player_state, opponent_state):
        state = player_state << 9 | opponent_state
        if state in self.states:
            return self.states[state]
        elif self.game.check_is_winning(player_state):
            return 1
        else:
            return 0.5

    def get_current_combined_state(self):
        player_state, opponent_state = self.get_current_states()
        return (player_state << 9) | opponent_state

    def get_player_states_from_combined_state(self, combined_state):
        left_state = combined_state >> 9
        right_state = combined_state & 0b111111111
        player_state = (
            left_state if self.player_type == PlayerType.PLAYER else right_state
        )
        opponent_state = (
            right_state if self.player_type != PlayerType.PLAYER else left_state
        )
        return player_state, opponent_state

    def log_value(self):
        # get state
        combined_state = self.get_current_combined_state()
        player_state, opponent_state = self.get_current_states()
        current_value = self.consider_state(player_state, opponent_state)

        # log if player has won
        self.is_winning = current_value == 1

        # log current state
        if combined_state not in self.states:
            self.states[combined_state] = current_value

        self.update_prev_state(current_value)

    def log_no_win(self):
        self.update_prev_state(0)

    def update_prev_state(self, current_value):
        # Log prev state and get value
        if self.prev_state not in self.states:
            prev_player_state, prev_opponent_state = (
                self.get_player_states_from_combined_state(self.prev_state)
            )
            prev_value = self.consider_state(prev_player_state, prev_opponent_state)
            self.states[self.prev_state] = prev_value
        prev_value = self.states[self.prev_state]

        # update previous state
        new_value = prev_value + self.learning_rate * (current_value - prev_value)
        self.states[self.prev_state] = new_value

        self.average_change = calculate_running_average(
            self.average_change, abs(new_value - prev_value), self.total_moves
        )

    def reset_player(self):
        self.total_games += 1
        self.win_rate = calculate_running_average(
            self.win_rate, self.is_winning, self.total_games
        )

        self.win_record.append(self.win_rate)

        self.is_winning = False
        self.prev_state = 0

    def print_top_states(self, top_n=10):
        sorted_states = sorted(self.states.items(), key=lambda x: x[1], reverse=True)
        print(
            f"\nTop {top_n} States for {self.player_type.name} (Explore Rate: {self.exploring_rate}):"
        )
        print("-" * 70)

        for rank, (combined_state, value) in enumerate(sorted_states[:top_n], 1):

            print(f"\nRank {rank}: Value = {value:.4f}")
            print("Player (X) | Opponent (O)")
            print("-" * 25)

            self.game.print_readable_state(combined_state)
        print()


def main(args):
    game = TicTacToeGame()

    player_arr = []
    opponent_arr = []

    for player_type in PlayerType:
        if player_type == PlayerType.PLAYER:
            continue

        # build player array
        player_arr.append(
            Player(game, PlayerType.PLAYER, args.seed, args.learning_rate, 0)
        )
        player_arr.append(
            Player(
                game,
                PlayerType.PLAYER,
                args.seed,
                args.learning_rate,
                args.exploring_rate,
            )
        )

        # build opponent array
        opponent_arr.append(Player(game, player_type, args.seed, args.learning_rate, 0))
        opponent_arr.append(
            Player(
                game, player_type, args.seed, args.learning_rate, args.exploring_rate
            )
        )

    for i in range(len(player_arr)):
        player = player_arr[i]
        opponent = opponent_arr[i]
        convergence_threshold = 0.001

        print()
        print()
        print(
            f"STARTING GAME VS {opponent.player_type.name} EXPLORING RATE: {player.exploring_rate}"
        )

        while player.total_games <= args.num_rounds:
            player_move = player.play_move()
            game_over = player_move == -1
            if not game_over:
                opponent_move = opponent.play_move(player_move)
                game_over = opponent_move == -1

            if game_over:
                game.clear_board()
                player.reset_player()
                opponent.reset_player()

                if (
                    player.average_change < convergence_threshold
                    and player.total_games > 100
                ):
                    print(
                        f"Converged after {player.total_games} games (avg_change={player.average_change:.6f})"
                    )
                    break

        player.print_top_states()
        plot_results(
            [
                {
                    "record": player.win_record,
                    "label": f"player exploration={player.exploring_rate}",
                },
            ],
            f"Record Vs. {opponent.player_type.name}",
            "Game",
            "Win Rate",
        )

    print("RESULTS")
    print("-" * 60)
    print(
        f"{'Explore Rate':<12} | {'Opponent':<12} | {'Win Rate':<10} | {'Explored States':<16}"
    )
    print("-" * 60)
    for i in range(len(player_arr)):
        disp_player = player_arr[i]
        disp_opponent = opponent_arr[i]
        print(
            f"{disp_player.exploring_rate:<12.1f} | {disp_opponent.player_type.name:<12} | {disp_player.win_rate:<10.4f} | {len(disp_player.states) + 1:<10}"
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
