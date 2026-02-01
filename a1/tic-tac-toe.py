# John's Implementation

import argparse
from util.tic_tac_toe_game import OpponentType, TicTacToeGame, TicTacToeOpponent


def main(args):
    game = TicTacToeGame()
    opponent = TicTacToeOpponent(OpponentType.RANDOM_ROW, game, debug=args.debug)
    move = 0
    game.play_move(move)
    print(f"{game.get_full_board():09b}")
    print(f"{opponent.choose_diagonal_move(move):09b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic Tac Toe Implementation")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
