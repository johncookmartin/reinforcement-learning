import argparse

from util.addition_world import AdditionWorld


def main(args):

    addition_world = AdditionWorld(args.digits, args.discount, args.accuracy, args.seed)
    addition_world.perform_value_iteration()
    print("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Addition Implementation")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--accuracy", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
