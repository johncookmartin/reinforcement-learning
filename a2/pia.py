import argparse

from util.interfaces import BellmanData, GridWorldPayload
from util.grid_world import GridWorld


def main(args):
    payload = GridWorldPayload(args.dimensions, args.accuracy, args.reward_states)
    grid = GridWorld(payload)

    bellman_data = BellmanData(args.p_one, args.p_two, args.discount, args.reward)
    grid.create_states(bellman_data)
    grid.join_states()
    grid.initialize_actions()

    grid.perform_policy_iteration()
    grid.print_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Iteration Implementation")
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--p_one", type=float, default=0.8)
    parser.add_argument("--p_two", type=float, default=0.1)
    parser.add_argument("--reward", type=float, default=-1)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--accuracy", type=float, default=0.001)
    parser.add_argument("--reward_states", type=int, nargs="+", default=[0, 15])

    args = parser.parse_args()
    main(args)
