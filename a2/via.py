import argparse

from util.interfaces import BellmanData, GridWorldPayload
from util.grid_world import GridWorld


def main(args):
    # creating the grid structure dynamically based on args
    payload = GridWorldPayload(args.dimensions, args.accuracy, args.reward_states)
    grid = GridWorld(payload)

    # this creates the states, joins them together in a graph, and defines the various
    # actions that can be taken per state
    bellman_data = BellmanData(args.p_one, args.p_two, args.discount, args.reward)
    grid.create_states(bellman_data)
    grid.join_states()
    grid.initialize_actions()

    # perform the sweeps and the backtracking algorithm until delta is less than or
    # equal to theta
    grid.perform_value_iteration()
    grid.print_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value Iteration Implementation")
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--p_one", type=float, default=0.8)
    parser.add_argument("--p_two", type=float, default=0.1)
    parser.add_argument("--reward", type=float, default=-1)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--accuracy", type=float, default=0.001)
    parser.add_argument("--reward_states", type=int, nargs="+", default=[0, 15])

    args = parser.parse_args()
    main(args)
