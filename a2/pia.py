import argparse

from util.grid_state import BellmanData, State
from util.grid_world import GridWorld, GridWorldPayload


def main(args):
    payload = GridWorldPayload(
        dimensions=args.dimensions,
        accuracy=0.95,
        reward_states=[0, 15],
        seed=42,
    )
    grid = GridWorld(payload)

    bellman_data = BellmanData(p_one=0.8, p_two=0.1, reward=1.0, discount=0.9)
    grid.create_states(bellman_data)
    grid.join_states()
    grid.initialize_actions()

    grid.perform_policy_sweep()
    grid.print_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic Tac Toe Implementation")
    parser.add_argument("--dimensions", type=int, default=4)

    args = parser.parse_args()
    main(args)
