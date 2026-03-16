import argparse

from util.grid_world import GridWorld
from util.interfaces import AgentData, GridWorldPayload


def setup_grid(args):
    payload = GridWorldPayload(
        args.dimensions,
        args.accuracy,
        args.terminal_states,
        args.wall_column,
        args.wall_row,
        args.doors,
    )
    grid = GridWorld(payload)

    agent_data = AgentData(
        args.p_one,
        args.p_two,
        args.discount,
        args.reward,
        args.terminal_reward,
        args.seed,
        args.max_episode_length,
        args.epsilon,
        args.alpha,
    )
    grid.create_states(agent_data)
    grid.join_states()
    grid.initialize_actions()

    return grid, agent_data


def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dimensions", type=int, default=11)
    parser.add_argument("--p_one", type=float, default=0.8)
    parser.add_argument("--p_two", type=float, default=0.1)
    parser.add_argument("--reward", type=float, default=-1)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--accuracy", type=float, default=0.001)
    parser.add_argument("--terminal_states", type=int, nargs="+", default=[10])
    parser.add_argument("--terminal_reward", type=float, default=500)
    parser.add_argument("--wall_column", type=int, default=5)
    parser.add_argument("--wall_row", type=int, default=55)
    parser.add_argument("--doors", type=int, nargs="+", default=[27, 57, 63, 93])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_episode_length", type=int, default=1000)
    return parser
