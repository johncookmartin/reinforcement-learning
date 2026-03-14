import argparse
from random import Random

from a3.util.grid_world import GridWorld
from a3.util.interfaces import AgentData, AgentPayload, BellmanData, GridWorldPayload


class EpisodePair:
    def __init__(self, state, action):
        self.state = state
        self.action = action


class MCAgent:
    def __init__(self, world: GridWorld, payload):
        self.world = world
        self.rng = Random(payload.seed)

        self.max_episode_length = payload.max_episode_length
        self.episode = []
        pass

    def create_episode(self):
        for i in self.max_episode_length:
            if i == 0:
                state = self.rng.choice(self.world.states)
                action = self.rng.choice(state)


def main(args):
    # creating the grid structure dynamically based on args
    payload = GridWorldPayload(
        args.dimensions,
        args.accuracy,
        args.terminal_states,
        args.wall_column,
        args.wall_row,
        args.doors,
    )
    grid = GridWorld(payload)

    # this creates the states, joins them together in a graph, and defines the various
    # actions that can be taken per state
    agent_data = AgentData(
        args.p_one,
        args.p_two,
        args.discount,
        args.reward,
        args.terminal_reward,
        args.seed,
        args.max_episode_length,
    )
    grid.create_states(agent_data)
    grid.join_states()
    grid.initialize_actions()

    agent_payload = AgentPayload(args.max_episode_length)
    agent = MCAgent(grid, agent_payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Implementation")
    parser.add_argument("--dimensions", type=int, default=11)
    parser.add_argument("--p_one", type=float, default=0.8)
    parser.add_argument("--p_two", type=float, default=0.1)
    parser.add_argument("--reward", type=float, default=-1)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--accuracy", type=float, default=0.001)
    parser.add_argument("--terminal_states", type=int, nargs="+", default=[10])
    parser.add_argument("--terminal_reward", type=float, default=500)
    parser.add_argument("--wall_column", type=int, default=5)
    parser.add_argument("--wall_row", type=int, default=55)
    parser.add_argument("--doors", type=int, nargs="+", default=[27, 57, 63, 93])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_episode_length", type=int, default=300)

    args = parser.parse_args()
    main(args)
