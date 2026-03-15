import argparse
from random import Random

from util.grid_world import GridWorld
from util.interfaces import AgentData, GridWorldPayload


class EpisodePair:
    def __init__(self, state, action):
        self.state = state
        self.action = action


class MCAgent:
    def __init__(self, world: GridWorld, payload: AgentData):
        self.world = world
        self.rng = Random(payload.seed)

        self.max_episode_length = payload.max_episode_length
        self.terminal_reward = payload.terminal_reward
        self.discount = payload.discount
        self.episode = []
        self.max_delta = 0

    def create_episode(self):
        # initialize first state
        self.episode = []
        state = self.rng.choice(self.world.states)
        for _ in range(self.max_episode_length):
            if state.terminal_state:
                break
            action = state.choose_action()
            next_state = action.take_action()
            reward = self.terminal_reward if next_state.terminal_state else -1
            self.episode.append(
                {"state": state, "action": action, "reward": reward, "return": 0}
            )
            state = next_state

    def update_action_values(self):
        g = 0
        for step in reversed(self.episode):
            reward = step["reward"]
            g = reward + self.discount * g
            step["return"] = g

        visited = set()
        self.max_delta = 0
        for step in self.episode:
            state = step["state"]
            action = step["action"]
            pair = (state.index, action.target.index)
            if pair in visited:
                continue

            visited.add(pair)
            g = step["return"]
            delta = action.average_value(g)
            state.adjust_weights()

            if delta > self.max_delta:
                self.max_delta = delta


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
        args.epsilon,
    )
    grid.create_states(agent_data)
    grid.join_states()
    grid.initialize_actions()

    agent = MCAgent(grid, agent_data)
    for _ in range(10000):
        agent.create_episode()
        agent.update_action_values()

    grid.print_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Implementation")
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
    parser.add_argument("--epsilon", type=int, default=0.1)

    parser.add_argument("--max_episode_length", type=int, default=1000)

    args = parser.parse_args()
    main(args)
