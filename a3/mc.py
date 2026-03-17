from decimal import Decimal
from random import Random

from util.args import setup_grid, get_parser
from util.grid_agent import GridAgent


class MCAgent(GridAgent):
    def average_action_value(self, action, new_value):
        action.visits += 1
        old_value = action.value
        action.value = action.value + (Decimal(1) / action.visits) * (
            Decimal(new_value) - action.value
        )
        return abs(action.value - old_value)

    def create_episode(self):
        # initialize first state
        self.episode = []
        self.num_of_episodes += 1
        visited = set()
        state = self.rng.choice(self.world.states)
        for _ in range(self.max_episode_length):
            if state.terminal_state:
                break
            action = self.choose_action(state)

            # log first visits
            pair = (state.index, action.action)
            first_visit = pair not in visited
            visited.add(pair)

            # take action and log
            next_state = self.take_action(action)
            reward = self.terminal_reward if next_state.terminal_state else -1
            self.episode.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "first_visit": first_visit,
                }
            )
            self.time_steps += 1
            state = next_state

    def update_action_values(self):
        g = 0
        episode_max_delta = Decimal(0)
        for step in reversed(self.episode):
            reward = step["reward"]
            g = reward + self.discount * g
            if step["first_visit"]:
                delta = self.average_action_value(step["action"], g)
                self.adjust_weights(step["state"])

                if delta > episode_max_delta:
                    episode_max_delta = delta

        self.avg_max_delta += (
            episode_max_delta - self.avg_max_delta
        ) / self.num_of_episodes


def main(args):
    grid, agent_data = setup_grid(args)

    agent = MCAgent(grid, agent_data)

    agent.start_timer()
    while agent.stable < args.min_episodes:
        agent.create_episode()
        agent.update_action_values()
        agent.compare_policy()
    agent.stop_timer()

    agent.print_results()


if __name__ == "__main__":
    args = get_parser("Monte Carlo Implementation").parse_args()
    main(args)
