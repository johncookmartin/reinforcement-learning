from decimal import Decimal

from util.args import setup_grid, get_parser
from util.grid_agent import GridAgent


class SarsaAgent(GridAgent):
    def create_episode(self):
        # initialize first state
        self.episode = []
        self.num_of_episodes += 1
        state = self.choose_init_state()
        action = self.choose_action(state)
        self.traverse_states(state, action)

    def traverse_states(self, state, action):
        not_terminal = True
        episode_max_delta = Decimal(0)
        while not_terminal:
            if state.terminal_state:
                not_terminal = False
            else:
                self.time_steps += 1
                state_prime = self.take_action(action)
                if state_prime.terminal_state:
                    reward = self.terminal_reward
                    action_prime = 0
                    td_error = reward - action.value
                else:
                    reward = self.reward
                    action_prime = self.choose_action(state_prime)
                    td_error = (
                        reward
                        + Decimal(self.discount) * action_prime.value
                        - action.value
                    )
                delta = self.update_action_value(action, td_error)
                if delta > episode_max_delta:
                    episode_max_delta = delta
                self.adjust_weights(state)

                state = state_prime
                action = action_prime

        self.avg_max_delta += (
            episode_max_delta - self.avg_max_delta
        ) / self.num_of_episodes

    def update_action_value(self, action, td_error):
        action.visits += 1
        old_action_value = action.value
        action.value = action.value + Decimal(self.alpha) * td_error
        return abs(action.value - old_action_value)


def main(args):
    grid, agent_data = setup_grid(args)

    agent = SarsaAgent(grid, agent_data)

    agent.start_timer()
    while agent.num_of_episodes < args.max_episodes:
        agent.create_episode()
        agent.compare_policy()
    agent.stop_timer()

    agent.print_results()


if __name__ == "__main__":
    args = get_parser("SARSA Implementation").parse_args()
    main(args)
