from decimal import Decimal

from util.args import setup_grid, get_parser
from util.grid_agent import GridAgent


class SarsaAgent(GridAgent):
    def create_episode(self):
        # initialize first state
        self.episode = []
        self.num_of_episodes += 1
        state = self.rng.choice(self.world.states)
        if state.terminal_state:
            return
        action = self.choose_action(state)
        self.traverse_states(state, action)

    def traverse_states(self, state, action):
        not_terminal = True
        loops = 0
        while not_terminal and loops < self.max_episode_length:
            loops += 1
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
                    reward = -1
                    action_prime = self.choose_action(state_prime)
                    td_error = (
                        reward
                        + Decimal(self.discount) * action_prime.value
                        - action.value
                    )
                action.value = action.value + Decimal(self.alpha) * td_error
                self.adjust_weights(state)

                state = state_prime
                action = action_prime


def main(args):
    grid, agent_data = setup_grid(args)

    agent = SarsaAgent(grid, agent_data)

    agent.start_timer()
    while agent.stable < args.min_episodes:
        agent.create_episode()
        agent.compare_policy()
    agent.stop_timer()

    agent.print_results()


if __name__ == "__main__":
    args = get_parser("SARSA Implementation").parse_args()
    main(args)
