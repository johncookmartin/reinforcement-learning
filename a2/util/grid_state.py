from util.interfaces import AdjacentStates
from util.grid_action import Action


def policy_evaluation_backup_summand(prob_action, prob_result, reward, discount, state):
    return prob_action * prob_result * (reward + discount * state)


class State:
    def __init__(self, i, bellman_data, reward_state=False):
        self.index = i
        self.bellman_data = bellman_data
        self.reward_state = reward_state

        # initialize value function to 0
        self.value = 0
        # initialize new value function to None (k+1)
        self.new_value = 0

        self.neighbours = [None] * 9
        self.actions = []

    def join_states(self, adjacent_state, state):
        self.neighbours[adjacent_state.value] = state

    def initialize_actions(self):
        self.actions.append(
            Action(AdjacentStates.TOP, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            Action(AdjacentStates.BOTTOM, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            Action(AdjacentStates.RIGHT, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            Action(AdjacentStates.LEFT, self, self.neighbours, self.bellman_data)
        )

    def iterate_policy(self):
        if self.reward_state:
            # we are in the terminal state, no need to further iterate
            return

        total_value = 0
        for action in self.actions:
            total_value += action.calculate_action_value()

        self.new_value = total_value

    def record_policy(self):
        self.value = self.new_value

    def print_state(self):
        print()
        print(f"State: {self.index}")
        print(f"-------------------")
        print(f"value: {self.value}")
        print("actions:")
        for action in self.actions:
            action.print_action()
            print()
