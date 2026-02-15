from decimal import Decimal
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
        self.value = Decimal(0)
        # initialize new value function to None (k+1)
        self.new_value = Decimal(0)

        self.neighbours = [None] * 9
        self.actions = []

    # assign another state to each adjacent index (see util.interfaces AdjecentStates)
    def join_states(self, adjacent_state, state):
        self.neighbours[adjacent_state.value] = state

    # load up the actions with the target states, probabilities and states
    # adjacent to the target
    def initialize_actions(self):
        # no need to calculate actions for the terminal state
        if self.reward_state:
            return

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

        # sum all expected values of all actions
        total_value = 0
        for action in self.actions:
            total_value += action.calculate_action_value()

        self.new_value = total_value

    def record_policy(self):
        self.value = self.new_value

    # iterate through all the actions and only keep the
    # actions with the highest value
    def greedify(self):
        new_actions = []
        max_value = None
        for action in self.actions:
            if max_value is None:
                max_value = action.value
                new_actions.append(action)
            elif action.value == max_value:
                new_actions.append(action)
            elif action.value > max_value:
                new_actions = []
                max_value = action.value
                new_actions.append(action)
        self.actions = new_actions

    def print_state(self):
        print()
        print(f"State: {self.index}")
        print(f"-------------------")
        print(f"value: {self.value}")
        print("actions:")
        for action in self.actions:
            action.print_action()
            print()
