from decimal import ROUND_HALF_UP, Decimal
from util.interfaces import AdjacentStates
from util.grid_action import GridAction


def policy_evaluation_backup_summand(prob_action, prob_result, reward, discount, state):
    return prob_action * prob_result * (reward + discount * state)


class GridState:
    def __init__(self, i, bellman_data, terminal_state=False, wall_state="None"):
        self.index = i
        self.bellman_data = bellman_data
        self.terminal_state = terminal_state
        self.wall_state = wall_state

        # initialize value to 0
        self.initial_value = (
            Decimal(0)
            if not terminal_state
            else Decimal(self.bellman_data.terminal_reward)
        )
        self.value = self.initial_value
        self.new_value = 0

        self.neighbours = [None] * 9
        self.actions = []

    # assign another state to each adjacent index (see util.interfaces AdjecentStates)
    def join_states(self, adjacent_state, state):
        self.neighbours[adjacent_state.value] = state

    # load up the actions with the target states, probabilities and states
    # adjacent to the target
    def initialize_actions(self):
        # no need to calculate actions for the terminal state
        if self.terminal_state:
            return

        self.actions.append(
            GridAction(AdjacentStates.TOP, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.BOTTOM, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.RIGHT, self, self.neighbours, self.bellman_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.LEFT, self, self.neighbours, self.bellman_data)
        )

    # calculate (v)k + 1 for current state and store value in new_value
    def evaluate_policy(self):
        if self.terminal_state:
            # we are in the terminal state, no need to further iterate
            return

        # sum of expected value of all remaining actions
        total_value = 0
        prob = Decimal(1 / len(self.actions))
        for action in self.actions:
            total_value += action.calculate_action_value(prob)

        self.new_value = total_value

    # iterate through all the actions and only keep the
    # actions with the highest value
    def greedify(self):
        new_actions = []
        max_value = None
        # adding precision to remove floating point errors in comparisons
        precision = Decimal("0.00000000001")
        for action in self.actions:
            rounded_value = action.value.quantize(precision, rounding=ROUND_HALF_UP)

            if max_value is None:
                max_value = rounded_value
                new_actions.append(action)
            elif rounded_value == max_value:
                new_actions.append(action)
            elif rounded_value > max_value:
                new_actions = []
                max_value = rounded_value
                new_actions.append(action)

        # if there are a different amount of new actions that
        # original actions, it means we have pruned some actions
        # and therefore we are not in the optimal state
        pruned = len(self.actions) != len(new_actions)
        self.actions = new_actions
        return pruned

    def print_state(self):
        print()
        print(f"State: {self.index}")
        print(f"-------------------")
        print(f"value: {self.value}")
        print("actions:")
        for action in self.actions:
            action.print_action()
            print()
