from decimal import ROUND_HALF_UP, Decimal
from util.interfaces import AdjacentStates
from util.grid_action import GridAction


def policy_evaluation_backup_summand(prob_action, prob_result, reward, discount, state):
    return prob_action * prob_result * (reward + discount * state)


class GridState:
    def __init__(self, i, agent_data, terminal_state=False, wall_state="None"):
        self.index = i
        self.agent_data = agent_data
        self.terminal_state = terminal_state
        self.wall_state = wall_state

        # initialize value to 0
        self.initial_value = (
            Decimal(0)
            if not terminal_state
            else Decimal(self.agent_data.terminal_reward)
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
            GridAction(AdjacentStates.TOP, self, self.neighbours, self.agent_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.BOTTOM, self, self.neighbours, self.agent_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.RIGHT, self, self.neighbours, self.agent_data)
        )
        self.actions.append(
            GridAction(AdjacentStates.LEFT, self, self.neighbours, self.agent_data)
        )
