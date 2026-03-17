from decimal import ROUND_HALF_UP, Decimal
from random import Random
from util.interfaces import AdjacentStates
from util.grid_action import GridAction


class GridState:
    def __init__(self, i, agent_data, terminal_state=False, wall_state="None"):
        self.rng = Random(agent_data.seed)

        self.index = i
        self.agent_data = agent_data
        self.terminal_state = terminal_state
        self.wall_state = wall_state
        self.epsilon = agent_data.epsilon

        self.distance_to_terminal = None
        self.neighbours = [None] * 9
        self.actions = []
        self.policy_actions = []
        self.weights = []

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
        self.weights = [0.25, 0.25, 0.25, 0.25]
