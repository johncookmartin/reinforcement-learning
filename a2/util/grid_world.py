import random

from util.interfaces import AdjacentStates, GridWorldPayload
from util.grid_state import State


class GridWorld:
    def __init__(self, payload: GridWorldPayload):
        self.rgn = random.seed(payload.seed)
        self.dimension = payload.dimensions
        self.accuracy = payload.accuracy
        self.reward_states = payload.reward_states

        self.size = self.dimension**2

        self.states = []
        self.k = 0
        self.delta = 0

    def create_states(self, bellman_data):
        # populate the dict with states
        for i in range(self.size):
            if i not in self.reward_states:
                self.states.append(State(i, bellman_data))
            else:
                # populate reward state in proper place
                self.states.append(State(i, bellman_data, reward_state=True))

    def join_states(self):
        for i in range(self.size):

            # create index of neighbouring states
            neighbour_indexes = {
                AdjacentStates.TOP_LEFT: i - self.dimension - 1,
                AdjacentStates.TOP: i - self.dimension,
                AdjacentStates.TOP_RIGHT: i - self.dimension + 1,
                AdjacentStates.LEFT: i - 1,
                AdjacentStates.SELF: i,
                AdjacentStates.RIGHT: i + 1,
                AdjacentStates.BOTTOM_LEFT: i + self.dimension - 1,
                AdjacentStates.BOTTOM: i + self.dimension,
                AdjacentStates.BOTTOM_RIGHT: i + self.dimension + self.dimension + 1,
            }

            # iterate through states and join adjacent states
            # this will create a graph with states as nodes
            state = self.states[i]
            for adjacent_state, index in neighbour_indexes.items():
                neighbour_state = None
                if 0 <= index < len(self.states):
                    neighbour_state = self.states[index]
                state.join_states(adjacent_state, neighbour_state)

    def initialize_actions(self):
        for state in self.states:
            state.initialize_actions()

    def perform_policy_sweep(self):
        # initialize the state actions using p_one, p_two, reward and discount
        delta = 0
        for state in self.states:
            state.iterate_policy()
            delta = max(delta, state.value - state.new_value)
        for state in self.states:
            state.record_policy()
        self.k += 1
        self.delta = delta

    def perform_policy_iteration(self):
        while self.delta > self.accuracy or self.delta == 0:
            self.perform_policy_sweep()

    def print_grid(self):
        print(f"GRID {self.size} k = {self.k}")
        print("-" * 25)
        for i, state in enumerate(self.states):
            if i % self.dimension == 0:
                print()
            indicator = "R" if state.reward_state else str(state.index)
            print(f"{indicator:>2}: {state.value:.2f}", end=" ")
        print()
