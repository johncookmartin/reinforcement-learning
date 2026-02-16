from decimal import Decimal
import random

from util.interfaces import AdjacentStates, GridWorldPayload
from util.grid_state import GridState


class GridWorld:
    def __init__(self, payload: GridWorldPayload):
        self.dimension = payload.dimensions
        self.accuracy = payload.accuracy
        self.reward_states = payload.reward_states

        self.size = self.dimension**2

        self.states = []
        self.k = 0
        self.i = 0
        self.delta = 0

    def create_states(self, bellman_data):
        # populate the dict with states
        for i in range(self.size):
            if i not in self.reward_states:
                self.states.append(GridState(i, bellman_data))
            else:
                # populate reward state in proper place
                self.states.append(GridState(i, bellman_data, reward_state=True))

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
                AdjacentStates.BOTTOM_RIGHT: i + self.dimension + 1,
            }

            # iterate through states and join adjacent states
            # this will create a graph with states as nodes
            state = self.states[i]
            # check to see if state is on the left or right side (or neither)
            is_left = state.index % self.dimension == 0
            is_right = state.index % self.dimension == self.dimension - 1
            for adjacent_state, index in neighbour_indexes.items():
                # if state is on the left edge, there are no states to the left
                if "LEFT" in adjacent_state.name and is_left:
                    state.join_states(adjacent_state, None)
                # if state is on the right edge, there are no states to the right
                elif "RIGHT" in adjacent_state.name and is_right:
                    state.join_states(adjacent_state, None)
                # if the index is out of bounds of the state array, the current state
                # must be on the top or bottom of the grid
                elif 0 <= index < len(self.states):
                    state.join_states(adjacent_state, self.states[index])
                else:
                    state.join_states(adjacent_state, None)

    def initialize_actions(self):
        for state in self.states:
            state.initialize_actions()

    def perform_policy_iteration(self):
        still_pruning = True
        while still_pruning:
            self.i += 1
            self.reset_states()
            self.perform_policy_evaluation()
            still_pruning = self.perform_policy_improvement()

    def perform_value_iteration(self):
        while self.delta > self.accuracy or self.k == 0:
            self.i += 1
            self.delta = self.perform_policy_sweep()
            self.k += 1
            self.perform_policy_improvement()

    # print the grid with values and policies
    def print_grid(self):
        print(f"GRID {self.size}      k = {self.k}      i = {self.i}")
        print("-" * 25)
        print()
        print("VALUES:")
        for i, state in enumerate(self.states):
            if i % self.dimension == 0 and i > 0:
                print()
            indicator = "**" if state.reward_state else str(state.index)
            print(f"{indicator:>2}: {state.value:.2f}", end=" ")
        print()
        print("ACTIONS")
        for i, state in enumerate(self.states):
            if i % self.dimension == 0 and i > 0:
                print()

            if not state.reward_state and state.actions:
                actions_str = "".join(
                    [action.action.name[0] for action in state.actions]
                )
                print(f"[{actions_str:4}]", end=" ")
            else:
                print("[TERM]", end=" ")
        print()

    # reset state values so that we can perform another loop of the policy iteration
    def reset_states(self):
        for state in self.states:
            state.value = 0

    # make one policy sweep through all states and record delta
    def perform_policy_sweep(self):
        delta = 0
        for state in self.states:
            # calculate the new values
            state.evaluate_policy()
            delta = max(delta, state.value - state.new_value)
        for state in self.states:
            # now we update the value of states with new values
            # must do this seperately so that we still have the old
            # values when doing the calculations
            state.value = state.new_value
        return delta

    # find stable policy as part of policy iteration
    def perform_policy_evaluation(self):
        # initialize the state actions using p_one, p_two, reward and discount
        self.delta = Decimal(0)
        local_k = 0
        self.k = 0
        while self.delta > self.accuracy or local_k == 0:
            self.delta = self.perform_policy_sweep()
            local_k += 1
            self.k += 1

    # use greedy algorithm to choose optimal actions based on policy
    # if no actions are pruned then policy is stable
    def perform_policy_improvement(self):
        still_pruning = False
        for state in self.states:
            pruned = state.greedify()
            if pruned:
                still_pruning = True
        return still_pruning
