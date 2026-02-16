from decimal import Decimal
from util.interfaces import AdjacentStates


class GridAction:
    def __init__(self, action, state, neighbours, bellman_data):
        self.action = action
        self.state = state
        self.target = self.get_target(neighbours)
        self.adjacent_states = self.get_adjacent_states(neighbours)
        self.bellman_data = bellman_data
        self.value = 0

    # if target is None, this action will result in agent returning to
    # original state
    def get_target(self, neighbours):
        target_state = neighbours[self.action.value]
        return target_state if target_state is not None else self.state

    # calculate adjacent states to target state
    def get_adjacent_states(self, neighbours):
        options = []
        # if any of the adjacent states are None, re distribute the
        # probability to the target state
        if self.action == AdjacentStates.TOP:
            top_left = neighbours[AdjacentStates.TOP_LEFT.value]
            top_right = neighbours[AdjacentStates.TOP_RIGHT.value]
            if top_left is not None:
                options.append(top_left)
            if top_right is not None:
                options.append(top_right)
        elif self.action == AdjacentStates.BOTTOM:
            bottom_left = neighbours[AdjacentStates.BOTTOM_LEFT.value]
            bottom_right = neighbours[AdjacentStates.BOTTOM_RIGHT.value]
            if bottom_left is not None:
                options.append(bottom_left)
            if bottom_right is not None:
                options.append(bottom_right)
        elif self.action == AdjacentStates.RIGHT:
            top_right = neighbours[AdjacentStates.TOP_RIGHT.value]
            bottom_right = neighbours[AdjacentStates.BOTTOM_RIGHT.value]
            if top_right is not None:
                options.append(top_right)
            if bottom_right is not None:
                options.append(bottom_right)
        else:
            top_left = neighbours[AdjacentStates.TOP_LEFT.value]
            bottom_left = neighbours[AdjacentStates.BOTTOM_LEFT.value]
            if top_left is not None:
                options.append(top_left)
            if bottom_left is not None:
                options.append(bottom_left)
        while len(options) < 2:
            options.append(self.target)
        return options

    def calculate_action_value(self, prob):
        # add target and self values
        target_value = self.state_reward_summand(self.bellman_data.p_one, self.target)
        self_value = self.state_reward_summand(self.bellman_data.p_two, self.state)

        # add target adjacent state values
        adjacent_value = 0
        d = len(self.adjacent_states)
        if d > 0:
            p_three = (1 - self.bellman_data.p_one - self.bellman_data.p_two) / d
            for adjacent_state in self.adjacent_states:
                adjacent_value += self.state_reward_summand(p_three, adjacent_state)

        summation = target_value + self_value + adjacent_value
        self.value = Decimal(prob) * summation
        return self.value

    # calculate the prob of state result * reward plus discount times previous value of
    # result state
    def state_reward_summand(self, prob, state):
        r = Decimal(str(self.bellman_data.reward))
        d = Decimal(str(self.bellman_data.discount))
        v = Decimal(str(state.value))
        p = Decimal(str(prob))
        result = p * (r + d * v)
        return result

    def print_action(self):
        print(f"{self.action.name}")
        print("-" * 10)
        print(f"value: {self.value}")
        print(f"target: {self.target.index}")
        for state in self.adjacent_states:
            print(f"adjacent: {state.index}")
