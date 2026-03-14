from random import Random
from util.interfaces import AdjacentStates


class GridAction:
    def __init__(
        self,
        action,
        state,
        neighbours,
        agent_data,
    ):
        self.rng = Random(agent_data.seed)
        self.action = action
        self.state = state
        self.target = self.get_target(neighbours)
        self.adjacent_states = self.get_adjacent_states(neighbours)
        self.agent_data = agent_data

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

    def take_action(self):
        results_list = [self.target, self.state]
        weights = [self.agent_data.p_one, self.agent_data.p_two]
        d = len(self.adjacent_states)
        if d > 0:
            p_three = (1 - self.agent_data.p_one - self.agent_data.p_two) / d
            for adjacent_state in self.adjacent_states:
                results_list.append(adjacent_state)
                weights.append(p_three)

        picked = self.rng.choices(results_list, weights=weights, k=1)[0]
        return picked
