from util.interfaces import AdjacentStates


class Action:
    def __init__(self, action, state, neighbours):
        self.action = action
        self.state = state
        self.target = self.get_target(neighbours)
        self.adjacent_states = self.get_adjacent_states(neighbours)

    def get_target(self, neighbours):
        state = neighbours[self.action.value]
        return state if state is not None else self.state

    def get_adjacent_states(self, neighbours):
        top = neighbours[AdjacentStates.TOP.value]
        bottom = neighbours[AdjacentStates.BOTTOM.value]
        right = neighbours[AdjacentStates.RIGHT.value]
        left = neighbours[AdjacentStates.LEFT.value]
        if self.action == AdjacentStates.TOP:
            option_a = left if left is not None else bottom
            option_b = right if right is not None else bottom
        if self.action == AdjacentStates.BOTTOM:
            option_a = left if left is not None else top
            option_b = right if right is not None else top
        if self.action == AdjacentStates.RIGHT:
            option_a = top if top is not None else left
            option_b = bottom if bottom is not None else left
        if self.action == AdjacentStates.LEFT:
            option_a = top if top is not None else right
            option_b = bottom if bottom is not None else right
        return [option_a, option_b]
