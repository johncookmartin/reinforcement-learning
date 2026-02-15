from util.interfaces import AdjacentStates


class Action:
    def __init__(self, action, state, neighbours):
        self.action = action
        self.state = state
        self.target = neighbours[self.action.value]
        self.adjacent_states = self.get_adjacent_states(neighbours)

    def get_adjacent_states(self, neighbours):
        if self.action == AdjacentStates.TOP:
            top_left = neighbours[AdjacentStates.TOP_LEFT.value]
            top_right = neighbours[AdjacentStates.TOP_RIGHT.value]
            return [top_left, top_right]
        elif self.action == AdjacentStates.BOTTOM:
            bottom_left = neighbours[AdjacentStates.BOTTOM_LEFT.value]
            bottom_right = neighbours[AdjacentStates.BOTTOM_RIGHT.value]
            return [bottom_left, bottom_right]
        elif self.action == AdjacentStates.RIGHT:
            top_right = neighbours[AdjacentStates.TOP_RIGHT.value]
            bottom_right = neighbours[AdjacentStates.BOTTOM_RIGHT.value]
            return [top_right, bottom_right]
        else:
            top_left = neighbours[AdjacentStates.TOP_LEFT.value]
            bottom_left = neighbours[AdjacentStates.BOTTOM_LEFT.value]
            return [top_left, bottom_left]

    def print_action(self):
        print(f"{self.action.name}")
        print("-" * 10)
        print(f"target: {self.target.index}")
        for state in self.adjacent_states:
            print(f"adjacent: {state.index}")
