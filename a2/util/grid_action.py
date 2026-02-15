from util.interfaces import AdjacentStates


class Action:
    def __init__(self, action, state, neighbours, bellman_data):
        self.action = action
        self.state = state
        self.target = self.get_target(neighbours)
        self.adjacent_states = self.get_adjacent_states(neighbours)
        self.bellman_data = bellman_data

    def get_target(self, neighbours):
        target_state = neighbours[self.action.value]
        return target_state if target_state is not None else self.state

    def get_adjacent_states(self, neighbours):
        options = []
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

    def calculate_action_value(self):
        target_value = self.state_reward_summand(self.bellman_data.p_one, self.target)
        self_value = self.state_reward_summand(self.bellman_data.p_two, self.state)

        adjacent_value = 0
        d = len(self.adjacent_states)
        if d > 0:
            p_three = (1 - self.bellman_data.p_one - self.bellman_data.p_two) / d
            for adjacent_state in self.adjacent_states:
                adjacent_value += self.state_reward_summand(p_three, adjacent_state)

        summation = target_value + self_value + adjacent_value
        result = 0.25 * summation
        return result

    def state_reward_summand(self, prob, state):
        r = self.bellman_data.reward
        d = self.bellman_data.discount
        v = state.value
        result = prob * (r + d * v)
        return result

    def print_action(self):
        print(f"{self.action.name}")
        print("-" * 10)
        print(f"target: {self.target.index}")
        for state in self.adjacent_states:
            print(f"adjacent: {state.index}")
