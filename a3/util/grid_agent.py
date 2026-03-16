from decimal import ROUND_HALF_UP, Decimal
from random import Random

from util.grid_world import GridWorld
from util.interfaces import AgentData


class GridAgent:
    def __init__(self, world: GridWorld, payload: AgentData):
        self.world = world
        self.rng = Random(payload.seed)

        self.max_episode_length = payload.max_episode_length
        self.terminal_reward = payload.terminal_reward
        self.discount = payload.discount
        self.p_one = payload.p_one
        self.p_two = payload.p_two
        self.epsilon = payload.epsilon
        self.alpha = payload.alpha

        self.episode = []
        self.max_delta = 0

    def choose_action(self, state):
        return self.rng.choices(state.actions, state.weights)[0]

    def take_action(self, action):
        results_list = [action.target, action.state]
        weights = [self.p_one, self.p_two]
        d = len(action.adjacent_states)
        if d > 0:
            p_three = (1 - self.p_one - self.p_two) / d
            for adjacent_state in action.adjacent_states:
                results_list.append(adjacent_state)
                weights.append(p_three)

        return self.rng.choices(results_list, weights=weights, k=1)[0]

    def adjust_weights(self, state):
        max_value = None
        optimal_actions = 0
        precision = Decimal("0.00000000001")
        for action in state.actions:
            rounded_value = action.value.quantize(precision, rounding=ROUND_HALF_UP)

            if max_value is None:
                max_value = rounded_value
                optimal_actions = 1
            elif rounded_value == max_value:
                optimal_actions += 1
            elif rounded_value > max_value:
                max_value = rounded_value
                optimal_actions = 1

        state.policy_actions = []
        for i in range(len(state.actions)):
            action = state.actions[i]

            if action.value.quantize(precision, rounding=ROUND_HALF_UP) == max_value:
                state.policy_actions.append(action)
                state.weights[i] = (
                    1 - self.epsilon + self.epsilon / (len(state.actions))
                )
            else:
                state.weights[i] = self.epsilon / len(state.actions)

    def create_episode(self):
        pass
