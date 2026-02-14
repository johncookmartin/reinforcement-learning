from enum import Enum
from typing import NamedTuple


class AdjacentStates(Enum):
    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    LEFT = 3
    SELF = 4
    RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM = 7
    BOTTOM_RIGHT = 8


class BellmanData(NamedTuple):
    p_one: float
    p_two: float
    discount: float
    reward: float


def policy_evaluation_backup_summand(prob_action, prob_result, reward, discount, state):
    return prob_action * prob_result * (reward + discount * state)


class State:
    def __init__(self, i, bellman_data, reward_state=False):
        self.index = i
        self.bellman_data = bellman_data
        self.reward_state = reward_state

        # initialize value function to 0
        self.value = 0
        # initialize new value function to None (k+1)
        self.new_value = None

        self.neighbours = [None] * 9
        self.actions = {
            AdjacentStates.TOP: {},
            AdjacentStates.BOTTOM: {},
            AdjacentStates.RIGHT: {},
            AdjacentStates.LEFT: {},
        }

    def join_states(self, adjacent_state, state):
        self.neighbours[adjacent_state.value] = state

    def initialize_actions(self):
        for action in self.actions:
            state = self.neighbours[action.value]
            self.actions[action]["target"] = state if state is not None else self
            self.actions[action]["adjacent_states"] = self.get_adjacent_states(action)

    def get_adjacent_states(self, action):
        top = self.neighbours[AdjacentStates.TOP.value]
        bottom = self.neighbours[AdjacentStates.BOTTOM.value]
        right = self.neighbours[AdjacentStates.RIGHT.value]
        left = self.neighbours[AdjacentStates.LEFT.value]
        if action == AdjacentStates.TOP:
            option_a = left if left is not None else bottom
            option_b = right if right is not None else bottom
        if action == AdjacentStates.BOTTOM:
            option_a = left if left is not None else top
            option_b = right if right is not None else top
        if action == AdjacentStates.RIGHT:
            option_a = top if top is not None else left
            option_b = bottom if bottom is not None else left
        if action == AdjacentStates.LEFT:
            option_a = top if top is not None else right
            option_b = bottom if bottom is not None else right
        return [option_a, option_b]

    def state_reward_summand(self, prob, state):
        return prob * (
            self.bellman_data.reward + self.bellman_data.discount * state.value
        )

    def action_summand(self, target, adjacent_states):
        target_value = self.state_reward_summand(self.bellman_data.p_one, target)
        self_value = self.state_reward_summand(self.bellman_data.p_two, self)

        adjacent_value = None
        d = len(adjacent_states)
        p_three = (1 - self.bellman_data.p_one - self.bellman_data.p_two) / d
        for adjacent_state in adjacent_states:
            adjacent_value += self.state_reward_summand(p_three, adjacent_state)

        return target_value + self_value + adjacent_value

    def iterate_policy(self):
        if self.reward_state:
            # we are in the terminal state, no need to further iterate
            return

        total_value = 0
        for action in self.actions:
            target, adjacent_states = self.actions[action]
            total_value += self.action_summand(target, adjacent_states)

        self.new_value = total_value

    def record_policy(self):
        self.value = self.new_value
