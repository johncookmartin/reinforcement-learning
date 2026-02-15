from util.interfaces import AdjacentStates
from util.grid_action import Action


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
        self.new_value = 0

        self.neighbours = [None] * 9
        self.actions = []

    def join_states(self, adjacent_state, state):
        self.neighbours[adjacent_state.value] = state

    def initialize_actions(self):
        self.actions.append(Action(AdjacentStates.TOP, self, self.neighbours))
        self.actions.append(Action(AdjacentStates.BOTTOM, self, self.neighbours))
        self.actions.append(Action(AdjacentStates.RIGHT, self, self.neighbours))
        self.actions.append(Action(AdjacentStates.LEFT, self, self.neighbours))

    def state_reward_summand(self, prob, state):
        r = self.bellman_data.reward
        d = self.bellman_data.discount
        v = state.value
        result = prob * (r + d * v)
        return result

    def action_summand(self, target, adjacent_states):
        target_value = self.state_reward_summand(self.bellman_data.p_one, target)
        self_value = self.state_reward_summand(self.bellman_data.p_two, self)

        adjacent_value = 0
        d = len(adjacent_states)
        if d > 0:
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
            target = action.target
            adjacent_states = action.adjacent_states
            total_value += 0.25 * self.action_summand(target, adjacent_states)

        self.new_value = total_value

    def record_policy(self):
        self.value = self.new_value

    def print_state(self):
        print()
        print(f"State: {self.index}")
        print(f"-------------------")
        print(f"value: {self.value}")
        print("actions:")
        for action in self.actions:
            action.print_action()
            print()
