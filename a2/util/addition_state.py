from decimal import Decimal
from util.addition_action import AdditionAction
from util.interfaces import (
    AdditionActionData,
    AdditionData,
)


class AdditionState:
    def __init__(self, payload: AdditionData):
        self.payload = payload
        self.sum = payload.sum
        self.digit_one = payload.digit_one
        self.digit_two = payload.digit_two
        self.carry = payload.carry
        self.actions = []

        self.value = 0
        self.new_value = 0

    def get_mask(self, sum=None):
        s = sum if sum is not None else self.sum
        mask = 0
        for i, v in enumerate(s):
            if v is not None:
                mask |= 1 << i
        return mask

    def initialize_actions(self, states, error_state, discount):
        # all sums have been added, no more actions needed because this is the terminal state
        if None not in self.sum:
            return

        for i in range(len(self.digit_one)):
            for j in range(len(self.digit_two)):
                for k in range(len(self.carry)):
                    # agent can go back and forth, adding any sum it wants
                    for s in range(len(self.sum)):
                        action_data = AdditionActionData(i, j, k, s)
                        action = AdditionAction(action_data, self.payload, discount)
                        # if the action doesn't have a result, the sum attempt was invalid
                        # this action will lead to the error state
                        if action.result is None:
                            action.result_state = error_state
                        else:
                            mask = self.get_mask(action.result)
                            action.result_state = states[mask]
                        self.actions.append(action)

    def evaluate_policy(self):
        # check to make sure we are not in the terminal state
        # if sum is full then we are done and value can remain 0
        if None not in self.sum:
            return

        total_value = 0
        prob = Decimal(1 / len(self.actions))
        for action in self.actions:
            total_value += action.calculate_action_value(prob)

        self.new_value = total_value

    def greedify(self):
        new_actions = []
        max_value = None
        for action in self.actions:
            if max_value is None:
                max_value = action.value
                new_actions.append(action)
            elif action.value == max_value:
                new_actions.append(action)
            elif action.value > max_value:
                new_actions = []
                max_value = action.value
                new_actions.append(action)

        # if there are a different amount of new actions that
        # original actions, it means we have pruned some actions
        # and therefore we are not in the optimal state
        pruned = len(self.actions) != len(new_actions)
        self.actions = new_actions
        return pruned
