import copy
from decimal import Decimal
from typing import List

from util.addition_action import AdditionAction
from util.interfaces import AdditionActionData, AdditionData


class AdditionState:
    def __init__(
        self,
        addition_data: AdditionData,
        state_sum: List[int],
        state_carry: List[int],
        attempted_s: List[bool],
        discount: float,
    ):
        self.addition_data = addition_data
        self.sum = state_sum
        self.carry = state_carry
        self.attempted_s = attempted_s
        self.discount = discount
        self.value = 0

        self.actions = []
        self.initialize_actions()

    def initialize_actions(self):
        digit_one = self.addition_data.digit_one
        digit_two = self.addition_data.digit_two
        for i in range(len(digit_one)):
            for j in range(len(digit_two)):
                for k in range(len(self.state_carry)):
                    for s in range(len(self.state_sum)):
                        action_data = AdditionActionData(i, j, k, s)
                        action = AdditionAction(
                            action_data,
                            self.addition_data,
                            self.carry,
                            self.sum,
                            self.attempted_s,
                        )
                        self.actions.append(action)

    def perform_sweep(self):
        total_value = 0
        prob = 1 / len(self.actions)
        for action in self.actions:
            total_value += prob * action.calculate_action_value()

        self.value = total_value
