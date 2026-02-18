import copy
from decimal import Decimal
from util.interfaces import AdditionActionData, AdditionData


class AdditionAction:
    def __init__(
        self,
        action_payload: AdditionActionData,
        addition_payload: AdditionData,
        discount: float,
    ):
        self.action_payload = action_payload
        self.addition_payload = addition_payload
        self.discount = discount

        self.reward = self.calculate_reward()
        self.result = None

        self.result_state = None
        self.value = 0

    def calculate_reward(self):
        i, j, k, s = self.action_payload
        total = (
            self.addition_payload.digit_one[i]
            + self.addition_payload.digit_two[j]
            + self.addition_payload.carry[k]
        )
        sum = total % 10

        if self.addition_payload.answer[s] != sum:
            return -1

        reward = -0.5
        if not self.in_order(s):
            reward -= 0.5

        if k == i == j == s:
            reward += 0.5

        new_result = copy.copy(self.addition_payload.sum)
        new_result[s] = sum
        self.result = new_result
        return reward

    def in_order(self, s):
        for i in range(len(self.addition_payload.sum)):
            value = self.addition_payload.sum[i]
            if value is not None and i < s:
                continue
            else:
                return False
        return True

    def calculate_action_value(self, prob):
        r = Decimal(str(self.reward))
        d = Decimal(str(self.discount))
        v = Decimal(str(self.result_state.value))
        p = Decimal(str(prob))
        result = p * (r + d * v)
        self.value = result
        return result
