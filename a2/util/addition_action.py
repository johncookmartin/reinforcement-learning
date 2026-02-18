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

        self.result = None
        self.reward = self.calculate_reward()

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

        # if the sum is incorrect, all subsequent actions will be
        # incorrect. agent gets -2 reward and goes to error state
        if self.addition_payload.answer[s] != sum:
            return -1

        # initialize reward to zero so we can stack various rewards
        # max penalty is - 2 but if agent recieves no penalty reward is 0
        reward = 0
        # this penalizes the agent for calculating the same sum multiple times
        if self.addition_payload.sum[s] is not None:
            reward -= 0.5

        # this penalizes the agent for adding a sum in the wrong order
        if not self.in_order(s):
            reward -= 0.5

        # this rewards the agent for adding the correct sums together
        if k == i == j == s:
            reward += 1

        new_result = copy.copy(self.addition_payload.sum)
        new_result[s] = sum
        self.result = new_result
        return reward

    def in_order(self, s):
        for i in range(len(self.addition_payload.sum)):
            if self.addition_payload.sum[i] is None:
                break
        return i == s

    def calculate_action_value(self, prob):
        r = Decimal(str(self.reward))
        d = Decimal(str(self.discount))
        v = Decimal(str(self.result_state.value))
        p = Decimal(str(prob))
        result = p * (r + d * v)
        self.value = result
        return result
