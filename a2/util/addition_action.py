import copy
from decimal import Decimal
from typing import List
from util.interfaces import AdditionActionData, AdditionData


class AdditionAction:
    def __init__(
        self,
        action_data: AdditionActionData,
        addition_data: AdditionData,
        carry: List[int],
        sum: List[int],
        attempted_s: List[bool],
    ):
        self.i, self.j, self.k, self.s = action_data

        self.reward = 0
        self.new_sum, self.new_carry, self.new_attempted_s = self.calculate_action(
            addition_data, carry, sum, attempted_s
        )

        self.result_state = None

    def calculate_action(self, addition_data, carry, sum, attempted_s):
        digit_one, digit_two, answer, discount = addition_data

        # this digit of the sum has already been calculated
        # give this a high penalty to prevent loops
        if attempted_s[self.s]:
            # result state is the same as origin state
            self.reward = -4
            return (sum, carry, attempted_s)

        temp_sum = digit_one[self.i] + digit_two[self.j] + carry[self.k]
        new_s = temp_sum % 10
        c = temp_sum // 10

        # attempted to place a sum in most significant
        # digit position that required a carry. This
        # shouldn't be possible and will cause an error
        # this gets a very high penalty
        if self.s >= len(digit_one) - 1 and c > 0:
            # this will cause error, we don't make a change
            # to the state
            self.reward = -5
            return (sum, carry, attempted_s)

        new_carry = copy.copy(carry)
        new_sum = copy.copy(sum)
        new_attempted_s = copy.copy(attempted_s)

        if c > 0:
            new_carry[self.s + 1] = c
        new_sum[self.s] = new_s
        new_attempted_s[self.s] = True

        if self.i == self.j == self.k:
            # added the correct digits together
            if new_s == answer[self.s]:
                # got the correct sum
                self.reward = 1
            else:
                # didn't get the correct sum
                self.reward = 0
        else:
            # didn't add the correct digits together
            if self.i == self.j or self.i == self.k or self.j == self.k:
                # added some of the correct digits together
                self.reward = -0.5
            else:
                # added none of the correct digits together
                self.reward = -1
        return (new_sum, new_carry, new_attempted_s)

    def calculate_action_value(self):
        r = Decimal((str(self.reward)))
        d = Decimal((str(self.addition_data.discount)))
        v = Decimal((str(self.result_state.value)))
        p = 1
        result = p * (r + d * v)
        return result
