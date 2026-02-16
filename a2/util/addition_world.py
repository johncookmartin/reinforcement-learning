import random
from typing import List

from util.addition_state import AdditionState
from util.interfaces import AdditionData, DigitData


class AdditionWorld:
    def __init__(self, digits: int, discount: float, seed: int, digit_data: DigitData):
        self.rng = random.Random(seed)
        self.digits = digits

        digit_one, digit_two = self.set_digits(digit_data)
        add_state_input = AdditionData(
            digit_one=digit_one,
            digit_two=digit_two,
            answer=self.get_answer(digit_one, digit_two),
        )
        self.root_state = AdditionState(
            add_state_input,
            # adding extra zero to the carry and sum in case
            # final carry is needed
            sum=[0 for _ in range(digits + 1)],
            carry=[0 for _ in range(digits + 1)],
            attempted_s=[False for _ in range(digits + 1)],
            discount=discount,
        )

    def normalize_digits(self, digit_one: List[int], digit_two: List[int]):
        # add zeros until the digits are the same length
        while len(digit_one) < len(digit_two):
            digit_one.append(0)
        while len(digit_two) < len(digit_one):
            digit_two.append(0)
        # add an extra digit for the final carry over if needed
        digit_one.append(0)
        digit_two.append(0)

    def set_digits(self, input: DigitData = None) -> tuple[List[int], List[int]]:
        digit_one = []
        digit_two = []
        if input is None:
            digit_one = [self.rng.randint(0, 9) for _ in range(self.digits - 1)]
            digit_one.append(self.rng.randint(1, 9))
            digit_two = [self.rng.randint(0, 9) for _ in range(self.digits - 1)]
            digit_two.append(self.rng.randint(1, 9))
        else:
            digit_one = input.digit_one
            digit_two = input.digit_two
        self.normalize_digits(digit_one, digit_two)
        return (digit_one, digit_two)

    def get_answer(self, digit_one: List[int], digit_two: List[int]) -> List[int]:
        digits = len(digit_one)
        carry = [0 for _ in range(digits)]
        s = [0 for _ in range(digits)]
        for i in range(digits):
            sum = digit_one[i] + digit_two[i] + carry[i]
            s[i] = sum % 10
            c = sum // 10
            if c > 0:
                carry[i + 1] = c
        return s
