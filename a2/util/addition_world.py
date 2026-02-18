from decimal import Decimal
import random
from typing import List

from util.interfaces import AdditionData
from util.error_state import ErrorState
from util.addition_state import AdditionState


class AdditionWorld:
    def __init__(self, digits: int, discount: float, accuracy: float, seed: int):
        self.rng = random.Random(seed)
        self.digits = digits
        self.accuracy = accuracy
        self.discount = discount

        digit_input = None
        print("would you like to enter custom digits?(Y/N):")
        response = input()
        if response.upper() == "Y":
            digit_input = (
                self.get_digit("digit 1"),
                self.get_digit("digit 2"),
            )

        self.digit_one, self.digit_two = self.set_digits(digit_input)
        self.answer, self.carry_answer = self.get_answer()

        self.error_state = ErrorState()
        self.states = {}
        self.initialize_states()
        self.initialize_actions()

        self.i = 0
        self.k = 0
        self.delta = 0

    def initialize_states(self):
        n = self.digits + 1
        for mask in range(1 << n):
            sum = [self.answer[i] if (mask & (1 << i)) else None for i in range(n)]
            carry = [self.carry_answer[i] if (mask & (1 << i)) else 0 for i in range(n)]
            self.states[mask] = AdditionState(
                AdditionData(
                    sum,
                    carry,
                    self.digit_one,
                    self.digit_two,
                    self.answer,
                    self.carry_answer,
                )
            )

    def initialize_actions(self):
        actions = 0
        for state in self.states.values():
            state.initialize_actions(self.states, self.error_state, self.discount)
            actions += len(state.actions)

    def get_digit(self, digit_name: str):
        is_number = False
        while not is_number:
            digit_arr = []
            print(f"enter {digit_name}:", end="")
            digit_one_str = input()
            if len(digit_one_str) < 1:
                print("you must enter a value!")
            else:
                # want the least significant digit to populate at 0 index
                # so that the i and j values make sense for the carry
                for char in digit_one_str[::-1]:
                    try:
                        num = int(char)
                        digit_arr.append(num)
                    except Exception:
                        print("you must enter only numbers!")
                        break

                if len(digit_arr) == len(digit_one_str):
                    is_number = True
                else:
                    print("please try again")
                    print()
        return digit_arr

    def set_digits(self, input):
        digit_one = []
        digit_two = []
        if input is None:
            digit_one = [self.rng.randint(0, 9) for _ in range(self.digits - 1)]
            digit_one.append(self.rng.randint(1, 9))
            digit_two = [self.rng.randint(0, 9) for _ in range(self.digits - 1)]
            digit_two.append(self.rng.randint(1, 9))
        else:
            digit_one, digit_two = input
        self.normalize_digits(digit_one, digit_two)
        return (digit_one, digit_two)

    def normalize_digits(self, digit_one: List[int], digit_two: List[int]):
        # add zeros until the digits are the same length
        while len(digit_one) < len(digit_two):
            digit_one.append(0)
        while len(digit_two) < len(digit_one):
            digit_two.append(0)
        # add an extra digit for the final carry over if needed
        digit_one.append(0)
        digit_two.append(0)

    def get_answer(self):
        carry = [0 for _ in range(self.digits + 1)]
        s = [0 for _ in range(self.digits + 1)]
        for i in range(self.digits + 1):
            sum = self.digit_one[i] + self.digit_two[i] + carry[i]
            s[i] = sum % 10
            c = sum // 10
            if c > 0:
                carry[i + 1] = c
        return s, carry

    def reset_states(self):
        for state in self.states.values():
            self.error_state.value = 0
            state.value = 0

    def perform_policy_iteration(self):
        still_pruning = True
        while still_pruning:
            self.i += 1
            self.reset_states()
            self.perform_policy_evalutation()
            still_pruning = self.perform_policy_improvement()

    def perform_value_iteration(self):
        while self.delta > self.accuracy or self.k == 0:
            self.i += 1
            self.delta = self.perform_policy_sweep()
            self.k += 1
            self.perform_policy_improvement()

    def perform_policy_sweep(self):
        delta = Decimal(0)
        for state in self.states.values():
            state.evaluate_policy()
            delta = max(delta, state.value - state.new_value)
        self.error_state.evaluate_policy()
        delta = max(delta, self.error_state.value - self.error_state.new_value)
        for state in self.states.values():
            state.value = state.new_value
        self.error_state.value = self.error_state.new_value
        return delta

    def perform_policy_evalutation(self):
        self.delta = Decimal(0)
        local_k = 0
        while self.delta > self.accuracy or local_k == 0:
            self.delta = self.perform_policy_sweep()
            local_k += 1
            self.k += 1

    def perform_policy_improvement(self):
        still_pruning = False
        for state in self.states.values():
            pruned = state.greedify()
            if pruned:
                still_pruning = True
        return still_pruning

    def produce_sum(self):
        current_state = self.states[0]
        for i in range(self.digits + 1):
            action = current_state.actions[0]
            print()
