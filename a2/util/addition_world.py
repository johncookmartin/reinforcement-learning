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
        # we want a state for every partially computed sum and carry array in any order
        # this results in a binary, summed or not summed across the list representation of the answer.
        # by iterating through each possible binary number from length 1 to length n, we use the zeros to
        # represent present sums and create the states accordingly.
        # We include the state of the carry array as part of the sum. Specifically we pair carry[i + 1]
        # with sum[i]. This is because carry[i + 1] is a result of the correct value in sum[i] and
        # they cannot exist apart.
        for mask in range(1 << n):
            carry_mask = mask << 1
            sum = [self.answer[i] if (mask & (1 << i)) else None for i in range(n)]
            carry = [
                (self.carry_answer[i] if (carry_mask & (1 << i)) else 0)
                for i in range(n)
            ]
            # states are stored in the dictionary by mask for ease of reference
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

    # once the states are created we initialize the actions
    def initialize_actions(self):
        actions = 0
        for state in self.states.values():
            state.initialize_actions(self.states, self.error_state, self.discount)
            actions += len(state.actions)

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
            # greedify the states and check if there has been any change
            # between the number of actions
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
        # after complete sweep, update the values of all states according to sweep
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
        print()
        print(f"converged after {self.k} sweeps and {self.i} iterations")
        print()
        print(f"adding:")
        str_digit_one = "".join(map(str, reversed(self.digit_one)))
        str_digit_two = "".join(map(str, reversed(self.digit_two)))
        print(f"  {str_digit_one}")
        print(f"+ {str_digit_two}")
        print("---------------")
        for i in range(self.digits + 1):
            action = self.rng.choice(current_state.actions)
            i = action.action_payload.i
            j = action.action_payload.j
            k = action.action_payload.k
            print(f"of possible: {len(current_state.actions)} actions")
            print(
                f"adding {self.digit_one[i]} from digit 1 index {i} to {self.digit_two[j]} from digit 2 index {j}"
            )
            if self.carry_answer[k] == 1:
                print(f"adding carry from {k}")

            next_state = action.result_state
            str_sum = "".join(map(str, reversed(next_state.sum)))
            print()
            print(f"result: {str_sum}")
            current_state = next_state

    ###################################
    ### FUNCTIONS FOR SETTING DIGITS
    ##################################

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
