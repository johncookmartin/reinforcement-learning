from decimal import Decimal


class ErrorState:
    def __init__(self):
        self.value = 0
        self.new_value = 0

    def evaluate_policy(self):
        # no matter what agent does in this state, the answer is wrong and reward is -1
        r = Decimal(str("-1"))
        d = Decimal(str(".9"))
        v = Decimal(str(self.value))
        p = Decimal(str("1"))
        result = p * (r + d * v)
        self.new_value = result
