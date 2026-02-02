import random


class BanditBuilder:
    def __init__(self, num_arms, seed=None, is_binary=True):
        self.rng = random.Random(seed)

        self.bandit_arr = []
        self.is_binary = is_binary

        best_action = None
        best_value = None
        for i in range(num_arms):
            self.bandit_arr.append(self.create_arm())
            expected_value = self.get_expected_value(i)
            if best_value is None or best_value < expected_value:
                best_action = i
                best_value = expected_value

        self.optimal_action = best_action
        self.optimal_action_value = best_value

    def create_arm(self):
        value = 1 if self.is_binary else self.rng.randint(1, 10)
        prob = self.rng.random()
        return {"value": value, "prob": prob}

    def get_expected_value(self, arm):
        value = self.bandit_arr[arm]["value"]
        prob = self.bandit_arr[arm]["prob"]
        return value * prob

    def get_optimal_action(self):
        return self.optimal_action

    def pull_arm(self, arm):
        arm = self.bandit_arr[arm]
        if self.rng.random() < arm["prob"]:
            return arm["value"]
        else:
            return 0
