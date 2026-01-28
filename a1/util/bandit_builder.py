import random


class BanditBuilder:
    def __init__(self, num_bandits, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()

        self.bandit_arr = []
        self.optimal_action = None
        self.optimal_action_value = None
        for i in range(num_bandits):
            self.bandit_arr.append(self.create_arm())
            expected_value = self.get_expected_value(i)
            if (
                self.optimal_action_value is None
                or self.optimal_action_value < expected_value
            ):
                self.optimal_action = i
                self.optimal_action_value = expected_value

    def create_arm(self):
        value = self.rng.randint(1, 10)
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
