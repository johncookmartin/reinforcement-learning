import random


class BanditBuilder:
    def __init__(self, num_arms, seed=None, is_binary=True):
        self.rng = random.Random(seed)

        self.is_binary = is_binary
        self.bandit_arr = [self.create_arm(i) for i in range(num_arms)]

        sorted_indices = sorted(
            range(num_arms), key=lambda i: self.get_expected_value(i), reverse=True
        )

        self.sorted_values = []
        for rank, arm_index in enumerate(sorted_indices):
            self.bandit_arr[arm_index]["rank"] = rank
            self.sorted_values.append(self.get_expected_value(arm_index))

        self.optimal_action = sorted_indices[0]
        self.optimal_action_value = self.get_expected_value(self.optimal_action)

    def create_arm(self, i):
        value = 1 if self.is_binary else self.rng.randint(1, 10)
        prob = self.rng.random()
        return {"value": value, "prob": prob, "index": i, "rank": None}

    def get_expected_value(self, arm):
        value = self.bandit_arr[arm]["value"]
        prob = self.bandit_arr[arm]["prob"]
        return value * prob

    def get_optimal_action(self):
        return self.optimal_action

    def pull_arm(self, arm):
        arm = self.bandit_arr[arm]
        if self.rng.random() < arm["prob"]:
            return arm["value"], arm["rank"]
        else:
            return 0, arm["rank"]
