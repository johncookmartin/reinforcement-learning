from util.functions import calculate_running_average


class DataCalculator:
    def __init__(self, num_arms, num_rounds):
        self.num_arms = num_arms
        self.num_rounds = num_rounds

        self.total_optimal_value_record = []
        self.total_optimal_pull_record = [[] for _ in range(num_arms)]
        self.average_pull_values = []

        self.base = 0
        self.optimal_value = 0

    def start_trial(self, i, optimal_value):
        self.base = self.num_rounds * i
        self.optimal_value = optimal_value

    def update_value(self, new_value, turn):
        normalized_value = new_value / self.optimal_value
        if len(self.total_optimal_value_record) <= turn:
            self.total_optimal_value_record.append(normalized_value)
        else:
            self.total_optimal_value_record[turn] = calculate_running_average(
                self.total_optimal_value_record[turn],
                normalized_value,
                self.base + turn + 1,
            )

    def update_pull_record(self, new_pulls, turn):
        for i in range(len(new_pulls)):
            normalized_pull = new_pulls[i] / (turn + 1)
            if len(self.total_optimal_pull_record[i]) <= turn:
                self.total_optimal_pull_record[i].append(normalized_pull)
            else:
                self.total_optimal_pull_record[i][turn] = calculate_running_average(
                    self.total_optimal_pull_record[i][turn],
                    normalized_pull,
                    self.base + turn + 1,
                )

    def update_average_pull_values(self, pull_values, trial):
        for i in range(len(pull_values)):
            if len(self.average_pull_values) <= i:
                self.average_pull_values.append(pull_values[i])
            else:
                self.average_pull_values[i] = calculate_running_average(
                    self.average_pull_values[i], pull_values[i], trial + 1
                )
