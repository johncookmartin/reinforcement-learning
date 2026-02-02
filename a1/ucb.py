# John's Implementation

import math
import argparse
import random
from util.bandit_builder import BanditBuilder
from util.functions import calculate_running_average, cs_log, plot_results


class UCBBanditPuller:
    def __init__(self, num_actions, confidence_rate=10):
        self.actions = [
            {"observed_reward": 0, "times_pulled": 0} for _ in range(num_actions)
        ]
        self.total_pulls = 0
        self.confidence_rate = confidence_rate
        self.average_reward = 0

        self.record = []
        self.pull_record = [0 for _ in range(num_actions)]

    def choose_action(self):
        best_action = 0
        best_potential_value = 0
        for i in range(len(self.actions)):
            potential_value = self.calculate_potential_value(i)
            if potential_value >= best_potential_value:
                best_potential_value = potential_value
                best_action = i
        return best_action

    def calculate_potential_value(self, action):
        observed_reward = self.actions[action]["observed_reward"]
        times_pulled = self.actions[action]["times_pulled"]
        potential = (
            math.sqrt(cs_log(self.total_pulls) / max(times_pulled, 1))
            * self.confidence_rate
        )
        return observed_reward + potential

    def log_action(self, action, reward, rank):
        # calculate average reward for specific action
        observed_reward = self.actions[action]["observed_reward"]
        self.actions[action]["observed_reward"] = calculate_running_average(
            observed_reward, reward, self.total_pulls
        )
        self.actions[action]["times_pulled"] += 1

        # calculate total average reward
        self.average_reward = calculate_running_average(
            self.average_reward, reward, self.total_pulls
        )
        self.record.append(self.average_reward)

        # add to pulls taken
        self.total_pulls += 1
        self.pull_record[rank] += 1


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


def main(args):

    data_calculator = DataCalculator(args.num_arms, args.num_rounds)

    # want to control the seed but also have a different seed for each trial
    rng = random.Random(args.seed)
    init_seed = rng.randint(1, 100)

    for i in range(args.num_trials):
        print()
        print()
        print(f"TRIAL {i+1}")

        bandit = BanditBuilder(args.num_arms, init_seed + i, True)
        puller = UCBBanditPuller(args.num_arms, args.confidence_rate)

        optimal_value = bandit.get_expected_value(bandit.get_optimal_action())
        optimal_record = [optimal_value for _ in range(1, args.num_rounds + 1)]

        data_calculator.start_trial(i, optimal_value)
        data_calculator.update_average_pull_values(bandit.sorted_values, i)

        for j in range(args.num_rounds):
            picked_arm = puller.choose_action()
            value, rank = bandit.pull_arm(picked_arm)
            puller.log_action(picked_arm, value, rank)

            data_calculator.update_value(puller.record[j], j)
            data_calculator.update_pull_record(puller.pull_record, j)

            if (j + 1) % 100 == 0:
                print(f"optimal arm pulled {puller.pull_record[0]} times")
                print(f"average reward at round {j}: {puller.average_reward}")

        print(f"Optimal Expected Value: {optimal_value}")

        if args.show_plot:
            plot_results(
                [
                    {"record": optimal_record, "label": "optimal"},
                    {"record": puller.record, "label": "puller record"},
                ]
            )
        if args.show_plot:
            plot_results([{"record": puller.pull_record, "label": "optimal"}])

    plot_results(
        [
            {"record": data_calculator.total_optimal_value_record, "label": "value"},
            {
                "record": [
                    1 for _ in range(len(data_calculator.total_optimal_value_record))
                ],
                "label": "Optimal",
            },
        ],
        "Normalized Optimal Value",
        "Round",
        "Optimal Value as Percentage",
    )

    plot_results(
        [
            {
                "record": data_calculator.total_optimal_pull_record[i],
                "label": f"choice {i}, average value {data_calculator.average_pull_values[i]}",
            }
            for i in range(len(data_calculator.total_optimal_pull_record))
        ],
        "Normalized Optimal Pulls",
        "Round",
        "Optimal Pull Percentage",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--confidence_rate", type=float, default=1)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
