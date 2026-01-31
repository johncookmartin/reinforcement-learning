# John's Implementation

import math
import argparse
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

    def log_action(self, action, reward):
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

        self.total_pulls += 1


def main(args):

    for i in range(args.num_trials):
        print()
        print()
        print(f"TRIAL {i+1}")
        bandit = BanditBuilder(args.num_arms, args.seed)
        puller = UCBBanditPuller(args.num_arms, args.confidence_rate)

        optimal_value = bandit.get_expected_value(bandit.get_optimal_action())

        record = []
        optimal_record = []
        optimal_choices = 0
        for j in range(0, args.num_rounds + 1):
            picked_arm = puller.choose_action()
            value = bandit.pull_arm(picked_arm)
            puller.log_action(picked_arm, value)

            record.append(puller.average_reward)
            optimal_record.append(optimal_value)
            if picked_arm == bandit.get_optimal_action():
                optimal_choices += 1

            if j % 100 == 0:
                print(f"optimal arm pulled {optimal_choices} times")
                print(f"average reward at round {j}: {puller.average_reward}")

        print(
            f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
        )

        if args.show_plot:
            plot_results(
                [
                    {"record": optimal_record, "label": "optimal"},
                    {"record": record, "label": "puller record"},
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--confidence_rate", type=float, default=10)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
