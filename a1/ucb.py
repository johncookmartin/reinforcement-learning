# John's Implementation

import math
import argparse

# import matplotlib.pyplot as plt
from util.bandit_builder import BanditBuilder


def calculate_running_average(prev_average, new_reward, total_attempts):
    return prev_average + (1 / max(total_attempts, 1)) * (new_reward - prev_average)


def cs_log(num):
    if num == 0:
        return 1
    return math.log(num)


def debug_print(str):
    # print(str)
    return


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
        debug_print(f"potential of bandit {action} is {potential}")
        debug_print(f"observed_reward is {observed_reward}")
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

    for i in range(args.num_times):
        print(f"TRIAL {i+1}")
        bandit = BanditBuilder(args.num_arms, args.seed)
        puller = UCBBanditPuller(args.num_arms, args.confidence_rate)

        record = []
        for j in range(0, args.num_rounds + 1):
            debug_print(puller.actions)
            picked_arm = puller.choose_action()
            debug_print(f"I picked arm {picked_arm}")
            value = bandit.pull_arm(picked_arm)
            puller.log_action(picked_arm, value)

            record.append(puller.average_reward)
            if j % 100 == 0:
                print(f"average reward at round {j}: {puller.average_reward}")

        print(
            f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
        )
        print()
        print()

        # plt.plot(record)
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_times", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--confidence_rate", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
