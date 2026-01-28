# John's Implementation

import math
import argparse
from util.bandit_builder import BanditBuilder


def calculate_running_average(prev_average, new_reward, total_attempts):
    return prev_average + (1 / max(total_attempts, 1)) * (new_reward - prev_average)


def cs_log(num):
    if num == 0:
        return 1
    return math.log(num)


class UCBBanditPuller:
    def __init__(self, num_actions, learning_rate=100):
        self.actions = [{"observed_reward": 0, "times_pulled": 0}] * num_actions
        self.total_pulls = 0
        self.learning_rate = learning_rate
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
            * self.learning_rate
        )
        return observed_reward + potential

    def log_action(self, action, reward):
        self.total_pulls += 1

        # calculate average reward for specific action
        observed_reward = self.actions[action]["observed_reward"]
        self.actions[action]["observed_reward"] = calculate_running_average(
            observed_reward, reward, self.total_pulls
        )

        # calculate total average reward
        self.average_reward = calculate_running_average(
            self.average_reward, reward, self.total_pulls
        )


def main():

    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_bandits", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    bandit = BanditBuilder(args.num_bandits, args.seed)
    puller = UCBBanditPuller(args.num_bandits, args.learning_rate)

    record = []
    print(
        f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
    )
    for i in range(args.num_rounds):
        picked_arm = puller.choose_action()
        value = bandit.pull_arm(picked_arm)
        puller.log_action(picked_arm, value)

        record.append(puller.average_reward)
        if i % 100 == 0:
            print(f"average reward at round {i}: {puller.average_reward}")

    print(
        f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
    )


if __name__ == "__main__":
    main()
