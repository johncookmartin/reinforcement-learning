# John's Implementation

import math
import argparse
import random

# import matplotlib.pyplot as plt


############### UTIL FUNCTIONS #################
def calculate_running_average(prev_average, new_reward, total_attempts):
    return prev_average + (1 / max(total_attempts, 1)) * (new_reward - prev_average)


def cs_log(num):
    if num == 0:
        return 1
    return math.log(num)


def debug_print(str):
    # print(str)
    return


############# END UTIL FUNCTIONS ################


class BanditBuilder:
    def __init__(self, num_arms, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()

        self.bandit_arr = []
        self.optimal_action = None
        self.optimal_action_value = None
        for i in range(num_arms):
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
        print()
        print()
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

        # plt.plot(record)
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_times", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--confidence_rate", type=float, default=10)

    args = parser.parse_args()
    main(args)
