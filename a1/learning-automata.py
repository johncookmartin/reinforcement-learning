# John's Implementation

import argparse
import random

import matplotlib.pyplot as plt


############### UTIL FUNCTIONS #################
def calculate_running_average(prev_average, new_reward, total_attempts):
    return prev_average + (1 / max(total_attempts, 1)) * (new_reward - prev_average)


def debug_print(str):
    # print(str)
    return


############## END UTIL FUNCTIONS #############


class BanditBuilder:
    def __init__(self, num_arms, seed=None, value_is_one=True):
        self.rng = random.Random(seed) if seed else random.Random()

        self.bandit_arr = []
        self.optimal_action = None
        self.optimal_action_value = None
        self.value_is_one = value_is_one
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
        value = 1 if self.value_is_one else self.rng.randint(1, 10)
        prob = self.rng.random()
        return {"value": value, "prob": prob}

    def get_expected_value(self, arm):
        value = self.bandit_arr[arm]["value"]
        prob = self.bandit_arr[arm]["prob"]
        return value * prob

    def get_optimal_action(self):
        return self.optimal_action

    def get_optimal_prob(self):
        return self.optimal_prob

    def pull_arm(self, arm):
        arm = self.bandit_arr[arm]
        if self.rng.random() < arm["prob"]:
            return arm["value"]
        else:
            return 0


class LinearRewardBanditPuller:
    def __init__(self, num_actions, reward_rate, penalty_rate):
        self.num_actions = num_actions
        self.actions = [1 / num_actions for _ in range(num_actions)]
        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate
        self.total_pulls = 0
        self.average_reward = 0

    def choose_action(self):
        best_action = 0
        highest_probability = 0
        for i in range(self.num_actions):
            if self.actions[i] > highest_probability:
                highest_probability = self.actions[i]
                best_action = i
        return best_action

    def log_action(self, action, reward):
        self.total_pulls += 1
        if reward > 0:
            self.average_reward = calculate_running_average(
                self.average_reward, 1, self.total_pulls
            )
            self.log_success(action)
        else:
            self.average_reward = calculate_running_average(
                self.average_reward, 0, self.total_pulls
            )
            self.log_failure(action)

    def log_success(self, action):
        for i in range(self.num_actions):
            current_prob = self.actions[i]
            if i == action:
                new_prob = current_prob + self.reward_rate * (1 - current_prob)
            else:
                new_prob = (1 - self.reward_rate) * current_prob
            self.actions[i] = new_prob

    def log_failure(self, action):
        for i in range(self.num_actions):
            current_prob = self.actions[i]
            if i == action:
                new_prob = (self.penalty_rate / (self.num_actions - 1)) + (
                    1 - self.penalty_rate
                ) * current_prob
            else:
                new_prob = (1 - self.penalty_rate) * current_prob
            self.actions[i] = new_prob


def main(args):

    for i in range(args.num_trials):
        print()
        print()
        print(f"TRIAL {i+1}")
        bandit = BanditBuilder(args.num_arms, args.seed)
        inaction_puller = LinearRewardBanditPuller(args.num_arms, args.reward_rate, 0)
        penalty_puller = LinearRewardBanditPuller(
            args.num_arms, args.reward_rate, args.penalty_rate
        )

        optimal_record = []
        inaction_record = []
        penalty_record = []
        for j in range(0, args.num_rounds + 1):
            i_picked_arm = inaction_puller.choose_action()
            p_picked_arm = penalty_puller.choose_action()

            i_value = bandit.pull_arm(i_picked_arm)
            p_value = bandit.pull_arm(p_picked_arm)

            inaction_puller.log_action(i_picked_arm, i_value)
            penalty_puller.log_action(p_picked_arm, p_value)

            inaction_record.append(inaction_puller.average_reward)
            penalty_record.append(penalty_puller.average_reward)
            optimal_record.append(bandit.get_optimal_action())

            if j % 100 == 0:
                print(
                    f"average reward for reward-inaction at round {j}: {inaction_puller.average_reward}"
                )
                print(
                    f"avergage reward for reward-penalty at round {j}: {penalty_puller.average_reward}"
                )
        print(
            f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
        )

        plt.plot(optimal_record, label="optimal")
        plt.plot(inaction_record, label="reward-inaction")
        plt.plot(penalty_record, label="reward-penalty")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Automata Implementation")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reward_rate", type=float, default=0.9)
    parser.add_argument("--penalty_rate", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
