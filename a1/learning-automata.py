# John's Implementation

import argparse
import random
from util.bandit_builder import BanditBuilder
from util.functions import calculate_running_average, plot_results


class LinearRewardBanditPuller:
    def __init__(self, num_actions, reward_rate, penalty_rate, seed=None):
        self.rng = random.Random(seed) if seed else random.Random()

        self.num_actions = num_actions
        self.actions = [1 / num_actions for _ in range(num_actions)]
        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate
        self.total_pulls = 0
        self.average_reward = 0

    def choose_action(self):
        choice = self.rng.random()
        current = 0
        for i in range(self.num_actions):
            current += self.actions[i]
            if current > choice:
                return i
        return i

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

        # we want the pullers to have the same random seed
        if args.seed is None:
            args.seed = int(random.Random().random() * 100)

        inaction_puller = LinearRewardBanditPuller(
            args.num_arms, args.reward_rate, 0, args.seed
        )
        penalty_puller = LinearRewardBanditPuller(
            args.num_arms, args.reward_rate, args.penalty_rate, args.seed
        )

        print(inaction_puller.actions)
        print(penalty_puller.actions)

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
            optimal_record.append(
                bandit.get_expected_value(bandit.get_optimal_action())
            )

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

        plot_results(
            [
                {"record": optimal_record, "label": "optimal"},
                {"record": inaction_record, "label": "reward-inaction"},
                {"record": penalty_record, "label": "reward-panalty"},
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Automata Implementation")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reward_rate", type=float, default=0.01)
    parser.add_argument("--penalty_rate", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
