# John's Implementation

import argparse
import random
from util.data_calculator import DataCalculator
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

        self.record = []
        self.pull_record = [0 for _ in range(num_actions)]

    def choose_action(self):
        choice = self.rng.random()
        current = 0
        for i in range(self.num_actions):
            current += self.actions[i]
            if current > choice:
                return i
        return i

    def log_action(self, action, reward, rank):
        self.total_pulls += 1
        self.pull_record[rank] += 1
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
        self.record.append(self.average_reward)

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

    inaction_calculator = DataCalculator(args.num_arms, args.num_rounds)
    penalty_calculator = DataCalculator(args.num_arms, args.num_rounds)

    # want to control the seed but also have a different see for each trial
    rng = random.Random(args.seed)
    init_seed = rng.randint(1, 100)

    for i in range(args.num_trials):
        print()
        print()
        print(f"TRIAL {i+1}")

        trial_seed = init_seed + i
        bandit = BanditBuilder(args.num_arms, trial_seed)
        inaction_puller = LinearRewardBanditPuller(
            args.num_arms, args.reward_rate, 0, trial_seed
        )
        penalty_puller = LinearRewardBanditPuller(
            args.num_arms, args.reward_rate, args.penalty_rate, trial_seed
        )

        optimal_value = bandit.get_expected_value(bandit.get_optimal_action())
        optimal_record = [optimal_value for _ in range(1, args.num_rounds + 1)]

        inaction_calculator.start_trial(i, optimal_value)
        penalty_calculator.start_trial(i, optimal_value)

        inaction_calculator.update_average_pull_values(bandit.sorted_values, i)
        penalty_calculator.update_average_pull_values(bandit.sorted_values, i)

        for j in range(args.num_rounds):
            i_picked_arm = inaction_puller.choose_action()
            p_picked_arm = penalty_puller.choose_action()

            i_value, i_rank = bandit.pull_arm(i_picked_arm)
            p_value, p_rank = bandit.pull_arm(p_picked_arm)

            inaction_puller.log_action(i_picked_arm, i_value, i_rank)
            penalty_puller.log_action(p_picked_arm, p_value, p_rank)

            inaction_calculator.update_value(inaction_puller.record[j], j)
            penalty_calculator.update_value(penalty_puller.record[j], j)

            inaction_calculator.update_pull_record(inaction_puller.pull_record, j)
            penalty_calculator.update_pull_record(penalty_puller.pull_record, j)

            if (j + 1) % 100 == 0:
                print(
                    f"reward-inaction pulled optimal arm {inaction_puller.pull_record[0]} times"
                )
                print(
                    f"average reward for reward-inaction at round {j+1}: {inaction_puller.average_reward}"
                )
                print(
                    f"reward-penalty pulled optimal arm {penalty_puller.pull_record[0]} times"
                )
                print(
                    f"avergage reward for reward-penalty at round {j+1}: {penalty_puller.average_reward}"
                )
        print(
            f"Optimal Expected Value: {bandit.get_expected_value(bandit.get_optimal_action())}"
        )

    print()
    print("FINAL RESULTS")
    print("-" * 50)
    print(f"Seed: {args.seed}")
    print(f"Reward Rate: {args.reward_rate}")
    print(f"Penalty Rate: {args.penalty_rate}")
    print(
        f"Final Average Value R-I: {inaction_calculator.total_optimal_value_record[-1]}"
    )
    print(
        f"Final Average Value R-P: {penalty_calculator.total_optimal_value_record[-1]}"
    )
    print("-" * 50)
    print(
        f"{'Pulled Arm':<10} | {'Inaction Percentage Pulled':<26} | {'Penalty Percentage Pulled':<25} | {'Average Value':<13}"
    )
    for i in range(len(inaction_calculator.total_optimal_pull_record)):
        print(
            f"{i:<10} | {inaction_calculator.total_optimal_pull_record[i][-1]:<26.15f} | {penalty_calculator.total_optimal_pull_record[i][-1]:<25.15f} | {inaction_calculator.average_pull_values[i]:<13.11f}"
        )

    plot_results(
        [
            {
                "record": [
                    1
                    for _ in range(len(inaction_calculator.total_optimal_value_record))
                ],
                "label": "optimal",
            },
            {
                "record": inaction_calculator.total_optimal_value_record,
                "label": "reward-inaction",
            },
            {
                "record": penalty_calculator.total_optimal_value_record,
                "label": "reward-penalty",
            },
        ],
        "Normalized Optimal Value",
        "Round",
        "Optimal Value as Percentage",
    )

    plot_results(
        [
            {
                "record": inaction_calculator.total_optimal_pull_record[0],
                "label": "reward-inaction",
            },
            {
                "record": penalty_calculator.total_optimal_pull_record[0],
                "label": "reward-penalty",
            },
        ],
        "Normalized Optimal Pulls",
        "Round",
        "Optimal Pull Percentage",
    )

    plot_results(
        [
            {
                "record": inaction_calculator.total_optimal_pull_record[i],
                "label": f"choice {i}, average value {inaction_calculator.average_pull_values[i]}",
            }
            for i in range(len(inaction_calculator.total_optimal_pull_record))
        ],
        "Normalized Inaction Optimal Pulls",
        "Round",
        "Optimal Pull Percentage",
    )

    plot_results(
        [
            {
                "record": penalty_calculator.total_optimal_pull_record[i],
                "label": f"choice {i}, average value {penalty_calculator.average_pull_values[i]}",
            }
            for i in range(len(penalty_calculator.total_optimal_pull_record))
        ],
        "Normalized Penalty Optimal Pulls",
        "Round",
        "Optimal Pull Percentage",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Automata Implementation")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward_rate", type=float, default=0.01)
    parser.add_argument("--penalty_rate", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
