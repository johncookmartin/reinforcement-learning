# John's Implementation

import random
import argparse
from util.bandit_builder import BanditBuilder


def main():

    parser = argparse.ArgumentParser(description="UCB Algorithm Implementaion")
    parser.add_argument("--num_bandits", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    bandit = BanditBuilder(args.num_bandits, args.seed)
    for i in range(args.num_bandits):
        print(f"arm {i}")
        print(f"value: {bandit.bandit_arr[i]["value"]}")
        print(f"prob: {bandit.bandit_arr[i]["prob"]}")
        print(f"expected_value: {bandit.get_expected_value(i)}")

    print(f"optimal_arm: {bandit.get_optimal_action()}")


if __name__ == "__main__":
    main()
