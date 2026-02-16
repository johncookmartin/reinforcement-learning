import argparse

from util.addition_world import AdditionWorld
from util.interfaces import DigitData


def get_digit(digit_name):
    is_number = False
    while not is_number:
        digit_arr = []
        print(f"enter {digit_name}:", end="")
        digit_one_str = input()
        if len(digit_one_str) < 1:
            print("you must enter a value!")
        else:
            # want the least significant digit to populate at 0 index
            # so that the i and j values make sense for the carry
            for char in digit_one_str[::-1]:
                try:
                    num = int(char)
                    digit_arr.append(num)
                except Exception:
                    print("you must enter only numbers!")
                    break

            if len(digit_arr) == len(digit_one_str):
                is_number = True
            else:
                print("please try again")
                print()
    return digit_arr


def main(args):

    digit_input = None
    print("would you like to enter custom digits?(Y/N):")
    response = input()
    if response.upper() == "Y":
        digit_input = DigitData(
            digit_one=get_digit("digit 1"), digit_two=get_digit("digit 2")
        )

    addition_world = AdditionWorld(args.digits, args.discount, args.seed, digit_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Addition Implementation")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
