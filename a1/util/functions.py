import math

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def calculate_running_average(prev_average, new_reward, total_attempts):
    return prev_average + (1 / max(total_attempts, 1)) * (new_reward - prev_average)


# because we are in computer science, we just have log0 return 1
def cs_log(num):
    if num == 0:
        return 1
    return math.log(num)


def debug_print(str, debug=False):
    if debug:
        print(str)
        input()
    return


# this will never run on moon
def plot_results(data, title, x_label, y_label):
    if MATPLOTLIB_AVAILABLE:
        for point in data:
            label = point.get("label", "")
            record = point.get("record")
            plt.plot(record, label=label)
        plt.title(title if title else "")
        plt.xlabel(x_label if x_label else "")
        plt.ylabel(y_label if y_label else "")
        plt.legend()
        plt.show()


def as_9bit(x):
    mask = 0b111111111
    return x & mask


def as_9bit_flipped(x):
    mask = 0b111111111
    return (~x) & mask
