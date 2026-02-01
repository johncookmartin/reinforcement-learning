# -*- coding: utf-8 -*-
"""Q2.ipynb

import random
import math
import time
import matplotlib.pyplot as plt
random.seed(time.time())

def make_environment(k=10):
    q = [random.random() for _ in range(k)]
    optimal_action = max(range(k), key=lambda i: q[i])
    return q, optimal_action

def pull_arm(q, action):
    return 1 if random.random() < q[action] else 0

def update_lr_p(p, chosen, reward, alpha=0.1, beta=0.1):
    k = len(p)

    if reward == 1:
        for j in range(k):
            if j == chosen:
                p[j] = p[j] + alpha * (1 - p[j])
            else:
                p[j] = (1 - alpha) * p[j]
    else:
        for j in range(k):
            if j == chosen:
                p[j] = (1 - beta) * p[j]
            else:
                p[j] = (beta / (k - 1)) + (1 - beta) * p[j]
    return p

def update_lr_i(p, chosen, reward, alpha=0.1):
    if reward == 1:
        k = len(p)
        for j in range(k):
            if j == chosen:
                p[j] = p[j] + alpha * (1 - p[j])
            else:
                p[j] = (1 - alpha) * p[j]
    return p

def choose_action_from_p(p):
    return random.choices(range(len(p)), weights=p, k=1)[0]

def run_learning_automaton_one_env(
    algo="LRP", k=10, steps=5000, alpha=0.1, beta=0.1, verbose=False
):
    q, optimal_action = make_environment(k)

    p = [1.0 / k] * k

    total_reward = 0
    optimal_count = 0

    checkpoints = []
    opt_counts = []
    avg_rewards = []

    for t in range(1, steps + 1):
        action = choose_action_from_p(p)
        reward = pull_arm(q, action)

        total_reward += reward
        if action == optimal_action:
            optimal_count += 1

        # Update probabilities
        if algo.upper() == "LRP":
            p = update_lr_p(p, chosen=action, reward=reward, alpha=alpha, beta=beta)
        elif algo.upper() == "LRI":
            p = update_lr_i(p, chosen=action, reward=reward, alpha=alpha)
        else:
            raise ValueError("algo must be 'LRP' or 'LRI'")

        # Record every 100 steps
        if t % 100 == 0:
            avg_r = total_reward / t
            checkpoints.append(t)
            opt_counts.append(optimal_count)
            avg_rewards.append(avg_r)

            if verbose:
                print(f"{algo}  t={t:4d}  optimal={optimal_count:4d}  avg_reward={avg_r:.4f}")

    return checkpoints, opt_counts, avg_rewards

cp, opt_counts, avg_rewards = run_learning_automaton_one_env(algo="LRI", verbose=True)
print("Done. Last avg reward:", avg_rewards[-1])

def run_many_envs_learning_automaton(
    algo="LRP", runs=100, k=10, steps=5000, alpha=0.1, beta=0.1
):
    num_checkpoints = steps // 100
    checkpoints = [(i + 1) * 100 for i in range(num_checkpoints)]

    sum_opt = [0] * num_checkpoints
    sum_avg_reward = [0.0] * num_checkpoints

    for _ in range(runs):
        cp, opt_counts, avg_rewards = run_learning_automaton_one_env(
            algo=algo, k=k, steps=steps, alpha=alpha, beta=beta, verbose=False
        )

        for i in range(num_checkpoints):
            sum_opt[i] += opt_counts[i]
            sum_avg_reward[i] += avg_rewards[i]

    avg_opt = [x / runs for x in sum_opt]
    avg_reward = [x / runs for x in sum_avg_reward]
    avg_opt_percent = [(avg_opt[i] / checkpoints[i]) * 100 for i in range(num_checkpoints)]

    return checkpoints, avg_opt, avg_opt_percent, avg_reward

runs = 100
steps = 5000
k = 10

alpha = 0.1
beta = 0.1

# LR-I results
cp, lri_opt, lri_opt_pct, lri_avg = run_many_envs_learning_automaton(
    algo="LRI", runs=runs, k=k, steps=steps, alpha=alpha, beta=beta
)

# LR-P results
cp, lrp_opt, lrp_opt_pct, lrp_avg = run_many_envs_learning_automaton(
    algo="LRP", runs=runs, k=k, steps=steps, alpha=alpha, beta=beta
)

print("Done running LR-I and LR-P.")
print("Example: first checkpoint =", cp[0],
      "LR-I avg reward =", round(lri_avg[0], 4),
      "LR-P avg reward =", round(lrp_avg[0], 4))

import matplotlib.pyplot as plt

# Plot 1: Average reward vs time
plt.figure()
plt.plot(cp, lri_avg, label="LR-I")
plt.plot(cp, lrp_avg, label="LR-P")
plt.xlabel("Time step (t)")
plt.ylabel("Average reward")
plt.title("Learning Automata: Average Reward vs Time (100 environments)")
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: Optimal action selection (%)
plt.figure()
plt.plot(cp, lri_opt_pct, label="LR-I")
plt.plot(cp, lrp_opt_pct, label="LR-P")
plt.xlabel("Time step (t)")
plt.ylabel("Optimal action chosen (%)")
plt.title("Learning Automata: Optimal Action Selection vs Time (100 environments)")
plt.grid(True)
plt.legend()
plt.show()
