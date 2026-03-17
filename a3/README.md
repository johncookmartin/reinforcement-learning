# A3 - Reinforcement Learning: Model-Free Control

Three model-free RL algorithms operating on a stochastic grid world: Monte Carlo, SARSA, and Q-Learning.

## Grid World

An 11x11 grid (by default) split by a wall into two rooms connected by doors. The agent navigates to a terminal state, receiving a small negative reward each step and a large positive reward on reaching the goal.

Actions are stochastic: with probability `p_one` the intended move results in the agent moving to the intended state, `p_two` the agent stays in place, and the remaining probability is split among the chance the moves to one of the two states adjacent to the target state.

## Usage

Run any of the three scripts from the `a3/` directory:

```bash
python mc.py       # Monte Carlo
python sarsa.py    # SARSA
python q.py        # Q-Learning
```

## Arguments

All three scripts share the same arguments:

| Argument               | Default       | Description                                           |
| ---------------------- | ------------- | ----------------------------------------------------- |
| `--dimensions`         | `11`          | Grid side length (NxN)                                |
| `--terminal_states`    | `10`          | Index(es) of terminal states                          |
| `--wall_column`        | `5`           | Column index of the dividing wall                     |
| `--wall_row`           | `55`          | Row index of the dividing wall                        |
| `--doors`              | `27 57 63 93` | Indexes of door openings in the wall                  |
| `--terminal_reward`    | `500`         | Reward for reaching a terminal state                  |
| `--reward`             | `-1`          | Per-step reward                                       |
| `--p_one`              | `0.8`         | Probability the intended action succeeds              |
| `--p_two`              | `0.1`         | Probability the agent stays in place                  |
| `--discount`           | `0.9`         | Discount factor (γ)                                   |
| `--epsilon`            | `0.1`         | Exploration rate for ε-greedy policy                  |
| `--alpha`              | `0.1`         | Learning rate (SARSA and Q-Learning only)             |
| `--max_episode_length` | `1000`        | Maximum steps per episode                             |
| `--max_episodes`       | `10000`       | Number of episodes to run                             |
| `--seed`               | `None`        | Random seed for reproducibility                       |
| `--weight_init`        | `False`       | Prioritize unvisited states when choosing start state |
| `--decay_epsilon`      | `False`       | Decay epsilon over episodes (ε₀ / episode)            |

### Examples

```bash
# Monte Carlo with a fixed seed and epsilon decay
python mc.py --seed 42 --decay_epsilon

# SARSA with a higher learning rate and more episodes
python sarsa.py --alpha 0.3 --max_episodes 20000

# Q-Learning with weighted initialisation and a custom discount
python q.py --weight_init --discount 0.95 --seed 7
```

## Output

After all episodes complete, each script prints:

- Total episodes, time steps, and wall-clock time
- Average step ratio (episode steps / optimal steps from start state)
- Average episode time
- The learned policy displayed on the grid, colour-coded by visit frequency relative to the most-visited state:

```
RED    > 50%
YELLOW   25–50%
GREEN    12.5–25%
BLUE     6.25–12.5%
BLACK  < 6.25%
```
