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

## Architecture

### Entry points: `mc.py`, `sarsa.py`, `q.py`

Each script follows the same pattern:

1. Parse args with `get_parser()` from `util/args.py`
2. Call `setup_grid(args)` to build the world and pack agent configuration
3. Instantiate the algorithm-specific agent subclass
4. Run the training loop, then call `agent.print_results()`

The training loop calls a fixed sequence of agent methods each episode:

```
create_episode()          # algorithm-specific: collect transitions and/or update Q
update_action_values()    # MC only: backward pass over buffered episode
calculate_episode()       # record timing and step-ratio stats
compare_policy()          # track policy stability
adjust_epsilon()          # decay epsilon if --decay_epsilon is set
```

---

### `util/args.py`

**`get_parser()`** — builds the shared `argparse` parser for all three scripts.

**`setup_grid(args)`** — factory function called by every `main()`. Packs args into `GridWorldPayload` and `AgentData`, constructs a `GridWorld`, then calls the four build steps in order:

| Call                             | Effect                                                                 |
| -------------------------------- | ---------------------------------------------------------------------- |
| `grid.create_states(agent_data)` | Instantiates a `GridState` for every cell                              |
| `grid.join_states()`             | Wires neighbour references, routing through doors                      |
| `grid.calculate_distances()`     | BFS from terminal(s) to populate `distance_to_terminal`                |
| `grid.initialize_actions()`      | Creates four `GridAction` objects on each non-wall, non-terminal state |

Returns `(grid, agent_data)` to `main()`.

### `util/grid_world.py` — `GridWorld`

Owns the list of all `GridState` objects and the grid topology.

| Method                      | Responsibility                                                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `determine_wall_state(i)`   | Classifies cell `i` as COL, ROW, COL_DOOR, ROW_DOOR, CROSS, or None                                                                     |
| `create_states(agent_data)` | Allocates `GridState` per cell with correct terminal/wall flags                                                                         |
| `join_states()`             | Sets `neighbours` on every non-wall state; door cells are skipped as targets and the cell on the far side of the door is linked instead |
| `calculate_distances()`     | BFS from terminal states outward, setting `distance_to_terminal` on every reachable state                                               |
| `initialize_actions()`      | Delegates to `state.initialize_actions()` on every state                                                                                |

---

### `util/grid_state.py` — `GridState`

Represents one cell in the grid.

| Attribute              | Purpose                                                                    |
| ---------------------- | -------------------------------------------------------------------------- |
| `neighbours`           | Length-9 list indexed by `AdjacentStates`; `None` means wall/edge          |
| `actions`              | Four `GridAction` objects (UP, DOWN, LEFT, RIGHT)                          |
| `weights`              | Per-action ε-greedy probabilities; updated by `GridAgent.adjust_weights()` |
| `policy_actions`       | Subset of `actions` currently considered optimal                           |
| `distance_to_terminal` | Shortest path distance used to normalise episode step counts               |

**`join_states(adjacent_state, state)`** — stores a neighbour reference at the given direction index.

**`initialize_actions()`** — creates the four `GridAction` objects using the current `neighbours` array.

---

### `util/grid_action.py` — `GridAction`

Represents one directional action available from a `GridState`.

| Attribute         | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `target`          | The state the agent intends to reach (falls back to `self.state` if wall) |
| `adjacent_states` | The two states flanking the target                                        |
| `visits`          | Incremented each time this action is updated                              |
| `value`           | Current Q-value estimate (Decimal for precision)                          |

**`get_target(neighbours)`** — resolves the intended destination; returns the current state if the direction is blocked.

**`get_adjacent_states(neighbours)`** — collects the two diagonal-adjacent states; defaults to `target` if a diagonal is `None`.

---

### `util/grid_agent.py` — `GridAgent`

Base class inherited by all three algorithm agents. Holds the RNG, all hyperparameters, and episode statistics. Implements everything that is shared across algorithms:

| Method                                    | Responsibility                                                                          |
| ----------------------------------------- | --------------------------------------------------------------------------------------- |
| `choose_init_state()`                     | Uniform random or visit-weighted start state                                            |
| `choose_action(state)`                    | ε-greedy sample using `state.weights`                                                   |
| `take_action(action)`                     | Stochastic outcome: target with p_one, stay with p_two, adjacent target with remainder  |
| `adjust_weights(state)`                   | Recomputes `state.weights` and `state.policy_actions` after a Q-value update            |
| `adjust_epsilon()`                        | Applies ε decay if enabled                                                              |
| `start_episode()` / `calculate_episode()` | Episode bookkeeping: reset buffer, update timing and step-ratio averages                |
| `compare_policy()`                        | Snapshots the greedy policy and tracks how many consecutive episodes it has been stable |
| `print_results()`                         | Prints summary stats and the colour-coded policy grid                                   |

`create_episode()` function stub overwritten by every child class.

---

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
