from decimal import ROUND_HALF_UP, Decimal
from random import Random
import time

from util.grid_world import GridWorld
from util.interfaces import AgentData


class GridAgent:
    def __init__(self, world: GridWorld, payload: AgentData):
        self.world = world
        self.rng = Random(payload.seed)

        self.max_episode_length = payload.max_episode_length
        self.terminal_reward = payload.terminal_reward
        self.discount = payload.discount
        self.p_one = payload.p_one
        self.p_two = payload.p_two
        self.epsilon = payload.epsilon
        self.alpha = payload.alpha

        self.episode = []

        self.avg_max_delta = Decimal(0)
        self.num_of_episodes = 0
        self.time_steps = 0

        self.start = None
        self.elapsed = None

        self.stable = 0
        self.prev_policy = None

    def choose_action(self, state):
        return self.rng.choices(state.actions, state.weights)[0]

    def take_action(self, action):
        results_list = [action.target, action.state]
        weights = [self.p_one, self.p_two]
        d = len(action.adjacent_states)
        if d > 0:
            p_three = (1 - self.p_one - self.p_two) / d
            for adjacent_state in action.adjacent_states:
                results_list.append(adjacent_state)
                weights.append(p_three)

        return self.rng.choices(results_list, weights=weights, k=1)[0]

    def adjust_weights(self, state):
        max_value = None
        optimal_actions = 0
        precision = Decimal("0.00000000001")
        for action in state.actions:
            rounded_value = action.value.quantize(precision, rounding=ROUND_HALF_UP)

            if max_value is None:
                max_value = rounded_value
                optimal_actions = 1
            elif rounded_value == max_value:
                optimal_actions += 1
            elif rounded_value > max_value:
                max_value = rounded_value
                optimal_actions = 1

        state.policy_actions = []
        for i in range(len(state.actions)):
            action = state.actions[i]

            if action.value.quantize(precision, rounding=ROUND_HALF_UP) == max_value:
                state.policy_actions.append(action)
                state.weights[i] = (
                    1 - self.epsilon + self.epsilon / (len(state.actions))
                )
            else:
                state.weights[i] = self.epsilon / len(state.actions)

    def print_results(self):
        value_cells = []
        for state in self.world.states:
            if "DOOR" in state.wall_state:
                value_cells.append("DOOR")
            elif state.wall_state != "None":
                value_cells.append("WALL")
            else:
                indicator = "**" if state.terminal_state else str(state.index)
                value_cells.append(
                    f"{indicator}: {state.policy_actions[0].value if state.policy_actions else 0:.2f}"
                )

        value_cell_width = max(len(cell) for cell in value_cells)

        print(
            f"GRID {self.world.size}      episodes = {self.num_of_episodes}      steps = {self.time_steps}      time = {self.elapsed}"
        )
        print("-" * 25)
        print()
        print("POLICY")
        for i, state in enumerate(self.world.states):
            if i % self.world.dimension == 0 and i > 0:
                print()

            if state.wall_state == "CROSS":
                print("||", end="=")
            elif state.wall_state == "COL":
                print("||", end=" ")
            elif state.wall_state == "ROW":
                print("======", end="=")
            elif state.wall_state == "COL_DOOR":
                print("  ", end=" ")
            elif state.wall_state == "ROW_DOOR":
                print("|    |", end="=")
            elif state.terminal_state:
                print("[TERM]", end=" ")
            else:
                actions_str = "".join(
                    [action.action.name[0] for action in state.policy_actions]
                )
                print(f"[{actions_str:4}]", end=" ")
        print()
        print("VALUES:")
        for i, value_cell in enumerate(value_cells):
            if i % self.world.dimension == 0 and i > 0:
                print()
            print(f"{value_cell:>{value_cell_width}}", end=" ")
        print()

    def start_timer(self):
        self.start = time.time()

    def stop_timer(self):
        elapsed = time.time() - self.start
        self.elapsed = f"{elapsed:.4f}"

    def policy_snapshot(self):
        return frozenset(
            (state.index, action.action)
            for state in self.world.states
            if not state.terminal_state and state.policy_actions
            for action in state.policy_actions
        )

    def compare_policy(self):
        current_policy = self.policy_snapshot()
        if current_policy == self.prev_policy:
            self.stable += 1
        else:
            self.stable = 0
        self.prev_policy = current_policy

    def create_episode(self):
        pass
