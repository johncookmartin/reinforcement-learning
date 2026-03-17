from decimal import ROUND_HALF_UP, Decimal
from random import Random
import time

from colorama import init, Back, Style

init()

from util.grid_world import GridWorld
from util.interfaces import AgentData


class GridAgent:
    def __init__(self, world: GridWorld, payload: AgentData):
        self.world = world
        self.rng = Random(payload.seed)

        self.max_episode_length = payload.max_episode_length
        self.reward = payload.reward
        self.terminal_reward = payload.terminal_reward
        self.discount = payload.discount
        self.p_one = payload.p_one
        self.p_two = payload.p_two
        self.epsilon_start = payload.epsilon
        self.epsilon = self.epsilon_start
        self.alpha = payload.alpha
        self.weight_init = payload.weight_init
        self.decay_epsilon = payload.decay_epsilon

        self.episode = []

        self.avg_max_delta = Decimal(0)
        self.num_of_episodes = 0
        self.time_steps = 0

        self.start = None
        self.elapsed = None

        self.stable = 0
        self.prev_policy = None

        self.choice_states = []
        for s in self.world.states:
            if not s.terminal_state and s.wall_state == "None":
                self.choice_states.append(s)

    def choose_weighted_state(self):
        weights = []
        for s in self.world.states:
            if s.terminal_state or s.wall_state != "None":
                weights.append(0)
            else:
                total_visits = sum(a.visits for a in s.actions)
                weights.append(1 / (total_visits + 1))
        return self.rng.choices(self.world.states, weights=weights)[0]

    def choose_init_state(self):
        if self.weight_init:
            return self.choose_weighted_state()
        else:
            return self.rng.choice(self.choice_states)

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

    def adjust_epsilon(self):
        if self.decay_epsilon:
            self.epsilon = self.epsilon_start / self.num_of_episodes

    def _visit_bg(self, state):
        visits = sum(a.visits for a in state.actions)
        all_visits = [
            sum(a.visits for a in s.actions)
            for s in self.world.states
            if s.actions and not s.terminal_state and s.wall_state == "None"
        ]
        max_v = max(all_visits) if all_visits else 0
        if max_v == 0:
            return Back.BLUE
        ratio = visits / max_v
        if ratio > 0.50:
            return Back.RED
        elif ratio > 0.25:
            return Back.YELLOW
        elif ratio > 0.125:
            return Back.GREEN
        elif ratio > 0.0625:
            return Back.BLUE
        else:
            return Back.BLACK

    def print_results(self):
        print(
            f"GRID {self.world.size}      episodes = {self.num_of_episodes}      steps = {self.time_steps}      time = {self.elapsed}"
        )
        print("-" * 25)
        print(
            f"KEY: "
            f"{Back.RED}  {Style.RESET_ALL} >50%  "
            f"{Back.YELLOW}  {Style.RESET_ALL} 25-50%  "
            f"{Back.GREEN}  {Style.RESET_ALL} 12.5-25%  "
            f"{Back.BLUE}  {Style.RESET_ALL} 6.25-12.5%  "
            f"{Back.BLACK}  {Style.RESET_ALL} <6.25%  "
            f"(visit ratio vs max)"
        )
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
                print(f"{self._visit_bg(state)}[TERM]{Style.RESET_ALL}", end=" ")
            else:
                actions_str = "".join(
                    [action.action.name[0] for action in state.policy_actions]
                )
                print(
                    f"{self._visit_bg(state)}[{actions_str:4}]{Style.RESET_ALL}",
                    end=" ",
                )
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
