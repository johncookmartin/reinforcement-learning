from decimal import Decimal

from util.interfaces import AdjacentStates, GridWorldPayload
from util.grid_state import GridState


class GridWorld:
    def __init__(self, payload: GridWorldPayload):
        self.dimension = payload.dimensions
        self.accuracy = payload.accuracy
        self.terminal_states = payload.terminal_states

        self.column = payload.column
        self.row = payload.row
        self.doors = payload.doors

        self.size = self.dimension**2

        self.states = []
        self.k = 0
        self.i = 0
        self.delta = 0

    def determine_wall_state(self, i):
        if (
            self.column is not None
            and self.row is not None
            and i % self.dimension == self.column
            and i >= self.column * self.dimension
            and i < self.column * self.dimension + self.dimension
        ):
            return "CROSS"
        elif self.column is not None and i % self.dimension == self.column:
            if i in self.doors:
                return "COL_DOOR"
            else:
                return "COL"
        elif (
            self.row is not None
            and i >= self.column * self.dimension
            and i < self.column * self.dimension + self.dimension
        ):
            if i in self.doors:
                return "ROW_DOOR"
            else:
                return "ROW"
        else:
            return "None"

    def create_states(self, agent_data):
        # populate the dict with states
        for i in range(self.size):
            if i in self.terminal_states:
                self.states.append(GridState(i, agent_data, terminal_state=True))
            else:
                self.states.append(
                    GridState(i, agent_data, wall_state=self.determine_wall_state(i))
                )

    def join_states(self):
        for i in range(self.size):
            if self.determine_wall_state(i) != "None":
                # no need to join to a wall or door
                continue

            # create index of neighbouring states
            neighbour_indexes = {
                AdjacentStates.TOP_LEFT: i - self.dimension - 1,
                AdjacentStates.TOP: i - self.dimension,
                AdjacentStates.TOP_RIGHT: i - self.dimension + 1,
                AdjacentStates.LEFT: i - 1,
                AdjacentStates.SELF: i,
                AdjacentStates.RIGHT: i + 1,
                AdjacentStates.BOTTOM_LEFT: i + self.dimension - 1,
                AdjacentStates.BOTTOM: i + self.dimension,
                AdjacentStates.BOTTOM_RIGHT: i + self.dimension + 1,
            }

            # iterate through states and join adjacent states
            # this will create a graph with states as nodes
            state = self.states[i]
            # check to see if state is on the left or right side (or neither)
            is_left = state.index % self.dimension == 0
            is_right = state.index % self.dimension == self.dimension - 1
            for adjacent_state, index in neighbour_indexes.items():
                # if state is on the left edge, there are no states to the left
                if "LEFT" in adjacent_state.name and is_left:
                    state.join_states(adjacent_state, None)
                # if state is on the right edge, there are no states to the right
                elif "RIGHT" in adjacent_state.name and is_right:
                    state.join_states(adjacent_state, None)
                # if the index is out of bounds of the state array, the current state
                # must be on the top or bottom of the grid
                elif 0 <= index < len(self.states):

                    # check if the state is a wall or door
                    wall_state = self.determine_wall_state(index)
                    # if adjacent to a door, re-route to the state on the
                    # opposite side
                    if wall_state == "COL_DOOR":
                        if "LEFT" in adjacent_state.name:
                            state.join_states(adjacent_state, self.states[index - 1])
                        else:
                            state.join_states(adjacent_state, self.states[index + 1])
                    elif wall_state == "ROW_DOOR":
                        if "TOP" in adjacent_state.name:
                            state.join_states(
                                adjacent_state, self.states[index - self.dimension]
                            )
                        else:
                            state.join_states(
                                adjacent_state, self.states[index + self.dimension]
                            )
                    elif wall_state != "None":
                        state.join_states(adjacent_state, None)
                    else:
                        state.join_states(adjacent_state, self.states[index])
                else:
                    state.join_states(adjacent_state, None)

    def initialize_actions(self):
        for state in self.states:
            state.initialize_actions()

    # print the grid with values and policies
    def print_grid(self):
        value_cells = []
        for state in self.states:
            if "DOOR" in state.wall_state:
                value_cells.append("DOOR")
            elif state.wall_state != "None":
                value_cells.append("WALL")
            else:
                indicator = "**" if state.terminal_state else str(state.index)
                value_cells.append(f"{indicator}: {state.value:.2f}")

        value_cell_width = max(len(cell) for cell in value_cells)

        print(f"GRID {self.size}      k = {self.k}      i = {self.i}")
        print("-" * 25)
        print()
        print("VALUES:")
        for i, value_cell in enumerate(value_cells):
            if i % self.dimension == 0 and i > 0:
                print()
            print(f"{value_cell:>{value_cell_width}}", end=" ")
        print()
        print("ACTIONS")
        for i, state in enumerate(self.states):
            if i % self.dimension == 0 and i > 0:
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
                    [action.action.name[0] for action in state.actions]
                )
                print(f"[{actions_str:4}]", end=" ")
        print()
