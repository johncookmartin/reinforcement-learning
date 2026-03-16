from enum import Enum
from typing import List, NamedTuple

### Grid world interfaces


class GridWorldPayload(NamedTuple):
    dimensions: int
    accuracy: float
    terminal_states: List[int]
    column: int
    row: int
    doors: List[int]


class AgentData(NamedTuple):
    p_one: float
    p_two: float
    discount: float
    reward: float
    terminal_reward: float
    seed: int
    max_episode_length: int
    epsilon: float
    alpha: float


class AdjacentStates(Enum):
    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    LEFT = 3
    SELF = 4
    RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM = 7
    BOTTOM_RIGHT = 8
