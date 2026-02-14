from enum import Enum
from typing import List, NamedTuple


class GridWorldPayload(NamedTuple):
    dimensions: int
    accuracy: float
    reward_states: List[int]
    seed: int


class BellmanData(NamedTuple):
    p_one: float
    p_two: float
    discount: float
    reward: float


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
