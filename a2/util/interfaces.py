from enum import Enum
from typing import List, NamedTuple

### Grid world interfaces


class GridWorldPayload(NamedTuple):
    dimensions: int
    accuracy: float
    reward_states: List[int]


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


### addition world interfaces


class DigitData(NamedTuple):
    digit_one: List[int]
    digit_two: List[int]


class AdditionData(NamedTuple):
    digit_one: list[int]
    digit_two: list[int]
    answer: list[int]
    discount: float


class AdditionActionData(NamedTuple):
    i: int
    j: int
    k: int
    s: int
