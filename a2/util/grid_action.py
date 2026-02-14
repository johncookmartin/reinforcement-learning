from enum import Enum, auto


class ActionTypes(Enum):
    UP = auto()
    DOWN = auto()
    RIGHT = auto()
    LEFT = auto()


class Action:
    def __init__(self, label, s, target_s, adjacent_s, r, p_one, p_two):
        self.label = label
        self.state = s
        self.target_state = target_s if target_s is not None else s
        self.adjacent_states = adjacent_s
        self.reward = r
        self.probability_array = [p_one, p_two, (1 - p_one - p_two) / 2]
