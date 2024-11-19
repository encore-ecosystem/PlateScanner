from dataclasses import dataclass
from enum import Enum


class Distance(Enum):
    CLOSE = 1
    MIDDLE = 2
    FAR = 3


class Time(Enum):
    DAY = 1
    NIGHT = 2


@dataclass
class CustomCategories:
    def __init__(self):
        self.distance = 0
        self.time = 0

    def __repr__(self):
        return f"{self.distance}, {self.time}"