from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Distance(Enum):
    CLOSE = 1
    MIDDLE = 2
    FAR = 3


class Time(Enum):
    DAY = 1
    NIGHT = 2


@dataclass
class CustomCriteria:
    def __init__(self):
        self.distance: Optional[Distance] = None
        self.time    : Optional[Time]     = None

    def __repr__(self) -> str:
        return f"{self.distance}, {self.time}"

    def __eq__(self, criteria: 'CustomCriteria') -> bool:
        if not (criteria.distance is None or self.distance is None or self.distance == criteria.distance):
            return False

        if not (criteria.time is None or self.time is None or self.time == criteria.time):
            return False

        return True



__all__ = [
    'Distance',
    'Time',
    'CustomCriteria',
]
