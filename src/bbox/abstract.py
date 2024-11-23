from abc import ABC, abstractmethod


class Bbox(ABC):
    def __init__(self, bbox: tuple[float, ...], category: int = -1, confidence: float = 0):
        self.bbox       = tuple(bbox)
        self.category   = category
        self.confidence = confidence

    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_poly(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def __and__(self, other: 'Bbox') -> float:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.bbox)

    def __eq__(self, other: 'Bbox'):
        return self.bbox == other.bbox

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.category}: {self.bbox} {self.confidence:.2f})"


__all__ = [
    'Bbox'
]
