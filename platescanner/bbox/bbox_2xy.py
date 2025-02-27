from .abstract import Bbox
from shapely.geometry import box


class Bbox_2xy(Bbox):
    def __init__(self, points: list[float], category: int = -1, confidence: float = 0):
        assert len(points) == 4
        super().__init__(points, category, confidence)

    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def get_poly(self) -> list[tuple[float, float]]:
        return list(box(*self.bbox).exterior.coords)

    def __and__(self, other: 'Bbox_2xy') -> float:
        return box(*self.bbox).intersection(box(*other.bbox)).area


__all__ = [
    'Bbox_2xy'
]
