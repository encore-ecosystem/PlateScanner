from .bbox_abs import Bbox
from shapely.geometry import Polygon


class BboxPointBasedOBB(Bbox):
    def __init__(self, points: tuple[float, ...], category: int = -1) -> None:
        assert len(points) == 8
        super().__init__(points, category)

    def area(self) -> float:
        return Polygon(zip(self.bbox[0::2], self.bbox[1::2])).area

    def __and__(self, other: 'BboxPointBasedOBB') -> float:
        poly_A = Polygon(zip(self.bbox[0::2], self.bbox[1::2]))
        poly_B = Polygon(zip(other.bbox[0::2], other.bbox[1::2]))
        return poly_A.intersection(poly_B).area


__all__ = [
    'BboxPointBasedOBB'
]
