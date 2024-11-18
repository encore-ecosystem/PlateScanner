from .bbox_abs import Bbox
from shapely.geometry import box


class BboxXYXY(Bbox):
    def __init__(self, points: tuple[float, ...], category: int = -1):
        assert len(points) == 4
        super().__init__(points, category)

    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def __and__(self, other: 'BboxXYXY') -> float:
        poly_a = box(*self.bbox)
        poly_b = box(*other.bbox)
        return poly_a.intersection(poly_b).area


__all__ = [
    'BboxXYXY'
]
