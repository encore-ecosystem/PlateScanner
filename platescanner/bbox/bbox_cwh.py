from .abstract import Bbox
from shapely.geometry import Polygon, box
from platescanner.bbox.bbox_2xy import Bbox_2xy


class Bbox_CWH(Bbox):
    def __init__(self, points: list[float], category: int = -1, confidence: float = 0):
        assert len(points) == 4
        super().__init__(points, category, confidence)

    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def __and__(self, other: 'Bbox_CWH') -> float:
        if isinstance(other, Bbox_CWH):
            poly_a = Polygon([
                (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
                (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
                (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
                (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
            ])
            poly_b = Polygon([
                (other.bbox[0] - other.bbox[2] / 2, other.bbox[1] - other.bbox[3] / 2),
                (other.bbox[0] + other.bbox[2] / 2, other.bbox[1] - other.bbox[3] / 2),
                (other.bbox[0] + other.bbox[2] / 2, other.bbox[1] + other.bbox[3] / 2),
                (other.bbox[0] - other.bbox[2] / 2, other.bbox[1] + other.bbox[3] / 2),
            ])
            return poly_a.intersection(poly_b).area
        elif isinstance(other, Bbox_2xy):
            poly_a = Polygon([
                (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
                (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
                (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
                (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
            ])
            poly_b = box(*other.bbox)
            return poly_a.intersection(poly_b).area


    def get_poly(self) -> list[tuple[float, float]]:
        poly_a = Polygon([
            (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
            (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] - self.bbox[3] / 2),
            (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
            (self.bbox[0] - self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
        ])
        return list(poly_a.exterior.coords)


__all__ = [
    'Bbox_CWH'
]
