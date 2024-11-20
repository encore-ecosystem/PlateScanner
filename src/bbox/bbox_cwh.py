from shapely.geometry import box

from src.bbox.bbox_xyxy import BboxXYXY
from .bbox_abs import Bbox
from shapely.geometry import Polygon


class BboxCWH(Bbox):
    def __init__(self, center_x: float, center_y: float, width: float, height: float, category: int = -1, confidence: float = 0):
        super().__init__((center_x, center_y, width, height), category, confidence)

    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def __and__(self, other: 'BboxCWH') -> float:

        if isinstance(other, BboxCWH):
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
        elif isinstance(other, BboxXYXY):
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
    'BboxCWH'
]
