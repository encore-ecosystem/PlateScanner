from PIL.Image import Image
from PIL.ImageFile import ImageFile

from .abstract import Bbox
from shapely.geometry import Polygon

# obb
class Bbox_4XY(Bbox):
    def __init__(self, points: list[float], category: int = -1, confidence: float = 0) -> None:
        assert len(points) == 8
        super().__init__(points, category, confidence)

    def area(self) -> float:
        return Polygon(zip(self.bbox[0::2], self.bbox[1::2])).area

    def get_poly(self) -> list[tuple[float, float]]:
        return list(zip(self.bbox[0::2], self.bbox[1::2]))

    def crop_on(self, image: ImageFile) -> Image:
        return image.crop((self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))

    def __and__(self, other: 'Bbox_4XY') -> float:
        poly_A = Polygon(zip(self.bbox[0::2], self.bbox[1::2]))
        poly_B = Polygon(zip(other.bbox[0::2], other.bbox[1::2]))
        return poly_A.intersection(poly_B).area


__all__ = [
    'Bbox_4XY'
]
