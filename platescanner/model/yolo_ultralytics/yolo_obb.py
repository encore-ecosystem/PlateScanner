from platescanner.utils import bbox_to_total_area_filter, nms
from .abstract import YoloBase
from ultralytics import YOLO
from PIL import Image
from platescanner.bbox import Bbox_4XY, Bbox
from typing import Unpack
from pathlib import Path
from functools import reduce


class YoloOBB(YoloBase):

    def predict(self, **kwargs: Unpack) -> dict[str, list[Bbox_4XY]]:
        """
        :param kwargs:
            source     : path to image
            conf       : confidence level
        :return:
        """
        source = Path(kwargs['source']).resolve()
        image = Image.open(source)
        results = YOLO(self.weights_path).predict(source=image, conf=kwargs['conf'], verbose=False)
        width, height = image.size
        bboxes = []
        for result in results[0]:
            for bbox_data in result.obb.xyxyxyxy.tolist():
                points = reduce(lambda a, b: a + b, bbox_data)

                # Normalize points
                for i in range(len(points)):
                    if i % 2 == 0:
                        points[i] /= width
                    else:
                        points[i] /= height

                bbox = Bbox_4XY(
                    points,
                    category   = int(result.obb.cls.tolist()[0]),
                    confidence = float(result.obb.conf.tolist()[0]),
                )
                bboxes.append(bbox)

        # area filter
        bboxes = bbox_to_total_area_filter(
                bboxes=bboxes,
                area_threshold=kwargs.get('area_threshold', float("inf")),
        )

        # nms filter
        bboxes = nms(bboxes, iou_threshold=0.01)

        # result
        return {source.stem: bboxes}


__all__ = [
    'YoloOBB'
]