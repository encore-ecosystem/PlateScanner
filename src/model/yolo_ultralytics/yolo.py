from typing import Unpack

from src.utils import bbox_to_total_area_filter, nms
from .abstract import YoloBase
from ultralytics import YOLO
from PIL import Image
from src.bbox import BboxXYXY
from pathlib import Path


class Yolo(YoloBase):

    def predict(self, **kwargs: Unpack) -> dict[str, BboxXYXY]:
        """
        :param kwargs:
            source     : path to image
            conf       : confidence level
        :return:
        """
        source  = Path(kwargs['source']).resolve()
        image   = Image.open(source)
        width, height = image.size

        results = YOLO(self.weights_path).predict(source=image, conf=kwargs['conf'], verbose=False, augment=True)

        confidence = {}
        bboxes     = []
        for result in results:
            for bbox_data in result.boxes.data.tolist():
                bbox      = BboxXYXY((bbox_data[0] / width, bbox_data[1] / height, bbox_data[2] / width, bbox_data[3] / height), category=int(bbox_data[5]))
                confidence[bbox] = bbox_data[4]
                bboxes.append(bbox)

        bboxes  = nms(
                bbox_to_total_area_filter(
                    bboxes           = tuple(bboxes),
                    area_threshold   = kwargs.get('area_threshold', 0.01),
                ),
            confidence,
            iou_threshold=0
        )
        return {source.stem: bboxes}


__all__ = [
    'Yolo'
]
