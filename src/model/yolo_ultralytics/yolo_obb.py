from src.utils import bbox_to_total_area_filter, nms
from .abstract import YoloBase
from ultralytics import YOLO
from PIL import Image
from src.bbox import BboxPointBasedOBB
from src.bbox.bbox_abs import Bbox
from typing import Unpack
from pathlib import Path


class YoloOBB(YoloBase):

    def predict(self, **kwargs: Unpack) -> dict[str, list[BboxPointBasedOBB]]:
        """
        :param kwargs:
            source     : path to image
            conf       : confidence level
        :return:
        """
        source = Path(kwargs['source']).resolve()
        image = Image.open(source)
        results = YOLO(self.weights_path).predict(source=image, conf=kwargs['conf'], verbose=False, augment=True)

        confidence = {}
        bboxes = []
        for result in results:
            for bbox_data in result.boxes.data.tolist():
                bbox = BboxPointBasedOBB((bbox_data[:8]), category=int(bbox_data[9]), confidence=bbox_data[8])
                bboxes.append(bbox)

        bboxes = nms(
            bbox_to_total_area_filter(
                bboxes=tuple(bboxes),
                area_threshold=kwargs.get('area_threshold', 0.05),
            ),
            iou_threshold=0.01
        )
        return {source.stem: bboxes}


__all__ = [
    'YoloOBB'
]