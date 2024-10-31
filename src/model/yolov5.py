from sympy.multipledispatch.dispatcher import RaiseNotImplementedError

from .model import Model
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

import numpy as np


class YoloV5(Model):
    def __init__(self, weights_path: Path):
        self.model = YOLO(weights_path)

    def fit(self, *args, **kwargs) -> 'YoloV5':
        raise RaiseNotImplementedError

    def predict(self, **kwargs) -> list:
        """
        :param kwargs:
            source     : path to image
            conf       : confidence level
        :return:
        """
        image = Image.open(kwargs['source'])
        results = self.model.predict(source=image, conf=kwargs['conf'], verbose=False, augment=True)
        bboxes = [result.boxes.data.tolist() for result in results][0]
        bboxes = self._filter_bboxes(
            bboxes,
            kwargs.get('area_threshold', 0.05),
            total_area = image.size[0] * image.size[1]
        )
        bboxes = self._nms(bboxes, iou_threshold=0.01)
        return bboxes

    def _filter_bboxes(self, bboxes, area_threshold, total_area: int) -> list:
        filtered_bboxes = [bbox for bbox in bboxes if self.box_area(bbox) / total_area <= area_threshold ]
        return filtered_bboxes

    @staticmethod
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _nms(self, bboxes: list, iou_threshold: float) -> list:
        bboxes        = tuple(tuple(x) for x in bboxes)
        result_bboxes = set(bboxes)
        for bbox_A in bboxes:
            to_delete = set()
            for bbox_B in result_bboxes - {bbox_A}:
                if bbox_A[5] == bbox_B[5] and self.bb_intersection_over_union(bbox_A, bbox_B) > iou_threshold:
                    if bbox_A[4] < bbox_B[4]:
                        to_delete.add(bbox_A)
                    else:
                        to_delete.add(bbox_B)
            result_bboxes -= to_delete
        return list(result_bboxes)

__all__ = ['YoloV5']
