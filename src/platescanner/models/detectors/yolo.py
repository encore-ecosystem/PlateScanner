import numpy as np
from cvtk import AbstractDataset, Bbox, bbox_to_total_area_filter, nms
from experimentator import Model, Measurer
from prologger import Logger
from ultralytics import YOLO
from cvtk.bbox import Bbox_2xy
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

import pickle
import torch
import json


@dataclass
class Yolo(Model):
    def __init__(
            self,
            weights: str | Path = 'yolov5nu',
            conf: float = 0,
            area_threshold: float = float('inf'),
    ):

        self._model = YOLO(weights, verbose=False)
        self._conf = conf
        self._area_threshold = area_threshold

    def predict(self, input_image: Image) -> list[Bbox]:
        input_tensor  = np.array(input_image)
        width, height = input_tensor.shape[0:2]

        results = self._model.predict(input_image, verbose=False)

        bboxes = []
        for result in results:
            for bbox_data in result.boxes.data.tolist():
                bbox = Bbox_2xy(
                    (bbox_data[0] / height, bbox_data[1] / width, bbox_data[2] / height, bbox_data[3] / width),
                    value=0,
                    category=int(bbox_data[5]),
                    confidence=bbox_data[4],
                )
                bboxes.append(bbox)

        # area filter
        bboxes = bbox_to_total_area_filter(
            bboxes=bboxes,
            area_threshold=self._area_threshold,
        )

        # nms filter
        bboxes = nms(bboxes, iou_threshold=0)

        # result
        return bboxes

    def train_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

    def test_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

    def eval_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

    def is_loses_information(self) -> bool:
        return True

    @classmethod
    def load(cls, weights_path: Path) -> "Yolo":
        path = weights_path / "model.pcl"
        with path.open("rb") as f:
            return pickle.load(f)

    def save(self, weights_path: Path):
        path = weights_path / "model.pcl"
        with path.open("wb") as f:
            pickle.dump(self, f)

    compute = predict


__all__ = ["Yolo"]
