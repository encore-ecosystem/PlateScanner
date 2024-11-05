from nodeflow.builtin import PathVariable, Integer, Boolean
from nodeflow import Dispenser, func2node

from src.nodeflow_env import train_yolo, load_augmentations, Compose
from src.utils import bbox_to_total_area_filter, nms
from src.model import Model

from ultralytics import YOLO
from pathlib import Path
from PIL import Image

import albumentations


class YoloV5(Model):
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path

    def fit(self, dataset_path: Path, augmentation: albumentations.Compose, use_clearml: bool):
        Compose(value=augmentation) >> func2node(load_augmentations)
        Dispenser(
            model_path   = PathVariable(self.weights_path),
            dataset_path = PathVariable(dataset_path),
            imgsz        = Integer(1280),
            epochs       = Integer(50),
            use_clearml  = Boolean(use_clearml),
        ) >> func2node(train_yolo)

    def predict(self, **kwargs) -> list:
        """
        :param kwargs:
            source     : path to image
            conf       : confidence level
        :return:
        """
        image   = Image.open(kwargs['source'])
        results = YOLO(self.weights_path).predict(source=image, conf=kwargs['conf'], verbose=False, augment=True)
        bboxes  = bbox_to_total_area_filter(
            bboxes         = [result.boxes.data.tolist() for result in results][0],
            area_threshold = kwargs.get('area_threshold', 0.05),
            total_area     = image.size[0] * image.size[1]
        )
        bboxes = nms(bboxes, iou_threshold=0.01)
        return bboxes


__all__ = [
    'YoloV5'
]
