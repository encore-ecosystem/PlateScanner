from nodeflow.builtin import PathVariable, Integer, Boolean
from nodeflow import Dispenser, func2node

from platescanner.nodeflow_env import train_yolo, load_augmentations, Compose, validate_yolo
from platescanner.model.model import Model
from pathlib import Path
from abc import ABCMeta

import albumentations


class YoloBase(Model, metaclass=ABCMeta):
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path

    def validate(self, dataset_path: Path, use_clearml: bool):
        Compose(value=albumentations.Compose([
            albumentations.ToGray(p=1.0),
            albumentations.CLAHE(p=1.0),
        ])) >> func2node(load_augmentations)
        Dispenser(
            model_path   = PathVariable(self.weights_path),
            dataset_path = PathVariable(dataset_path),
            imgsz        = Integer(1280),
            use_clearml  = Boolean(use_clearml),
        ) >> func2node(validate_yolo)

    def fit(self, dataset_path: Path, augmentation: albumentations.Compose, use_clearml: bool):
        Compose(value=augmentation) >> func2node(load_augmentations)
        Dispenser(
            model_path   = PathVariable(self.weights_path),
            dataset_path = PathVariable(dataset_path),
            imgsz        = Integer(1280),
            epochs       = Integer(30),
            use_clearml  = Boolean(use_clearml),
        ) >> func2node(train_yolo)


__all__ = [
    'YoloBase'
]
