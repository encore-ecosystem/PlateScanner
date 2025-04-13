import random
from typing import Optional

import matplotlib.pyplot as plt
from cvtk import AbstractDataset, MVP_Dataset, Bbox
from experimentator import Pipeline, Measurer
from prologger import Logger
from tqdm import tqdm
from PIL import Image

from platescanner import EXPERIMENTATOR_PATH, MODELS_PATH
from platescanner.models.detectors import Yolo
from platescanner.models.recognizers import Parseq


class PTPipeline(Pipeline):
    def __init__(self, experiment_name_to_load: Optional[str] = None):
        if experiment_name_to_load:
            models = Pipeline.load(EXPERIMENTATOR_PATH / experiment_name_to_load)._models
        else:
            models = self.__default()

        super().__init__(models)

    @staticmethod
    def __default():
        models = [
            Yolo(  # Plate Detector
                weights=MODELS_PATH / "detector" / "yolo" / "poc_yolo5nu.pt",
            ),
            Parseq(  # Plate Recognizer
                device="cuda:0",
            ),
        ]
        return models

    def predict(self, dataset: MVP_Dataset) -> list[list[tuple[Bbox, str]]]:
        split = "test"
        pbar  = tqdm(dataset.images[split])
        result = []
        for image_stem in pbar:
            pairs: list[tuple[Bbox, str]] = []

            # step 0: Open image
            image_path = dataset.images[split][image_stem]
            image      = Image.open(image_path)

            # step 1: Plate detection
            bboxes: list[Bbox] = self._models[0].predict(image)

            # step 2: Plate recognition
            for bbox in bboxes:
                plate_image = bbox.crop_on(image)
                recognized_text = self._models[1].predict(plate_image)
                pairs.append((bbox, recognized_text))

            # step 3: Update result
            result.append(pairs)
        return result

    def train_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError

    def test_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError

    def eval_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError

    def compute(self, variable: object):
        raise NotImplementedError

    def is_loses_information(self) -> bool:
        raise NotImplementedError


__all__ = [
    'PTPipeline'
]