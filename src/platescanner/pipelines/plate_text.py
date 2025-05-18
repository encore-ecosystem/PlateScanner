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

from pathlib import Path


class PTPipeline(Pipeline):
    def __init__(self, experiment_name_to_load: Optional[str] = None):
        if experiment_name_to_load:
            models = PTPipeline.load(EXPERIMENTATOR_PATH / experiment_name_to_load)._models
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

    def predict(self, inp: MVP_Dataset) -> list[list[Bbox]]:
        split = "test"
        pbar  = tqdm(inp.images[split])
        result = []
        for image_stem in pbar:
            image_path = inp.images[split][image_stem]
            bboxes = self.predict_on_image(image_path)
            result.append(bboxes)
        return result

    def predict_on_image(self, image_path: Path) -> list[Bbox]:
        image = Image.open(image_path)

        # step 1: Plate detection
        bboxes: list[Bbox] = self._models[0].predict(image)

        # step 2: Plate recognition
        for bbox in bboxes:
            plate_image = bbox.crop_on(image)
            recognized_text = self._models[1].predict(plate_image)
            bbox.value = recognized_text
        return bboxes

    def save(self, dst_path: Path):
        for i, model in enumerate(self._models):
            path = dst_path / str(i)
            path.mkdir(parents=True, exist_ok=True)
            model.save(path)

    @classmethod
    def load(cls, src_path: Path) -> 'PTPipeline':
        pipeline = PTPipeline()
        for i, model in enumerate(pipeline._models):
            pipeline._models[i] = model.load(src_path / str(i))
        return pipeline

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
