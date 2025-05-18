from cvtk import AbstractDataset, MVP_Dataset
from prologger import Logger

from platescanner import EXPERIMENTATOR_PATH, MODELS_PATH
from experimentator import Pipeline, Measurer, Model
from typing import Optional, Sequence

from platescanner.models.detectors import Yolo
from platescanner.models.recognizers import Parseq

from PIL import Image
from tqdm import tqdm
from pathlib import Path


class CPTPipeline(Pipeline):
    def __init__(self, experiment_name_to_load: Optional[str] = None):
        if experiment_name_to_load:
            models = Pipeline.load(EXPERIMENTATOR_PATH / experiment_name_to_load)._models
        else:
            models = self.__default()

        super().__init__(models)

    @staticmethod
    def __default() -> Sequence[Model]:
        models = [
            Yolo(  # Car Detector
                weights=MODELS_PATH / "detector" / "yolo" / "yolov5nu.pt",
            ),
            Yolo(  # Plate Detector
                weights=MODELS_PATH / "detector" / "yolo" / "yolov5nu.pt",
            ),
            Parseq(  # Plate Recognizer
                device="cuda:0"
            ),
        ]
        return models

    def predict(self, dataset: MVP_Dataset):
        split = 'test'
        pbar  = tqdm(dataset.images[split])
        for image_stem in pbar:
            image_path = dataset.images[split][image_stem]
            image      = Image.open(image_path)
            attributes = dataset.attributes[split][image_stem]

            # step 1: Car detector
            print(image, attributes)

    def save(self, dst_path: Path):
        for i, model in enumerate(self._models):
            model.save(dst_path / str(i))

    @classmethod
    def load(cls, src_path: Path) -> 'CPTPipeline':
        pipeline = CPTPipeline()
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
    "CPTPipeline"
]
