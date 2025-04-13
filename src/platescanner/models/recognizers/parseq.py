import numpy as np
import torchvision.transforms as torch_transforms

from platescanner.utils.plate.russian_plate import process_text
from experimentator import Model, Measurer
from cvtk import AbstractDataset
from prologger import Logger
from pathlib import Path
from PIL.Image import Image

import torch


class Parseq(Model):
    def __init__(self, device: str = "cpu"):
        self._device      = torch.device(device)
        self._model_cache = {}
        self._model       = torch.hub.load(
            'baudm/parseq', 'parseq', pretrained=True
        ).to(self._device)

    @torch.inference_mode()
    def __call__(self, image: Image) -> str:
        preprocessor = torch_transforms.Compose(
            [
                torch_transforms.Resize((32, 128), torch_transforms.InterpolationMode.BICUBIC),
                torch_transforms.ToTensor(),
            ]
        )
        image = preprocessor(image.convert("RGB"))
        if len(image.shape) == 3: # add batch dim
            image = image.unsqueeze(0)
        image = image.to(self._device)

        # Greedy decoding
        pred = self._model(image).softmax(-1)
        label, _ = self._model.tokenizer.decode(pred)

        # Format confidence values
        recognized_text = label[0]
        if recognized_text:
            recognized_text = process_text(recognized_text)

        return recognized_text

    @classmethod
    def load(cls, weights_path: Path):
        raise NotImplementedError

    def save(self, weights_path: Path):
        raise NotImplementedError

    def train_step(
        self,
        dataset: AbstractDataset,
        measurer: Measurer,
        logger: Logger,
        current_epoch: int,
    ):
        raise NotImplementedError

    def test_step(
        self,
        dataset: AbstractDataset,
        measurer: Measurer,
        logger: Logger,
        current_epoch: int,
    ):
        raise NotImplementedError

    def eval_step(
        self,
        dataset: AbstractDataset,
        measurer: Measurer,
        logger: Logger,
        current_epoch: int,
    ):
        raise NotImplementedError

    def compute(self, variable: object):
        raise NotImplementedError

    def is_loses_information(self) -> bool:
        raise NotImplementedError

    predict = __call__


__all__ = [
    'Parseq'
]
