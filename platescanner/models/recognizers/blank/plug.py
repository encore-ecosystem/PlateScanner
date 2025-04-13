from pathlib import Path

from cvtk import AbstractDataset
from experimentator.interfaces import Model
from prologger import Logger


class Plug(Model):
    def __init__(self, weight_path: Path):
        pass

    @classmethod
    def load(cls, weights_path: Path):
        pass

    def save(self, weights_path: Path):
        pass

    def compute(self, variable: object):
        pass

    def is_loses_information(self) -> bool:
        pass

    def predict(self, input_tensor):
        pass

    def train_step(self, dataset: AbstractDataset, measurer: 'Measurer', logger: Logger, current_epoch: int):
        pass

    def test_step(self, dataset: AbstractDataset, measurer: 'Measurer', logger: Logger, current_epoch: int):
        pass

    def eval_step(self, dataset: AbstractDataset, measurer: 'Measurer', logger: Logger, current_epoch: int):
        pass

__all__ = [
    'Plug'
]
