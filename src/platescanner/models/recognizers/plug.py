from pathlib import Path

from cvtk import AbstractDataset
from experimentator import Measurer
from experimentator.interfaces import Model
from prologger import Logger


class Plug(Model):
    @classmethod
    def load(cls, weights_path: Path):
        return Plug()

    def save(self, weights_path: Path):
        pass

    def compute(self, variable: object):
        return self.predict(variable)

    def is_loses_information(self) -> bool:
        return True

    def predict(self, input_tensor) -> str:
        import random
        random_num = lambda : str(random.choice(range(10)))
        random_chr = lambda : random.choice("ETYOPAHKXCBM")
        plate_text = ""
        plate_text += random_chr()
        for _ in range(3): plate_text += random_num()
        for _ in range(2): plate_text += random_chr()
        for _ in range(3): plate_text += random_num()
        return plate_text

    def train_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

    def test_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

    def eval_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        pass

__all__ = [
    'Plug'
]
