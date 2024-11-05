from ultralytics.data.augment import Albumentations
from nodeflow.node import Variable
from nodeflow.builtin import Result
import albumentations


class Compose(Variable):
    def __init__(self, value: albumentations.Compose):
        super().__init__(value)


def load_augmentations(transform_compose: Compose) -> Result:
    Albumentations.__init__ = lambda self, p: __apply_augmentations(self, transform_compose)
    return Result(True)

def __apply_augmentations(self, transform_compose):
        self.p = 1.0
        self.transform = None
        self.transform = transform_compose


__all__ = [
    'Compose',
    'load_augmentations',
]
