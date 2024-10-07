from abc import ABC, abstractmethod
from pathlib import Path


class AbstractDataset(ABC):
    def __init__(self):
        self.dataset = {
            type_: {'images_path': [], 'images_label': []}
            for type_ in ['train', 'test', 'valid']
        }
        self.categories = []

    @abstractmethod
    def save(self, save_dir: Path):
        pass

    @abstractmethod
    def __add__(self, other: 'AbstractDataset') -> 'AbstractDataset':
        pass

__all__ = [
    'AbstractDataset'
]
