from .abstract import AbstractDataset
from pathlib import Path



class CocoDataset(AbstractDataset):
    def __init__(self):
        super().__init__()
        self.annotations = {
            type_: None for type_ in ['train', 'val', 'test']
        }

    def save(self, save_dir: Path):
        raise NotImplementedError

    def __add__(self, other: 'CocoDataset') -> 'CocoDataset':
        ...


__all__ = [
    'CocoDataset'
]
