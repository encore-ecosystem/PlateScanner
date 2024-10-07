from abc import ABC, abstractmethod
from ..datasets import AbstractDataset
from .enum import AdapterOutputType
from pathlib import Path

class AbstractAdapter(ABC):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def forward(self, output_type: AdapterOutputType) -> AbstractDataset:
        match output_type:
            case AdapterOutputType.YOLO:
                return self.to_yolo()
            case AdapterOutputType.COCO:
                return self.to_coco()
            case _:
                raise NotImplementedError(f"Unsupported output type: {output_type}")

    @abstractmethod
    def to_yolo(self) -> AbstractDataset:
        raise NotImplementedError

    @abstractmethod
    def to_coco(self) -> AbstractDataset:
        raise NotImplementedError

__all__ = [
    'AbstractAdapter',
]
