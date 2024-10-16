from .abstract import AbstractAdapter
from ..datasets import AbstractDataset
from .yolo import YoloAdapter
from .coco import CocoAdapter

class AutoAdapter(AbstractAdapter):
    def _detect(self) -> YoloAdapter | CocoAdapter:
        # YOLO
        if (self.dataset_path / 'data.yaml').exists() or (self.dataset_path / 'data.yml').exists():
            return YoloAdapter(self.dataset_path)
        # COCO
        else:
            return CocoAdapter(self.dataset_path)

    def to_yolo(self) -> AbstractDataset:
        return self._detect().to_yolo()

    def to_coco(self) -> AbstractDataset:
        return self._detect().to_coco()


__all__ = [
    'AutoAdapter'
]