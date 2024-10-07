from .abstract import AbstractAdapter
from ..datasets import AbstractDataset, CocoDataset, YoloDataset
import json
import pprint

class CocoAdapter(AbstractAdapter):

    def to_yolo(self) -> YoloDataset:
        result_dataset = CocoDataset()

        for (path, key) in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            result_dataset.dataset[key] = [x for x in (self.dataset_path / path).iterdir() if x.name != '_annotations.coco.json']
            annotations_path = self.dataset_path / path / '_annotations.coco.json'
            assert annotations_path.exists(), f"Annotation file does not exist: {annotations_path}"

            with open(annotations_path) as f:
                annotations = json.load(f)
            result_dataset.annotations[key] = annotations
        raise NotImplementedError

    def to_coco(self) -> CocoDataset:
        raise NotImplementedError


__all__ = [
    'CocoAdapter'
]
