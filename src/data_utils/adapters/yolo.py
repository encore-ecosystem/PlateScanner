from .abstract import AbstractAdapter
from ..datasets import AbstractDataset, YoloDataset
import yaml


class YoloAdapter(AbstractAdapter):

    def to_yolo(self) -> AbstractDataset:
        result_dataset = YoloDataset()

        if (self.dataset_path / 'data.yaml').exists():
            data_yaml_path = self.dataset_path / 'data.yaml'
        elif (self.dataset_path / 'data.yml').exists():
            data_yaml_path = self.dataset_path / 'data.yml'
        else:
            raise FileNotFoundError(f"There is no data.yaml or data.yml file in YOLO dataset: {self.dataset_path}")

        with open(data_yaml_path) as stream:
            try:
                data_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

        for subpath in result_dataset.dataset:
            if subpath not in data_yaml:
                raise KeyError(f"There is no key in data.yaml: {subpath}")

            target_path = self.dataset_path / data_yaml[subpath].lstrip('../')
            assert target_path.exists(), f"There is no target path: {target_path}"

            result_dataset.dataset[subpath]['images_path']  = [x for x in target_path.iterdir()]
            result_dataset.dataset[subpath]['images_label'] = [x for x in (target_path.parent / 'labels').iterdir()]

        if 'names' not in data_yaml:
            raise KeyError(f"There is no key in daya.yaml: names")

        result_dataset.categories = data_yaml['names']
        return result_dataset

    def to_coco(self) -> AbstractDataset:
        raise NotImplementedError


__all__ = [
    'YoloAdapter'
]
