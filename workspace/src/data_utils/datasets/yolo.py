from .abstract import AbstractDataset
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml


class YoloDataset(AbstractDataset):
    def __init__(self):
        super().__init__()
        self.dataset['val'] = self.dataset['valid']
        del self.dataset['valid']

    def save(self, save_dir: Path):
        if save_dir.exists():
            shutil.rmtree(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        for type_ in self.dataset:
            type_path = save_dir / (type_ if type_ != 'val' else 'valid')
            images_path = type_path / 'images'
            labels_path = type_path / 'labels'

            type_path.mkdir(parents=True, exist_ok=True)
            images_path.mkdir(parents=True, exist_ok=True)
            labels_path.mkdir(parents=True, exist_ok=True)

            for src_image_path, src_label in tqdm(
                zip(self.dataset[type_]['images_path'], self.dataset[type_]['images_label']),
                desc = f'Working on {type_} data'
            ):
                shutil.copy(src=src_image_path, dst=images_path / src_image_path.name)
                shutil.copy(src=src_label, dst=labels_path / src_label.name)

        data_yaml = {
            'train': '../train/images',
            'val'  : '../valid/images',
            'test' : '../test/images',

            'nc': len(self.categories),
            'names': self.categories
        }
        with open(save_dir / 'data.yaml', 'w') as stream:
            yaml.dump(data_yaml, stream)

    def __add__(self, other: 'YoloDataset') -> 'YoloDataset':
        res = YoloDataset()
        res.categories = list(set(self.categories) | set(other.categories))
        for type_ in self.dataset:
            for subtype in self.dataset[type_]:
                res.dataset[type_][subtype] = self.dataset[type_][subtype] + other.dataset[type_][subtype]
        return res


__all__ = [
    'YoloDataset'
]
