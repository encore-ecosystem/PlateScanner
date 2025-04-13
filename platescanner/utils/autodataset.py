from cvtk import autoconvert_dataset, MVP_Dataset
from pathlib import Path

def to_mvp(dataset_path: Path) -> MVP_Dataset:
    return autoconvert_dataset(dataset_path, MVP_Dataset)


__all__ = [
    'to_mvp'
]
