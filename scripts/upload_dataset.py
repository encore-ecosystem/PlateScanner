from clearml import Dataset
from pathlib import Path


DATASET_PATH = Path().resolve() / 'dataset' / 'detection' / 'AUGMerged' / 'YOLO'

dataset = Dataset.create(
    dataset_project="PlateScanner", dataset_name=f"{DATASET_PATH.parent.name}_{DATASET_PATH.name}"
)

dataset.add_files(path=DATASET_PATH)
dataset.upload()
dataset.finalize()
