import shutil

from cvtk import Bbox_CWH, Bbox_4XY

from platescanner import TEMP_FOLDER
from pathlib import Path
from cvtk.interfaces import Bbox
from tqdm import tqdm
from cvtk.utils.determinator import determine_dataset
from cvtk.utils import autoconvert_dataset
from cvtk.supported_datasets import YOLO_Dataset

from platescanner.models.detectors import Yolo


def get_predicted_bboxes(dataset_path: Path, model: Yolo, conf: float, use_pbar: bool = True) -> dict[str, list[Bbox]]:
    bboxes = {}
    pbar = list((dataset_path / "valid" / "images").glob("*"))
    pbar = tqdm(pbar, total=len(pbar), desc='Processing predicted bboxes') if use_pbar else pbar
    for image_path in pbar:
        predicted_bboxes = model.predict(source=image_path, conf=conf)
        bboxes.update(predicted_bboxes)
    return bboxes


def get_target_bboxes(dataset_path: Path) -> dict[str, list[Bbox]]:
    if not isinstance(determine_dataset(dataset_path), YOLO_Dataset):
        dataset = autoconvert_dataset(dataset_path, YOLO_Dataset)
        dataset_path = TEMP_FOLDER / f"{dataset_path.name}_temp"
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset.write(dataset_path)

    return get_target_bboxes_yolo(dataset_path)


def get_target_bboxes_yolo(dataset_path: Path) -> dict[str, list[Bbox]]:
    bboxes = {}
    for label_name in tqdm(list((dataset_path / "valid" / "labels").glob("*.txt")), desc='Processing target bboxes'):
        with open(dataset_path / "valid" / "labels" / label_name, "r") as f:
            label_name = label_name.stem
            bboxes[label_name] = []
            for bbox_data in f.readlines():
                bbox_data = bbox_data.split()
                category = int(bbox_data.pop(0))
                points = list(map(float, bbox_data))
                match len(points):
                    # non-obb bbox
                    case 4:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            Bbox_CWH(points=points, category=category)
                        ]
                    # point-based obb bbox
                    case 8:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            Bbox_4XY(points=points, category=category)
                        ]
    return bboxes


__all__ = [
    'get_predicted_bboxes',
    'get_target_bboxes',
]