from platescanner import TEMP_FOLDER
from platescanner.model import YoloBase
from pathlib import Path
from platescanner.bbox import *
from tqdm import tqdm
from nodeflow.builtin.variables import PathVariable
from cvtk.utils.determinator import determine_dataset
from cvtk.supported_datasets.yolo import YOLO_Dataset, yolo_writer
from cvtk.supported_datasets.mvp import MVP_Dataset, MVP2YOLO_Adapter


def get_predicted_bboxes(dataset_path: Path, model: YoloBase, conf: float, use_pbar: bool = True) -> dict[str, list[Bbox]]:
    bboxes = {}
    pbar = list((dataset_path / "valid" / "images").glob("*"))
    pbar = tqdm(pbar, total=len(pbar), desc='Processing predicted bboxes') if use_pbar else pbar
    for image_path in pbar:
        predicted_bboxes = model.predict(source=image_path, conf=conf)
        bboxes.update(predicted_bboxes)
    return bboxes

def get_target_bboxes(dataset_path: Path) -> dict[str, list[Bbox]]:
    dataset = determine_dataset(dataset_path)
    if isinstance(dataset, YOLO_Dataset):
        return get_target_bboxes_yolo(dataset_path)
    elif isinstance(dataset, MVP_Dataset):
        yolo_dataset = MVP2YOLO_Adapter().compute(dataset)
        temp = PathVariable(TEMP_FOLDER)
        yolo_writer(yolo_dataset, PathVariable(TEMP_FOLDER))
        return get_target_bboxes_yolo(temp.value)
    else:
        raise TypeError("Unsupported dataset type.")


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