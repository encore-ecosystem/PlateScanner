from src.model import YoloBase
from pathlib import Path
from src.bbox import *
from tqdm import tqdm


def get_predicted_bboxes(dataset_path: Path, model: YoloBase) -> dict[str, list[Bbox]]:
    bboxes = {}
    for image_path in tqdm(list((dataset_path / "valid" / "images").glob("*")), desc='Processing predicted bboxes'):
        predicted_bboxes = model.predict(source=image_path, conf=0.1)
        bboxes.update(predicted_bboxes)
    return bboxes

def get_target_bboxes(dataset_path: Path) -> dict[str, list[Bbox]]:
    bboxes = {}
    for label_name in tqdm(list((dataset_path / "valid" / "labels").glob("*.txt")), desc='Processing target bboxes'):
        with open(dataset_path / "valid" / "labels" / label_name, "r") as f:
            label_name = label_name.stem
            bboxes[label_name] = []
            for bbox_data in f.readlines():
                bbox_data = bbox_data.split()
                category = int(bbox_data.pop(0))
                points = tuple(map(float, bbox_data))
                match len(points):
                    # non-obb bbox
                    case 4:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            Bbox_CWH(*points, category=category)
                        ]
                    # point-based obb bbox
                    case 8:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            Bbox_4XY(points, category=category)
                        ]
    return bboxes


