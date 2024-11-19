import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.model import Yolo
from src.bbox import BboxCWH, BboxPointBasedOBB
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.validator.stat import Validator
from src.bbox import *
from src.validator.custom_categories import CustomCategories
#
# VALIDATE
#
DATASET_PATH = (
    Path(__file__).parent.parent.resolve()
    / "dataset"
    / "detection"
    / "validation_dataset"
)


def parse_dataset(dataset_path: Path):
    bboxes = {}
    for label_name in tqdm((dataset_path / "valid" / "labels").glob("*.txt")):
        with open(dataset_path / "valid" / "labels" / label_name, "r") as f:
            label_name = label_name.stem
            for bbox_data in f.readlines():
                bbox_data = bbox_data.split()
                category = int(bbox_data.pop(0))
                points = tuple(map(float, bbox_data))
                match len(points):
                    # non-obb bbox
                    case 4:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            BboxCWH(*points, category=category)
                        ]
                    # point-based obb bbox
                    case 8:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            BboxPointBasedOBB(points, category=category)
                        ]
    return bboxes


def parse_predicted(dataset_path: Path):
    bboxes = {}
    model = Yolo(weights_path=Path(__file__).parent.parent / "model" / "yolov5nu.pt")
    for image_path in tqdm((dataset_path / "valid" / "images_old_pred").glob("*")):
        predicted_bboxes = model.predict(source=image_path, conf=0.1)
        bboxes.update(predicted_bboxes)
    return bboxes

def plot_conf_matrix(tp, fp, fn):
    conf_matrix = np.array([[tp, fp], [fn, 0]])

    categories = ['Positive', 'Negative']
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(conf_matrix, cmap='Blues')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    original_bboxes = parse_dataset(DATASET_PATH)
    predicted_bboxes = parse_predicted(DATASET_PATH)

    v = Validator()
    v.fit_brightness(DATASET_PATH)
    v.fit_distance(DATASET_PATH, original_bboxes)
    new_bboxes = v.predict(DATASET_PATH, original_bboxes)

    criteria = CustomCategories()
    TP, FP, FN = v.compute_confusion_matrix(new_bboxes, predicted_bboxes, criteria, selected_category = 0)

    plot_conf_matrix(TP, FP, FN)

    for image_stem in original_bboxes:
        if len(original_bboxes[image_stem]):
            image_for_debug = (DATASET_PATH / 'valid' / 'images_old_pred').glob(f'{image_stem}.*').__next__()
            fig, axs = plt.subplots()

            image = Image.open(image_for_debug)
            width, height = image.size

            axs.imshow(image, cmap='gray')
            axs.set_title(image_stem)
            # Draw predicted bboxes
            for idx, bbox in enumerate(original_bboxes[image_stem]):
                if isinstance(bbox, BboxCWH):
                    cx, cy = (bbox.bbox[0] * width, bbox.bbox[1] * height)
                    b_width, b_height = (bbox.bbox[2] * width, bbox.bbox[3] * height)
                    axs.add_patch(Rectangle((cx - b_width // 2, cy + b_height // 2 - b_height), width=b_width, height=b_height, edgecolor='red', fill=False))
                    axs.text(cx - b_width // 2 - 100,  cy + b_height // 2 + 50, new_bboxes[image_stem][idx][1], color='red', fontsize=6)
                elif isinstance(bbox, BboxXYXY):
                    lup = (bbox.bbox[0] * width, bbox.bbox[1] * height)
                    rdp = (bbox.bbox[2] * width, bbox.bbox[3] * height)
                    b_width, b_height = rdp[0] - lup[0], rdp[1] - lup[1]
                    axs.add_patch(Rectangle((lup[0], lup[1]), width=b_width, height=b_height, edgecolor = 'red', fill = False))
                    # axs.text(lup[0] + 100, lup[1] + 100, new_bboxes[image_stem][1])

                elif isinstance(bbox, BboxPointBasedOBB):
                    ...

            for idx, bbox in enumerate(predicted_bboxes[image_stem]):
                if isinstance(bbox, BboxCWH):
                    cx, cy = (bbox.bbox[0] * width, bbox.bbox[1] * height)
                    b_width, b_height = (bbox.bbox[2] * width, bbox.bbox[3] * height)
                    axs.add_patch(Rectangle((cx - b_width // 2, cy + b_height // 2 - b_height), width=b_width, height=b_height, edgecolor='red', fill=False))
                elif isinstance(bbox, BboxXYXY):
                    lup = (bbox.bbox[0] * width, bbox.bbox[1] * height)
                    rdp = (bbox.bbox[2] * width, bbox.bbox[3] * height)
                    b_width, b_height = rdp[0] - lup[0], rdp[1] - lup[1]
                    axs.add_patch(Rectangle((lup[0], lup[1]), width=b_width, height=b_height, edgecolor = 'green', fill = False))
                    # axs.text(lup[0] + 100, lup[1] + 100, new_bboxes[image_stem][1])

                elif isinstance(bbox, BboxPointBasedOBB):
                    ...
            plt.show()




if __name__ == "__main__":
    main()
