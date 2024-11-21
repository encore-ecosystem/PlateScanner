from PIL import Image
from matplotlib.patches import Polygon

from src.model import Yolo, YoloOBB, YoloBase
from matplotlib import pyplot as plt
from pathlib import Path
from src.bbox import *
from tqdm import tqdm

import numpy as np

from src.validator.criteria import CustomCriteria, Distance, Time
from src.validator.stat import Validator


CALIBRATION_DATASET = Path(__file__).parent.parent.parent.resolve() / 'calibration'
models = {
    'yolo11n-obb'     : True,
    'yolo11x'         : False,
    'yolo11x-overfit' : False,
    'yolov5nu'        : False,
}


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
                            BboxCWH(*points, category=category)
                        ]
                    # point-based obb bbox
                    case 8:
                        bboxes[label_name] = bboxes.get(label_name, []) + [
                            BboxPointBasedOBB(points, category=category)
                        ]
    return bboxes


def plot_conf_matrix(tp: int, fp: int, fn: int, output_path: Path, title: str):
    conf_matrix = np.array([[tp, fn], [fp, 0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(conf_matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    # Подсказки
    plt.text(-1, 0, "TP", fontsize=14,     color="black", va="center")
    plt.text(-1, 1, "FP", fontsize=14,     color="black", va="center")
    plt.text(1.6, 0, "FN", fontsize=14, color="black", va="center")
    plt.text(1.6, 1, "TN", fontsize=14, color="black", va="center")

    plt.tight_layout()
    plt.savefig(output_path / f"0_ConfusionMatrix.png")



def view():
    try:
        while True:
            input_path = Path(input('Enter input directory path: ')).resolve()
            if not input_path.exists():
                print('Input directory does not exist')
                continue
            if not input_path.is_dir():
                print('Input directory is not a directory')
                continue
            break

        while True:
            output_path = Path(input('Enter output directory path: ')).resolve()
            if not output_path.exists():
                print('Output directory does not exist')
                continue
            if not output_path.is_dir():
                print('Output directory is not a directory')
                continue
            break

        while True:
            print('Please, choose a model:')
            for model in models:
                print(f"\t{model}")
            model = input('[yolov5nu] >> ')
            model = 'yolov5nu' if len(model) == 0 else model
            if not model in models:
                print('Invalid model name')
                continue
            break

        root = Path(__file__).parent.parent.parent.resolve()
        model = (YoloOBB if models[model] else Yolo)(root / 'model' / f"{model}.pt")
        original_bboxes  = get_target_bboxes(input_path)
        predicted_bboxes = get_predicted_bboxes(input_path, model)

        v = Validator()
        v.fit_brightness(CALIBRATION_DATASET)
        v.fit_distance(input_path, original_bboxes)

        classified_original_bboxes  = v.predict(input_path, original_bboxes)
        classified_predicted_bboxes = v.predict(input_path, predicted_bboxes)

        while True:
            print('='*64)
            print("Choose the distance criteria:")
            print("0) Close\n1) Middle\n2) FAR")
            distance = input("[any] >> ")
            if distance.isdigit() and int(distance) not in (0, 1, 2):
                print("Invalid mode")
                continue
            distance = None if len(distance) == 0 else (Distance.CLOSE, Distance.MIDDLE, Distance.FAR)[int(distance)]

            print("Choose the daytime criteria:")
            print("0) Day\n1) Night")
            daytime = input("[any] >> ")
            if daytime.isdigit() and int(daytime) not in (0, 1):
                print("Invalid mode")
                continue
            daytime = None if len(daytime) == 0 else (Time.DAY, Time.NIGHT)[int(daytime)]

            criteria = CustomCriteria()
            criteria.distance = distance
            criteria.time = daytime
            filtered_predicted_bboxes, filtered_original_bboxes, TP, FP, FN = v.compute_confusion_matrix(
                classified_original_bboxes,
                classified_predicted_bboxes,
                criteria,
                selected_category=0
            )

            plot_conf_matrix(TP, FP, FN, output_path, criteria.__repr__())
            print(f"Confusion matrix with criteria {criteria} saved.")

            for image_stem in tqdm(original_bboxes, desc="Saving validation images"):
                image_for_debug = (input_path / 'valid' / 'images').glob(f'{image_stem}.*').__next__()
                fig, axs = plt.subplots()

                image = Image.open(image_for_debug)
                width, height = image.size
                axs.imshow(image, cmap='gray')
                axs.axis('off')
                fig.patch.set_visible(False)

                for source, color in [(filtered_original_bboxes.get(image_stem, []), 'red'),
                                      (filtered_predicted_bboxes.get(image_stem, []), 'green')]:
                    for idx, (bbox, criteria) in enumerate(source):
                        polygone = [[int(point[0] * width), int(point[1] * height)] for point in bbox.get_poly()]
                        axs.add_patch(Polygon(polygone, fill=False, edgecolor=color))
                        point = polygone[0]

                        if color == 'green':
                            axs.text(point[0] - 200, point[1] + 75, s=criteria.__repr__(), color=color, fontsize=4)
                        elif color == 'red':
                            axs.text(point[0] - 125, point[1] - 20, s=criteria.__repr__(), color=color, fontsize=4)

                plt.savefig(output_path / f"{image_stem}.png", dpi=300)
                plt.close(fig)


    except KeyboardInterrupt:
        print('Returning back.')
