import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.model import Yolo
from src.bbox import BboxCWH, BboxPointBasedOBB
from pathlib import Path
from tqdm import tqdm
from PIL import Image

#
# VALIDATE
#
DATASET_PATH = Path(__file__).parent.parent.resolve() / 'dataset' / 'detection' / 'validation_dataset'

def parse_dataset(dataset_path: Path):
    bboxes = {}
    for label_name in tqdm((dataset_path / 'valid' / 'labels').glob('*.txt')):
        with open(dataset_path / 'valid' / 'labels' / label_name, 'r') as f:
            label_name = label_name.stem
            for bbox_data in f.readlines():
                bbox_data = bbox_data.split()
                category = int(bbox_data.pop(0))
                points   = tuple(map(float, bbox_data))
                match len(points):
                    # non-obb bbox
                    case 4:
                        bboxes[label_name] = bboxes.get(label_name, []) + [BboxCWH(*points, category=category)]
                    # point-based obb bbox
                    case 8:
                        bboxes[label_name] = bboxes.get(label_name, []) + [BboxPointBasedOBB(points, category=category)]
    return bboxes


def parse_predicted(dataset_path: Path):
    bboxes = {}
    model  = Yolo(weights_path=Path(__file__).parent.parent / 'model' / 'yolov5nu.pt')
    for image_path in tqdm((dataset_path / 'valid' / 'images').glob('*')):
        predicted_bboxes = model.predict(source=image_path, conf = 0.1)
        bboxes.update(predicted_bboxes)
    return bboxes


def main():
    original_bboxes  = parse_dataset(DATASET_PATH)
    predicted_bboxes = parse_predicted(DATASET_PATH)

    for image_stem in original_bboxes:
        if len(original_bboxes[image_stem]) > 0 and len(predicted_bboxes[image_stem]) > 0:
            image_for_debug = (DATASET_PATH / 'valid' / 'images').glob(f'{image_stem}.*').__next__()
            fig, axs = plt.subplots()

            image = Image.open(image_for_debug)
            width, height = image.size

            axs.imshow(image, cmap='gray')

            # Draw predicted bboxes
            for bbox in predicted_bboxes[image_stem]:
                lup = (bbox.bbox[0] * width, bbox.bbox[1] * height)
                rdp = (bbox.bbox[2] * width, bbox.bbox[3] * height)
                b_width, b_height = rdp[0] - lup[0], rdp[1] - lup[1]
                axs.add_patch(Rectangle((lup[0], lup[1]), width=b_width, height=b_height, edgecolor = 'red', fill = False))
            plt.show()

if __name__ == '__main__':
    main()
