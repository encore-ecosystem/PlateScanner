from src.model import Yolo, YoloOBB

from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import os

models = {
    'yolo11n-obb'     : True,
    'yolo11x'         : False,
    'yolo11x-overfit' : False,
    'yolov5nu'        : False,
}

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
        while True:
            print('Choose a confidence level in percent.')
            confidence_level = input('[default=6] >> ')
            if not (len(confidence_level) == 0 or confidence_level.isdigit()):
                print("Invalid confidence level percent.")
                continue
            confidence_level = 0.06 if len(confidence_level) == 0 else int(confidence_level) / 100
            break

        # Validate and prepare args
        root = Path(__file__).parent.parent.parent.resolve()

        # Computation
        model = (YoloOBB if models[model] else Yolo)(root / 'model' / f"{model}.pt")

        for img_path in tqdm(os.listdir(input_path), desc='Prediction'):
            bboxes = model.predict(
                source=(input_path / img_path).__str__(),
                conf=confidence_level,
                line_width=None,
            )
            image = cv2.imread(str(input_path / img_path))
            width, height = image.shape[:2]
            for bbox in bboxes[Path(img_path).stem]:
                polygone = [[int(point[0] * width), int(point[1] * height)] for point in bbox.get_poly()]
                pts = np.array(polygone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=False, color=(205, 0, 255), thickness=3)
                first_point = polygone[0]
                cv2.putText(
                    image,
                    # parse class id from yaml
                    f"Plate: {bbox.confidence:.2f}",
                    (int(first_point[0] - 15), int(first_point[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(205, 0, 255), thickness=3, lineType=cv2.LINE_AA)

            cv2.imwrite((output_path / img_path).__str__(), image)

    except KeyboardInterrupt:
        print('Returning back.')
