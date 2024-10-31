from src.model import YoloV5
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import os


def main():
    # Parse args
    parser = argparse.ArgumentParser(
        prog='PlateScanner',
        description='<todo>',
        epilog='<todo>')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-m', '--model'      , type=str,   default='yolov5nu')
    parser.add_argument('-c', '--confidence' , type=float, default=0.06)

    args = parser.parse_args()

    # Validate and prepare args
    root = Path().resolve()

    input_dir = Path(args.input_dir).resolve()
    assert input_dir.exists() and input_dir.is_dir(), f"Unable to find input directory {input_dir}"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    model = args.model.lstrip('.pt')
    assert f"{model.lower()}.pt" in os.listdir(root / 'model'), f"Unable to find model {model}"

    # Computation
    model = YoloV5(root / 'model' / f"{model}.pt")


    for img_path in tqdm(os.listdir(input_dir), desc='Prediction'):
        results = model.predict(
            source     = (input_dir / img_path).__str__(),
            conf       = args.confidence,
            line_width = None,
        )
        image = cv2.imread(str(input_dir / img_path))
        for box in results:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (205, 0, 255), 3)
            cv2.putText(
                image,
                # parse class id from yaml
                f"Plate: {score:.2f}",
                (int(x1 - 15), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite((output_dir / img_path).__str__(), image)


if __name__ == '__main__':
    main()
