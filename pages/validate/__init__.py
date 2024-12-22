from PIL import Image
from matplotlib.patches import Polygon
from scripts.validate import get_target_bboxes, get_predicted_bboxes
from src import DEFAULT_CONFIDENCE_LEVEL
from src.model import Yolo, YoloOBB
from src.utils import get_model_cli, plot_conf_matrix
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.utils.draw_bbox import draw_bbox
from src.validator.criteria import CustomCriteria, Distance, Time
from src.validator.stat import Validator
import random

CALIBRATION_DATASET = Path(__file__).parent.parent.parent.resolve() / 'calibration'


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
            confidence_level = input(f'Enter a confidence level in percent [default={int(DEFAULT_CONFIDENCE_LEVEL * 100)}]: ')
            if not (len(confidence_level) == 0 or confidence_level.isdigit()):
                print("Invalid confidence level percent.")
                continue
            confidence_level = DEFAULT_CONFIDENCE_LEVEL if len(confidence_level) == 0 else (int(confidence_level) / 100)
            break

        model_path = get_model_cli()
        model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

        original_bboxes  = get_target_bboxes(input_path)
        predicted_bboxes = get_predicted_bboxes(input_path, model, confidence_level)
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

            #
            # Fetch criteria
            #
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
            print(f"Confusion matrix with criteria <{criteria}> saved.")

            #
            # Choose sample images
            #
            while True:
                num = input("Enter num of image samples to save [default=all]: ")
                if len(num) == 0:
                    num = len(original_bboxes)
                elif num.isdigit() and int(num) > 0:
                    num = min(int(num), len(original_bboxes))
                else:
                    print("Invalid input")
                    continue
                break
            images_samples = list(original_bboxes.keys())[:num]

            #
            # Save sample images
            #
            for image_stem in tqdm(images_samples, desc="Saving validation images"):
                image = Image.open((input_path / 'valid' / 'images').glob(f'{image_stem}.*').__next__())
                width, height = image.size

                fig, axs = plt.subplots()
                axs.imshow(image, cmap='gray')
                axs.axis('off')
                fig.patch.set_visible(False)

                for bbox, criteria in filtered_original_bboxes.get(image_stem, []):
                    draw_bbox(
                        axs          = axs,
                        bbox         = bbox.to_image_scale(width, height),
                        text         = criteria.__repr__(),
                        text_h_shift = int(-0.05 * width),
                        text_v_shift = int(0.05 * height),
                        text_color   = 'red',
                        edge_color   = 'red',
                    )

                for bbox, criteria in filtered_predicted_bboxes.get(image_stem, []):
                    draw_bbox(
                        axs          = axs,
                        bbox         = bbox.to_image_scale(width, height),
                        text         = f"{criteria.__repr__()} {bbox.confidence:.2f}",
                        text_h_shift = int(-0.08 * width),
                        text_v_shift = int(-0.01 * height),
                        text_color   = 'green',
                        edge_color   = 'green',
                    )

                plt.savefig(output_path / f"{image_stem}.png", dpi=300)
                plt.close(fig)


    except KeyboardInterrupt:
        print('Returning back.')


__all__ = [
    'view'
]
