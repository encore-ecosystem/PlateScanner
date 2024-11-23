from PIL import Image
from matplotlib.patches import Polygon
from scripts.validate import get_target_bboxes, get_predicted_bboxes
from src.model import Yolo, YoloOBB
from src.utils import get_model_cli, plot_conf_matrix
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
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

        model_path = get_model_cli()
        model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

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
            print(f"Confusion matrix with criteria {criteria} saved.")

            #
            # Choose sample images
            #
            while True:
                num = input("Enter num of image samples to save [default=all]: ")
                if len(num) == 0:
                    num = len(original_bboxes)
                elif num.isdigit():
                    num = int(num)
                else:
                    print("Invalid input")
                    continue
                break
            images_samples = random.choices(list(original_bboxes.keys()), k=num)

            #
            # Save sample images
            #
            for image_stem in tqdm(images_samples, desc="Saving validation images"):
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
