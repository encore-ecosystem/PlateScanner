from matplotlib.patches import Polygon

from src.model import Yolo, YoloOBB
from src.utils import get_model_cli

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import os


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
            print('Choose a confidence level in percent.')
            confidence_level = input('[default=6] >> ')
            if not (len(confidence_level) == 0 or confidence_level.isdigit()):
                print("Invalid confidence level percent.")
                continue
            confidence_level = 0.06 if len(confidence_level) == 0 else int(confidence_level) / 100
            break


        # Computation
        model_path = get_model_cli()
        model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

        bboxes = {}
        for img_path in tqdm(os.listdir(input_path / 'valid' / 'images'), desc="Prediction"):
            curr_bboxes = model.predict(
                source=(input_path / 'valid' / 'images' / img_path).__str__(),
                conf=confidence_level,
                line_width=None,
            )
            bboxes.update(curr_bboxes)

        for img_stem in tqdm(bboxes, desc="Drawing bboxes on images"):
            image = Image.open((input_path / 'valid' / 'images').glob(f'{img_stem}.*').__next__())
            width, height = image.size

            fig, axs = plt.subplots()
            axs.imshow(image, cmap='gray')
            axs.axis('off')
            fig.patch.set_visible(False)

            for idx, bbox in enumerate(bboxes[img_stem]):
                polygone = [[int(point[0] * width), int(point[1] * height)] for point in bbox.get_poly()]
                axs.add_patch(Polygon(polygone, fill=False, edgecolor='red'))
                point = polygone[0]
                axs.text(point[0] - 125, point[1] - 20, s=f"{bbox.confidence:.2f}", color='red', fontsize=4)
            plt.savefig(output_path / f"{img_stem}.png", dpi=300)
            plt.close(fig)

    except KeyboardInterrupt:
        print('Returning back.')
