from matplotlib.patches import Polygon

from src import DEFAULT_CONFIDENCE_LEVEL, BBOX_EDGE_COLOR, BBOX_TEXT_COLOR, BBOX_TEXT_FONTSIZE, \
    BBOX_TEXT_HORIZONTAL_SHIFT, BBOX_TEXT_VERTICAL_SHIFT
from src.model import Yolo, YoloOBB
from src.utils import get_model_cli

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import os

from src.utils.draw_bbox import draw_bbox


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
                draw_bbox(
                    axs  = axs,
                    bbox = bbox.to_image_scale(width, height),
                    text = f"{bbox.confidence:.2f}",
                )

            plt.savefig(output_path / f"{img_stem}.png", dpi=300)
            plt.close(fig)

    except KeyboardInterrupt:
        print('Returning back.')


__all__ = [
    'view'
]
