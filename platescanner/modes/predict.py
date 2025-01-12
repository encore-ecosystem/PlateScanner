from platescanner import DEFAULT_CONFIDENCE_LEVEL
from platescanner.model import Yolo, YoloOBB
from platescanner.utils import handle_path, handle_confidence_level, draw_bbox

from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import os

from platescanner.utils.draw_bbox import draw_bbox


def mode(args):
    config = {
        '-dataset-path'     : None,
        '-weights-path'     : None,
        '-output-path'      : None,
        '-confidence-level' : DEFAULT_CONFIDENCE_LEVEL,
    }

    # parse args
    args.pop(0)
    current = 0
    while current < len(args):
        match args[current]:
            case '-dataset-path':
                config['-dataset-path'] = handle_path(args[current + 1])
                current += 1
            case '-output-path':
                config['-output-path'] = handle_path(args[current + 1])
                current += 1
            case '-confidence_level':
                config['-confidence_level'] = handle_confidence_level(args[current + 1])
                current += 1
            case '-weights-path':
                config['-weights-path'] = handle_path(args[current + 1])
                current += 1
            case _:
                print(f"Unknown argument: {args[current]}")
                exit(-1)
        current += 1

    # check args
    if config['-dataset-path'] is None:
        print("Please, specify the dataset path")
        exit(-1)

    if config['-output-path'] is None:
        print("Please, specify the output path")
        exit(-1)

    if config['-weights-path'] is None:
        print("Please, specify the weights path")
        exit(-1)

    input_path = config['-dataset-path']
    output_path = config['-output-path']
    confidence_level = config['-confidence_level']

    # Computation
    model_path = config['-weights-path']
    model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

    bboxes = {}
    for img_path in tqdm(os.listdir(input_path / 'test' / 'images'), desc="Prediction"):
        curr_bboxes = model.predict(
            source=(input_path / 'test' / 'images' / img_path).__str__(),
            conf=confidence_level,
            line_width=None,
        )
        bboxes.update(curr_bboxes)

    for img_stem in tqdm(bboxes, desc="Drawing bboxes on images"):
        image = Image.open((input_path / 'test' / 'images').glob(f'{img_stem}.*').__next__())
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


__all__ = [
    'mode'
]
