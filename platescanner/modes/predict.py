from paddleocr import PaddleOCR

from platescanner import DEFAULT_CONFIDENCE_LEVEL
from platescanner.model import Yolo, YoloOBB
from platescanner.utils import handle_path, handle_confidence_level, draw_bbox, align_license_plate, \
    preprocess_license_plate, preprocess_license_plate_without_aug

from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
from platescanner.utils.draw_bbox import draw_bbox


def mode(args):
    config = {
        '-dataset_path'     : None,
        '-weights_path'     : None,
        '-output_path'      : None,
        '-confidence_level' : DEFAULT_CONFIDENCE_LEVEL,
        '-detection_only'   : False,
    }

    # parse args
    args.pop(0)
    current = 0
    while current < len(args):
        match args[current]:
            case '-dataset_path':
                config['-dataset_path'] = handle_path(args[current + 1])
                current += 1
            case '-output_path':
                config['-output_path'] = handle_path(args[current + 1])
                current += 1
            case '-confidence_level':
                config['-confidence_level'] = handle_confidence_level(args[current + 1])
                current += 1
            case '-weights_path':
                config['-weights_path'] = handle_path(args[current + 1])
                current += 1
            case '-detection_only':
                assert current + 1 < len(args), "Please, provide a boolean for this flag"
                match args[current + 1].lower():
                    case 'true':
                        config['-detection_only'] = True
                    case 'false':
                        config['-detection_only'] = False
                    case _:
                        raise ValueError("Invalid value for '-detection_only'")
                current += 1
            case _:
                print(f"Unknown argument: {args[current]}")
                exit(-1)
        current += 1

    # check args
    if config['-dataset_path'] is None:
        print("Please, specify the dataset path")
        exit(-1)

    if config['-output_path'] is None:
        print("Please, specify the output path")
        exit(-1)

    if config['-weights_path'] is None:
        print("Please, specify the weights path")
        exit(-1)

    run(config)


def run(config: dict):
    # run
    input_path = config['-dataset_path']
    output_path = config['-output_path']
    confidence_level = config['-confidence_level']

    # Computation
    model_path = config['-weights_path']
    model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

    for img_abs_path in tqdm(list((input_path / 'test' / 'images').glob("*")), desc="Prediction"):
        curr_bboxes = model.predict(
            source=img_abs_path.__str__(),
            conf=confidence_level,
            line_width=None,
        )

        image = Image.open(img_abs_path)
        width, height = image.size

        fig, axs = plt.subplots()
        axs.imshow(image, cmap='gray')
        axs.axis('off')
        fig.patch.set_visible(False)

        for idx, bbox in enumerate(curr_bboxes[img_abs_path.stem]):
            plate_image = bbox.crop_on(image)
            aligned = align_license_plate(plate_image)

            preprocessed_plate = preprocess_license_plate(aligned)
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                det_db_box_thresh=0.4,
                rec_algorithm='SVTR_LCNet',
                use_space_char=False,
            )

            text = "NOT RECOGNIZED"
            try:
                # Используем cls=True для лучшего распознавания
                result = ocr.ocr(preprocessed_plate, cls=True)
                if result and result[0]:
                    text = "".join([line[1][0] for line in result[0]])
                else:
                    preprocessed_plate = preprocess_license_plate_without_aug(aligned)
                    result = ocr.ocr(preprocessed_plate, cls=True)
                    if result and result[0]:
                        text = "".join([line[1][0] for line in result[0]])
            except IndexError:
                pass

            draw_bbox(
                axs  = axs,
                bbox = bbox.to_image_scale(width, height),
                text = text,
            )

        plt.savefig(output_path / f"{img_abs_path.stem}.png", dpi=300)
        plt.close(fig)


__all__ = [
    'mode'
]
