from platescanner import DEFAULT_CONFIDENCE_LEVEL
from platescanner.model import Yolo, YoloOBB
from platescanner.utils import handle_path, handle_confidence_level, preprocess_license_plate, RecognitionModel
from platescanner.utils.draw_bbox import put_info_on_image

from tqdm import tqdm
from PIL import Image


def mode(args):
    config = {
        '-dataset_path'     : None,
        '-weights_path'     : None,
        '-output_path'      : None,
        '-confidence_level' : DEFAULT_CONFIDENCE_LEVEL,
        '-detection_only'   : None,
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

    rec_model = RecognitionModel()

    for img_abs_path in tqdm(list((input_path / 'test' / 'images').glob("*")), desc="Prediction"):
        image = Image.open(img_abs_path)

        curr_bboxes = model.predict(
            source=img_abs_path.__str__(),
            conf=confidence_level,
            line_width=None,
        )

        for bboxes in curr_bboxes[img_abs_path.stem]:
            data = []
            for bbox in bboxes:
                if config['-detection_only']:
                    text = f"{bbox.confidence}"
                else:
                    plate_image = bbox.crop_on(image)
                    preprocessed_plate = preprocess_license_plate(plate_image)
                    text, _ = rec_model("parseq", preprocessed_plate)

                data.append((bbox, text))
            put_info_on_image(image, img_abs_path.stem, output_path, data)


__all__ = [
    'mode'
]
