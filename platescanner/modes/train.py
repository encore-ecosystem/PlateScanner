from src.utils import handle_path
from src.model import Yolo

import albumentations

def mode(args):
    config = {
        '-weights_path' : None,
        '-dataset_path' : None,
        '-use-clearml'  : False,
    }

    # parse args
    args.pop(0)
    current = 0
    while current < len(args):
        match args[current]:
            case '-weights_path':
                path = handle_path(args[current + 1])
                if not path.exists():
                    print(f"Weights path does not exist: {path}")
                    exit(-1)

                config['-weights_path'] = args[current + 1]
                current += 1
            case '-dataset_path':
                path = handle_path(args[current + 1])
                if not path.exists():
                    print(f"Dataset path does not exist: {path}")
                    exit(-1)

                config['-dataset_path'] = args[current + 1]
                current += 1
            case '-use-clearml':
                config['-use-clearml'] = True
            case _:
                print(f"Unknown argument: {args[current]}")
                exit(-1)
        current += 1

    # validate args
    if config['-weights_path'] is None:
        print("Please, specify weights path")
        exit(-1)
    if config['-dataset_path'] is None:
        print("Please, specify dataset path")
        exit(-1)

    # run
    Yolo(
        weights_path = config['-weights_path']
    ).fit(
        dataset_path = config['-dataset_path'],
        use_clearml  = config['-use-clearml'],
        augmentation = albumentations.Compose(
            transforms = [
                        albumentations.RandomBrightnessContrast(p=0.5,brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4)),
                        albumentations.RandomRain(p=0.5, brightness_coefficient=1),
                        albumentations.RandomSnow(p=0.5),
                        albumentations.ISONoise(p=0.5),
                        albumentations.CLAHE(p=1.0),
                        albumentations.ToGray(p=1.0),
            ],
            bbox_params = albumentations.BboxParams(format="yolo")), # strict=False
    )
