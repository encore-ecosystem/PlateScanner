from platescanner.validator.stat import Validator
from platescanner.utils import handle_path
from platescanner import PLATESCANNER_ROOT_PATH

import pickle


def mode(args):
    config = {
        '-dataset_path' : None,
    }
    # parse args
    args.pop(0)
    current = 0
    while current < len(args):
        match args[current]:
            case '-dataset_path':
                path = handle_path(args[current + 1])
                if not path.exists():
                    print(f"Calibration dataset path does not exist: {path}")
                    exit(-1)

                config['-dataset_path'] = path
                current += 1
            case _:
                print(f"Unknown argument: {args[current]}")
                exit(-1)
        current += 1

    # check args
    if config['-dataset_path'] is None:
        print("Please, specify -dataset_path")
        exit(-1)

    v = Validator()
    v.fit_brightness(config['-dataset_path'])

    with open(PLATESCANNER_ROOT_PATH / 'pretrained_validator.pickle', 'wb') as f:
        pickle.dump(v, f)


__all__ = [
    'mode'
]
