import pickle

from platescanner.utils.validate_funcs import get_target_bboxes, get_predicted_bboxes
from platescanner.model import YoloOBB, Yolo
from platescanner.utils import handle_path
from platescanner.validator.criteria import CustomCriteria
from platescanner.validator.stat import Validator
from platescanner import PROJECT_ROOT_PATH
from tqdm import tqdm


def mode(args):
    config = {
        '-weights_path'     : None,
        '-dataset_path'     : None,
        '-iters'            : 7,
        '-target_precision' : 0.95,
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

                config['-weights_path'] = path
                current += 1
            case '-dataset_path':
                path = handle_path(args[current + 1])
                if not path.exists():
                    print(f"Dataset path does not exist: {path}")
                    exit(-1)

                config['-dataset_path'] = path
                current += 1
            case '-iters':
                iters = args[current + 1]
                if not iters.isdigit() or int(iters) <= 0:
                    print(f"Invalid number of iterations: {iters}")
                    exit(-1)

                config['-iters'] = int(iters)
                current += 1
            case '-target_precision':
                target_precision = args[current + 1]
                if not target_precision.replace('.', '', 1).isdigit() or float(target_precision) < 0 or float(target_precision) > 1:
                    print(f"Invalid target precision: {target_precision}")
                    exit(-1)

                config['-target_precision'] = float(target_precision)
                current += 1
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
    model_path = config['-weights_path']
    model      = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)
    input_path = config['-dataset_path']

    original_bboxes  = get_target_bboxes(input_path)

    with open(PROJECT_ROOT_PATH / 'pretrained_validator.pickle', 'rb') as f:
        v : Validator = pickle.load(f)
    v.fit_distance(input_path, original_bboxes)
    classified_original_bboxes = v.predict(input_path, original_bboxes)

    criteria = CustomCriteria()
    lower = 0
    upper = 1

    pbar = tqdm(range(7))
    TP, FP, FN = 0, 0, 0
    for _ in pbar:
        mid = (lower + upper) / 2
        predicted_bboxes = get_predicted_bboxes(input_path, model, conf=mid, use_pbar=False)
        classified_predicted_bboxes = v.predict(input_path, predicted_bboxes)
        _, _, TP, FP, FN = v.compute_confusion_matrix(
            classified_original_bboxes,
            classified_predicted_bboxes,
            criteria,
            selected_category=0
        )
        precision = TP / (TP + FP)
        pbar.set_description(f"Precision = {precision:.2f} confidence = {mid:.2f}")
        if precision < config['-target_precision']:
            lower = mid
        elif precision > config['-target_precision']:
            upper = mid
        else:
            break
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(f"confidence: {((lower + upper) / 2):.2f} TP: {TP} FN: {FN} FP: {FP} Precision: {precision:.2f} Recall: {recall:.2f} F1-Score: {2*precision * recall / (precision + recall):.2f}")
