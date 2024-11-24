from scripts.validate import get_target_bboxes, get_predicted_bboxes
from src.model import YoloOBB, Yolo
from src.utils import get_model_cli
from src.validator.criteria import CustomCriteria
from src.validator.stat import Validator
from src import PROJECT_ROOT_PATH
from tqdm import tqdm

CALIBRATION_DATASET = PROJECT_ROOT_PATH / 'calibration'
TARGET_PRECISION = 0.95

def main():
    model_path = get_model_cli()
    model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)
    input_path = PROJECT_ROOT_PATH / "dataset" / f"target_pictures_{"obb" if 'obb' in model_path.stem else "bb"}"

    original_bboxes  = get_target_bboxes(input_path)
    v = Validator()
    v.fit_brightness(CALIBRATION_DATASET)
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
        if precision < TARGET_PRECISION:
            lower = mid
        elif precision > TARGET_PRECISION:
            upper = mid
        else:
            break
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(f"confidence: {((lower + upper) / 2):.2f} TP: {TP} FN: {FN} FP: {FP} Precision: {precision:.2f} Recall: {recall:.2f} F1-Score: {2*precision * recall / (precision + recall):.2f}")

if __name__ == '__main__':
    main()