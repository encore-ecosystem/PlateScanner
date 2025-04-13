from typing import Optional

import cvtk
from PIL import Image

from platescanner.utils import get_target_bboxes, get_predicted_bboxes, handle_path, plot_conf_matrix, \
    preprocess_license_plate, RecognitionModel
from platescanner import DEFAULT_CONFIDENCE_LEVEL, PLATESCANNER_ROOT_PATH
from platescanner.models import Yolo, YoloOBB
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from platescanner.utils.draw_bbox import draw_bbox
from platescanner.utils.recognition_metrics import evaluate_metrics
from platescanner.validator.criteria import CustomCriteria, Distance, Time
from platescanner.validator.stat import Validator

from cvtk.utils.determinator import determine_dataset
from cvtk.supported_datasets.mvp import MVP_Dataset

import pickle


def mode(args):
    config = {
        '-dataset_path'     : None,
        '-output_path'      : None,
        '-weights_path'     : None,
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
            case '-weights_path':
                config['-weights_path'] = handle_path(args[current + 1])
                current += 1
            case '-detection_only':
                if current + 1 >= len(args):
                    print("Please, specify a detection-only parameter")
                    exit(-1)

                match args[current + 1].lower():
                    case 'true':
                        config['-detection_only'] = True
                    case 'false':
                        config['-detection_only'] = False
                    case _:
                        print("Please, specify a detection-only parameter")
                        exit(-1)
                current += 1


            case _:
                print(f"Unknown argument: {args[current]}")
                exit(-1)

        current += 1

    # validate args
    if config['-dataset_path'] is None:
        print("Missing dataset path.")
        exit(-1)

    if config['-output_path'] is None:
        print("Missing output path.")
        exit(-1)

    run(config)

def run(config: dict):
    match config.get('-detection_only'):
        case True:
            only_detection(config)
        case False:
            only_recognition(config)
        case None:
            overall_pipeline(config)


def only_detection(config: dict):
    # run
    input_path = Path(config['-dataset_path'])
    output_path = Path(config['-output_path'])

    for filtered_predicted_bboxes, filtered_original_bboxes, images_samples, _ in detect_bboxes(config):
        for image_stem in tqdm(images_samples, desc="Saving validation images"):
            image = Image.open((input_path / 'valid' / 'images').glob(f'{image_stem}.*').__next__())
            width, height = image.size

            fig, axs = plt.subplots()
            axs.imshow(image, cmap='gray')
            axs.axis('off')
            fig.patch.set_visible(False)

            for bbox, criteria in filtered_original_bboxes.get(image_stem, []):
                draw_bbox(
                    axs=axs,
                    bbox=bbox.to_image_scale(width, height),
                    text=criteria.__repr__(),
                    text_h_shift=int(-0.05 * width),
                    text_v_shift=int(0.05 * height),
                    text_color='red',
                    edge_color='red',
                )

            for bbox, criteria in filtered_predicted_bboxes.get(image_stem, []):
                draw_bbox(
                    axs=axs,
                    bbox=bbox.to_image_scale(width, height),
                    text=f"{criteria.__repr__()} {bbox.confidence:.2f}",
                    text_h_shift=int(-0.08 * width),
                    text_v_shift=int(-0.01 * height),
                    text_color='green',
                    edge_color='green',
                )

            plt.savefig(output_path / f"{image_stem}.png", dpi=300)
            plt.close(fig)

def overall_pipeline(config: dict):
    input_path = Path(config['-dataset_path'])
    output_path = Path(config['-output_path'])

    recognized_text_all_images = {}
    ground_truth_text_all_images = {}

    rec_model = RecognitionModel()
    for filtered_predicted_bboxes, filtered_original_bboxes, images_samples, fn in detect_bboxes(config):
        for image_stem in tqdm(images_samples, desc="Processing images with recognition"):
            img_abs_path = (input_path / 'valid' / 'images').glob(f"{image_stem}.*").__next__()
            image = Image.open(img_abs_path)
            width, height = image.size

            fig, axs = plt.subplots()
            axs.imshow(image, cmap='gray')
            axs.axis('off')
            fig.patch.set_visible(False)

            for idx, bbox in enumerate(filtered_predicted_bboxes[image_stem]):
                plate_image = bbox[0].crop_on(image)
                preprocessed_plate = preprocess_license_plate(plate_image)
                recognized_text, raw_output = rec_model("parseq", preprocessed_plate)

                recognized_text_all_images[image_stem] = recognized_text_all_images.get(image_stem, []) + [(bbox[0], recognized_text)]

                draw_bbox(
                        axs=axs,
                        bbox=bbox[0].to_image_scale(width, height),
                        text=f"{recognized_text}",
                        text_h_shift=int(-0.05 * width),
                        text_v_shift=int(-0.01 * height),
                        text_color='green',
                        edge_color='green',
                )

            # DRAW GT BBOXES
            for bbox, criteria, text in filtered_original_bboxes.get(image_stem, []):
                ground_truth_text_all_images[image_stem] = ground_truth_text_all_images.get(image_stem, []) + [(bbox, text)]
                draw_bbox(
                        axs=axs,
                        bbox=bbox.to_image_scale(width, height),
                        text=text,
                        text_h_shift=int(-0.01 * width),
                        text_v_shift=int(0.05 * height),
                        text_color='red',
                        edge_color='red',
                )
            # SAVE IMAGES WITH RECOGNIZED TEXT
            plt.savefig(output_path / f"{image_stem}.png", dpi=300)
            plt.close(fig)


        lev_scores, business_scores = evaluate_metrics(ground_truth_text_all_images, recognized_text_all_images)

        total_gt_boxes = 0
        valid_gt_boxes = 0
        total_business_score = 0

        for image_stem, gt_boxes in ground_truth_text_all_images.items():
            for _, gt_text in gt_boxes:
                total_gt_boxes += 1
                if gt_text is not None:
                    valid_gt_boxes += 1

        for image_stem, score in business_scores.items():
            if ground_truth_text_all_images[image_stem][0][1] is not None:
                total_business_score += score

        n = 0
        for x in filtered_original_bboxes.values():
            n += len(x)
        print(f"Business score: {total_business_score / valid_gt_boxes:.4f}")
        print(f"Mean Average Levenshtein: {sum(lev_scores.values()) / len(lev_scores.values()):.4f}")


def only_recognition(config: dict):
    input_path = Path(config['-dataset_path'])
    output_path = Path(config['-output_path'])

    # 0. Convert to MVP
    recognized_text_all_images = {}
    ground_truth_text_all_images = {}

    # 1. get target bboxes
    mvp_dataset = cvtk.MVP_Dataset.read(input_path)
    valid_split = 'valid'
    curr_bboxes = {}
    rec_model = RecognitionModel()
    for image_stem in mvp_dataset.attributes[valid_split]:
        curr_bboxes[image_stem] = []
        for bbox_data in mvp_dataset.attributes[valid_split][image_stem]['Detection']['bboxes']:
            bbox_type = bbox_data['bbox_type']
            bbox_points = bbox_data['points']
            bbox_text = bbox_data['text'] if 'text' in bbox_data else bbox_data['recognition_text']

            bbox = None
            match bbox_type:
                case "bb":
                    bbox = cvtk.Bbox_CWH(bbox_points)

            if bbox is None:
                raise ValueError(f"Bbox type {bbox_type} not supported yet")

            curr_bboxes[image_stem].append((bbox, bbox_text))

        image = Image.open((input_path / 'valid' / 'images').glob(f"{image_stem}.*").__next__())
        width, height = image.size
        fig, axs = plt.subplots()
        axs.imshow(image, cmap='gray')
        axs.axis('off')
        fig.patch.set_visible(False)

        for idx, (bbox, bbox_text) in enumerate(curr_bboxes[image_stem]):
            plate_image = bbox.crop_on(image)
            preprocessed_plate = preprocess_license_plate(plate_image)
            recognized_text, raw_output = rec_model("parseq", preprocessed_plate)
            ground_truth_text_all_images[image_stem] = ground_truth_text_all_images.get(image_stem, []) + [
                (bbox, bbox_text)]
            recognized_text_all_images[image_stem] = recognized_text_all_images.get(image_stem, []) + [
                (bbox, recognized_text)]
            color = 'green' if recognized_text == bbox_text else 'red'

            draw_bbox(
                axs=axs,
                bbox=bbox.to_image_scale(width, height),
                text=f"R: {recognized_text} | E: {bbox_text}",
                text_h_shift=int(-0.05 * width),
                text_v_shift=int(-0.01 * height),
                text_color=color,
                edge_color=color,
            )
        # SAVE IMAGES WITH RECOGNIZED TEXT
        plt.savefig(output_path / f"{image_stem}.png", dpi=300)
        plt.close(fig)

    lev_scores, business_scores = evaluate_metrics(ground_truth_text_all_images, recognized_text_all_images)

    total_gt_boxes = 0
    valid_gt_boxes = 0
    total_business_score = 0

    for image_stem, gt_boxes in ground_truth_text_all_images.items():
        for _, gt_text in gt_boxes:
            total_gt_boxes += 1
            if gt_text is not None:
                valid_gt_boxes += 1

    for image_stem, score in business_scores.items():
        if ground_truth_text_all_images[image_stem][0][1] is not None:
            total_business_score += score

    # print("Levenshtein scores:", lev_scores)
    # print("Business scores:", business_scores)
    print(f"Mean Average Levenshtein: {sum(lev_scores.values()) / len(lev_scores.values()):.4f}")
    print(f"Business score: {total_business_score / valid_gt_boxes:.4f}")

def detect_bboxes(config: dict):
    # run
    input_path = Path(config['-dataset_path'])
    output_path = Path(config['-output_path'])
    confidence_level = config['-confidence_level']
    model_path = config['-weights_path']

    model = (YoloOBB if 'obb' in model_path.stem else Yolo)(model_path)

    original_bboxes = get_target_bboxes(input_path)
    predicted_bboxes = get_predicted_bboxes(input_path, model, confidence_level)
    with open(PLATESCANNER_ROOT_PATH / 'pretrained_validator.pickle', 'rb') as f:
        v: Validator = pickle.load(f)
    v.fit_distance(input_path, original_bboxes)

    classified_original_bboxes = v.predict(input_path, original_bboxes)
    classified_predicted_bboxes = v.predict(input_path, predicted_bboxes)

    original_dataset = determine_dataset(input_path)
    original_bboxes_with_text = None
    if isinstance(original_dataset, MVP_Dataset):
        original_bboxes_with_text = original_dataset.attributes
    classified_original_bboxes = add_rec_text_to_bboxes(classified_original_bboxes, original_bboxes_with_text)

    while True:
        print('=' * 64)
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
        print(f"Confusion matrix with criteria <{criteria}> saved.")

        #
        # Choose sample images
        #
        while True:
            num = input("Enter num of image samples to save [default=all]: ")
            if len(num) == 0:
                num = len(original_bboxes)
            elif num.isdigit() and int(num) > 0:
                num = min(int(num), len(original_bboxes))
            else:
                print("Invalid input")
                continue
            break
        images_samples = list(original_bboxes.keys())[:num]

        yield filtered_predicted_bboxes, filtered_original_bboxes, images_samples, FN


def add_rec_text_to_bboxes(bboxes_without_text: dict, bboxes_with_text: Optional[dict]) -> dict:
    if bboxes_with_text is None:
        return bboxes_without_text

    for image_stem, bboxes in bboxes_with_text['valid'].items():
        for idx in range(len(bboxes['Detection']['bboxes'])):
            bboxes_without_text[image_stem][idx] += (bboxes_with_text['valid'][image_stem]['Detection']['bboxes'][idx]['recognition_text'],)

    return bboxes_without_text


__all__ = [
    'mode'
]
