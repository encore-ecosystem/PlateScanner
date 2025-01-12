from bdb import Breakpoint

import cv2
import numpy as np

from pathlib import Path

from src.bbox.abstract import Bbox
from src.validator.criteria import CustomCriteria, Time, Distance
from src.utils import bbox_iou


class Validator:
    def __init__(self):
        self.brightnesses_percentile_point        = None
        self.bbox_size_ratios_percentile_point_33 = None
        self.bbox_size_ratios_percentile_point_66 = None

    def fit_brightness(self, dataset_path: Path):
        average_brightnesses = []

        for image_path in (dataset_path / 'valid' / 'images').glob("*"):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_image)
            average_brightnesses.append(mean_brightness)

        self.brightnesses_percentile_point = np.percentile(average_brightnesses, 40)

    def fit_distance(self, dataset_path: Path, bboxes: dict[str, list[Bbox]]):
        average_bbox_size_ratios = []

        for image_path in (dataset_path / 'valid' / 'images').glob("*"):
            image = cv2.imread(image_path)
            image_area = image.shape[0] * image.shape[1]
            image_bboxes = bboxes[image_path.stem]

            for bbox in image_bboxes:
                average_bbox_size_ratios.append(bbox.area() / image_area)

        self.bbox_size_ratios_percentile_point_33 = np.percentile(average_bbox_size_ratios, 33)
        self.bbox_size_ratios_percentile_point_66 = np.percentile(average_bbox_size_ratios, 66)

    def predict(self, dataset_path: Path, bboxes: dict[str, list[Bbox]]) -> dict[str, list[tuple[Bbox, CustomCriteria]]]:
        new_bboxes = {}

        for image_path in (dataset_path / 'valid' / 'images').glob("*"):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_image)

            image_area = image.shape[0] * image.shape[1]

            if mean_brightness > self.brightnesses_percentile_point:
                time = Time.DAY
            else:
                time = Time.NIGHT

            new_bboxes[image_path.stem] = new_bboxes.get(image_path.stem, [])
            for bbox in bboxes[image_path.stem]:
                custom_categories = CustomCriteria()

                custom_categories.time = time

                bbox_area_ratio = bbox.area() / image_area

                if bbox_area_ratio < self.bbox_size_ratios_percentile_point_33:
                    custom_categories.distance = Distance.FAR
                elif self.bbox_size_ratios_percentile_point_33 < bbox_area_ratio < self.bbox_size_ratios_percentile_point_66:
                    custom_categories.distance = Distance.MIDDLE
                else:
                    custom_categories.distance = Distance.CLOSE

                new_bboxes[image_path.stem] += [(bbox, custom_categories)]
        return new_bboxes

    @staticmethod
    def compute_confusion_matrix(
            original_bboxes   : dict[str, list[tuple[Bbox, CustomCriteria]]],
            predicted_bboxes  : dict[str, list[tuple[Bbox, CustomCriteria]]],
            criteria          : CustomCriteria,
            selected_category : int,
            threshold         : float = 0.4
    ) -> tuple[list[tuple[Bbox, CustomCriteria]]:2, int:3]:
        filtered_original_bboxes = {}
        filtered_predicted_bboxes = {}

        glob_TP = 0
        glob_FP = 0
        glob_FN = 0

        for image_name in original_bboxes.keys():
            filtered_original_bboxes[image_name] = filtered_original_bboxes.get(image_name, [])
            for bbox, bbox_criteria in original_bboxes[image_name]:
                if bbox.category != selected_category:
                    continue
                if bbox_criteria != criteria:
                    continue
                filtered_original_bboxes[image_name] += [(bbox, bbox_criteria)]

        for image_name in predicted_bboxes.keys():
            filtered_predicted_bboxes[image_name] = []
            for predicted_bbox, bbox_criteria in predicted_bboxes[image_name]:
                for original_bbox, original_criteria in original_bboxes[image_name]:
                    if bbox_iou(original_bbox, predicted_bbox) > threshold:
                        bbox_criteria.distance = original_criteria.distance
                        bbox_criteria.time     = original_criteria.time
                if predicted_bbox.category != selected_category:
                    continue
                if bbox_criteria != criteria:
                    continue
                filtered_predicted_bboxes[image_name] += [(predicted_bbox, bbox_criteria)]

        for image_name in filtered_predicted_bboxes.keys():
            classified_bboxes = set()

            TP_counter = 0
            for predicted_bbox, _ in filtered_predicted_bboxes[image_name]:
                flag = False
                flag_orig = None
                for original_bbox, _ in filtered_original_bboxes[image_name]:
                    if bbox_iou(original_bbox, predicted_bbox) > threshold and \
                        original_bbox not in classified_bboxes and \
                        predicted_bbox not in classified_bboxes:
                        flag = True
                        flag_orig = original_bbox
                        break
                if flag:
                    classified_bboxes.add(flag_orig)
                    TP_counter += 1

            FP_counter = len(filtered_predicted_bboxes[image_name]) - TP_counter
            FN_counter = len(filtered_original_bboxes[image_name])  - TP_counter

            glob_TP += TP_counter
            glob_FP += FP_counter
            glob_FN += FN_counter

        return filtered_predicted_bboxes, filtered_original_bboxes, glob_TP, glob_FP, glob_FN


__all__ = [
    'Validator'
]
