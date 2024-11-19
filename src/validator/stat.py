from bdb import Breakpoint

import cv2
import numpy as np

from pathlib import Path

from src.bbox.bbox_abs import Bbox
from src.validator.custom_categories import CustomCategories, Time, Distance
from src.utils import bbox_iou


class Validator:
    def __init__(self):
        pass

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

        for image_path in (dataset_path / 'valid' / 'images_old_pred').glob("*"):
            image = cv2.imread(image_path)
            image_area = image.shape[0] * image.shape[1]
            image_bboxes = bboxes[image_path.stem]

            for bbox in image_bboxes:
                average_bbox_size_ratios.append(bbox.area() / image_area)

        self.bbox_size_ratios_percentile_point_33 = np.percentile(average_bbox_size_ratios, 33)
        self.bbox_size_ratios_percentile_point_66 = np.percentile(average_bbox_size_ratios, 66)

    def predict(self, dataset_path: Path, bboxes: dict[str, list[Bbox]]):
        new_bboxes = {}

        for image_path in (dataset_path / 'valid' / 'images_old_pred').glob("*"):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_image)

            image_area = image.shape[0] * image.shape[1]

            if mean_brightness > self.brightnesses_percentile_point:
                time = Time.DAY
            else:
                time = Time.NIGHT

            for bbox in bboxes[image_path.stem]:
                custom_categories = CustomCategories()

                custom_categories.time = time

                bbox_area_ratio = bbox.area() / image_area

                if bbox_area_ratio < self.bbox_size_ratios_percentile_point_33:
                    custom_categories.distance = Distance.FAR
                elif self.bbox_size_ratios_percentile_point_33 < bbox_area_ratio < self.bbox_size_ratios_percentile_point_66:
                    custom_categories.distance = Distance.MIDDLE
                else:
                    custom_categories.distance = Distance.CLOSE

                new_bboxes[image_path.stem] = new_bboxes.get(image_path.stem, []) + [(bbox, custom_categories)]

        print(new_bboxes)

        return new_bboxes

    def compute_confusion_matrix(self,
                                 original_bboxes: dict[str, list[tuple[Bbox, CustomCategories]]],
                                 predicted_bboxes: dict[str, list[Bbox]],
                                 criteria: CustomCategories,
                                 selected_category: int,
                                 threshold=0.05):
        filtered_original_bboxes = {}
        filtered_predicted_bboxes = {}

        for image_name in original_bboxes.keys():
            for bbox, custom_criteria in original_bboxes[image_name]:
                if bbox.category != selected_category:
                    continue
                if not (custom_criteria.time != criteria.time or custom_criteria.time == 0):
                    continue
                if not (custom_criteria.distance != criteria.distance or custom_criteria.distance == 0):
                    continue
                filtered_original_bboxes[image_name] = filtered_original_bboxes.get(image_name, []) + [(bbox, custom_criteria)]

        for image_name in predicted_bboxes.keys():
            for bbox in predicted_bboxes[image_name]:
                if bbox.category == selected_category:
                    filtered_predicted_bboxes[image_name] = filtered_predicted_bboxes.get(image_name, []) + [bbox]

        for image_name in filtered_predicted_bboxes.keys():
            classified_bboxes = set()

            TP_counter = 0
            for predicted_bbox in filtered_predicted_bboxes[image_name]:
                flag = False
                flag_orig = None
                for original_bbox, custom_criteria in filtered_original_bboxes[image_name]:
                    if bbox_iou(original_bbox, predicted_bbox) > threshold and \
                        original_bbox not in classified_bboxes and \
                        predicted_bbox not in classified_bboxes:
                        flag = True
                        flag_orig = original_bbox
                        break
                if flag:
                    classified_bboxes.add(predicted_bbox)
                    classified_bboxes.add(flag_orig)
                    TP_counter += 1

            FP_counter = len(predicted_bboxes[image_name]) - TP_counter

            FN_counter = len(original_bboxes[image_name]) - TP_counter


            print(f"TP: {TP_counter} FP: {FP_counter} FN: {FN_counter}, len_orig {len(filtered_original_bboxes[image_name])}, len_pred {len(filtered_predicted_bboxes[image_name])} im_name: {image_name} ")


#
# Folder with original labels
# Dict[filename : bbox_structure]
# Dataset_path <- filename; img -> 0/1
# 

#
# Folder with predicted labels
#
