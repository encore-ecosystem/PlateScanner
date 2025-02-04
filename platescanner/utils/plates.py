from PIL.Image import Image
from numpy import ndarray

import albumentations as A
import numpy as np
import cv2

from platescanner import FSR_MODEL_PATH

# LOAD FSR
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(FSR_MODEL_PATH.__str__())
sr.setModel("fsrcnn", 3)


def preprocess_for_alignment(image: ndarray) -> ndarray:
    """Предварительная обработка для выделения контуров."""
    if not (image.ndim == 2 or image.shape[-1] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def align_license_plate(plate_image: Image) -> ndarray:
    """Выравнивание номера без изменений ориентации."""
    plate_image = np.array(plate_image)
    binary = preprocess_for_alignment(plate_image)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Размеры выровненного изображения
        width, height = int(rect[1][0]), int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32"
        )

        m = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned = cv2.warpPerspective(plate_image, m, (width, height))
        if aligned.shape[0] > aligned.shape[1]:
            aligned = cv2.rotate(aligned, cv2.ROTATE_90_CLOCKWISE)
        return aligned

    return plate_image

def preprocess_license_plate(plate_image):
    resized = cv2.resize(plate_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    augmented = A.Compose([
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 2.0), p=1.0),
        A.MedianBlur(blur_limit=3, p=0.5),
    ])(image=resized)['image']
    super_resolved = sr.upsample(augmented)
    return super_resolved


def preprocess_license_plate_without_aug(plate_image):
    print("Hello FSR")
    augmented = A.Compose([
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 2.0), p=1.0),
    ])(image=plate_image)['image']
    super_resolved = sr.upsample(augmented)
    return super_resolved


__all__ = [
    'preprocess_for_alignment',
    'align_license_plate',
    'preprocess_license_plate',
    'preprocess_license_plate_without_aug',
]
