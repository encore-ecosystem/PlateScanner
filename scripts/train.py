from src import PROJECT_ROOT_PATH
from src.model import Yolo
from pathlib import Path
import albumentations


def main():
    Yolo(
        weights_path= PROJECT_ROOT_PATH / 'models' / 'yolov11v2-obb.pt'
    ).fit(
        dataset_path= PROJECT_ROOT_PATH  / 'dataset' / 'target_pictures_OBB',
        augmentation=albumentations.Compose([
            albumentations.RandomBrightnessContrast(p=0.5,brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4)),
            albumentations.RandomRain(p=0.5, brightness_coefficient=1),
            albumentations.RandomSnow(p=0.5),
            albumentations.ISONoise(p=0.5),
            albumentations.CLAHE(always_apply=True),
            albumentations.ToGray(always_apply=True),
        ],
        bbox_params=albumentations.BboxParams(format="yolo"), strict=False),
        # todo: fix False usage <on arch linux>
        use_clearml=False,
    )


if __name__ == '__main__':
    main()
