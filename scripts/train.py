from src.model import Yolo
from pathlib import Path
import albumentations


def main():
    Yolo(
        weights_path=Path(__file__).parent.parent / 'models' / 'yolo11n-obb.pt'
    ).fit(
        dataset_path=Path(__file__).parent.parent / 'dataset' / 'dataset3_obb',
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
        use_clearml=True,
    )


if __name__ == '__main__':
    main()
