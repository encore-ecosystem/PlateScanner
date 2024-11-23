from src.model import Yolo
from pathlib import Path
import albumentations


def main():
    Yolo(
        weights_path=Path(__file__).parent.parent / 'model' / 'yolo11n.pt'
    ).fit(
        dataset_path=Path(__file__).parent.parent / 'dataset' / 'detection' / 'dataset3',
        augmentation=albumentations.Compose([
            albumentations.RandomRain(p=0.5, brightness_coefficient=1),
            # to do: Не работает тренировка на Morphological, исправить
            #albumentations.Morphological(p=0.4, scale=(2, 3), operation="erosion"),
            albumentations.RandomBrightnessContrast(p=0.5),

            albumentations.CLAHE(always_apply=True),
            albumentations.ToGray(always_apply=True),
        ],
        bbox_params=albumentations.BboxParams(format="yolo")),
        # todo: fix False usage <on arch linux>
        use_clearml=True,
    )


if __name__ == '__main__':
    main()
