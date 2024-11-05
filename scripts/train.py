from src.model import YoloV5
from pathlib import Path
import albumentations


def main():
    YoloV5(
        weights_path=Path(__file__).parent.parent / 'model' / 'yolov5nu.pt'
    ).fit(
        dataset_path=Path(__file__).parent.parent / 'dataset' / 'detection' / 'data1',
        augmentation=albumentations.Compose([
            albumentations.RandomRain(p=0.5, brightness_coefficient=1),
            # albumentations.Morphological(p=0.4, operation="erosion"),
            albumentations.RandomBrightnessContrast(p=0.5),

            albumentations.CLAHE(always_apply=True),
            albumentations.ToGray(always_apply=True),
        ],
        bbox_params=albumentations.BboxParams(format="yolo")),
        # todo: fix False usage
        use_clearml=True,
    )


if __name__ == '__main__':
    main()
