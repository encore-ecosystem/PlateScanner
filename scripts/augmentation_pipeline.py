from src.data_utils import Combiner, AutoAdapter, AdapterOutputType
from src import DATA_PATH
from pathlib import Path
from albumentations import TransformType
from tqdm import tqdm
from PIL import Image

import albumentations
import numpy as np
import random
import shutil


# <ALL TRAINING DATASETS>
#  |
#  V
#  Merged Dataset -> Only Gray Augmentation with CLOHA
#  |                                                |
#  V                                                |
# Blur + Noise + Gray Augmentation with CLOHA       |
#  |                                                |
#  V                                                V
#  <            Augmented Dataset                   >
#                       |
#                     / | \
#                    /  |  \
#         +contrast     |   -contrast
#                   \   |   /
#                  Final Dataset

MERGED_DATASET          = DATA_PATH / 'Merged'        / 'YOLO'
GRAY_ONLY_DATASET       = DATA_PATH / 'GrayOnly'      / 'YOLO'
BLUR_NOISE_GRAY_DATASET = DATA_PATH / 'GrayBlurNoise' / 'YOLO'
FINAL_DATASET           = DATA_PATH / 'Final'         / 'YOLO'

def random_hash(k: int = 256):
    return hex(random.getrandbits(k))[2:]

def augmentation(src: Path, dst: Path, augmentations: list[TransformType]):
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src=src / 'data.yaml', dst=dst / 'data.yaml')

    transform = albumentations.Compose(augmentations)

    for type_ in ['train', 'valid', 'test']:
        (dst / type_ / 'images').mkdir(parents=True, exist_ok=True)
        (dst / type_ / 'labels').mkdir(parents=True, exist_ok=True)
        files = [*(src / type_ / 'images').iterdir()]
        for image_path in tqdm(
                files,
                desc=f'Processing {type_} images',
                total=len(files),
        ):
            image_name = image_path.name.rstrip('.jpg')
            image_new_id = random_hash()

            image = np.array(Image.open(image_path))
            transformed = Image.fromarray(transform(image=image)['image'].astype(np.uint8))
            transformed.save(dst / type_ / 'images' / (image_new_id + '.jpg'))

            shutil.copy(
                src=src / type_ / 'labels' / (image_name + '.txt'),
                dst=dst / type_ / 'labels' / (image_new_id + '.txt')
            )

def step1():
    Combiner.merge(
        adapters    = [AutoAdapter(dataset_path=path) for path in (DATA_PATH / 'Train' / 'YOLO').iterdir()],
        output_type = AdapterOutputType.YOLO
    ).save(MERGED_DATASET)

def step2():
    augmentation(
        src           = MERGED_DATASET,
        dst           = GRAY_ONLY_DATASET,
        augmentations = [
            albumentations.CLAHE(p=1),
            albumentations.ToGray(p=1),
        ]
    )

def step3():
    augmentation(
        src = MERGED_DATASET,
        dst = BLUR_NOISE_GRAY_DATASET,
        augmentations = [
            albumentations.Blur(blur_limit=(3, 7), p=1),
            albumentations.ISONoise(p=1),
            albumentations.CLAHE(p=1),
            albumentations.ToGray(p=1),
        ]
    )

def step4():
    Combiner.merge(
        adapters=[AutoAdapter(dataset_path=path) for path in [
            GRAY_ONLY_DATASET,
            BLUR_NOISE_GRAY_DATASET,
        ]],
        output_type=AdapterOutputType.YOLO
    ).save(MERGED_DATASET)

def step5():
    augmentation(
        src = MERGED_DATASET,
        dst = FINAL_DATASET,
        augmentations = [
            albumentations.RandomBrightnessContrast(p=1/3)
        ]
    )

def main():
    step1()  # Merge All Datasets
    step2()  # step1 -> gray + cloha
    step3()  # step1 -> blur + noise + gray + cloha
    step4()  # dataset = step2 + step3
    step5()  # dataset + random contrast p=1/3


if __name__ == '__main__':
    main()
