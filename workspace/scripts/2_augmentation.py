from src import DATA_PATH
from tqdm import tqdm
from PIL import Image

import albumentations
import numpy as np
import shutil
import random

INPUT_YOLO_DATASET  = DATA_PATH / 'Merged'
OUTPUT_YOLO_DATASET = DATA_PATH / 'AugmentedFull' / 'YOLO'

def random_hash(k: int = 256):
    return hex(random.getrandbits(k))[2:]

def main():
    if OUTPUT_YOLO_DATASET.exists():
        shutil.rmtree(OUTPUT_YOLO_DATASET)

    OUTPUT_YOLO_DATASET.mkdir(parents=True, exist_ok=True)
    shutil.copy(src=INPUT_YOLO_DATASET / 'data.yaml', dst=OUTPUT_YOLO_DATASET / 'data.yaml')

    transform = albumentations.Compose([
        # albumentations.Blur(blur_limit=(3, 7), p=1),
        # albumentations.GaussNoise(var_limit=(10., 50.), p=1),

        albumentations.CLAHE(p=1),
        albumentations.ToGray(p=1),
    ])

    for type_ in ['train', 'valid', 'test']:
        (OUTPUT_YOLO_DATASET / type_ / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_YOLO_DATASET / type_ / 'labels').mkdir(parents=True, exist_ok=True)
        files = [*(INPUT_YOLO_DATASET / type_ / 'images').iterdir()]
        for image_path in tqdm(
            files,
            desc=f'Processing {type_} images',
            total=len(files),
        ):
            image_name = image_path.name.rstrip('.jpg')
            image_new_id = random_hash()

            image = np.array(Image.open(image_path))
            transformed = Image.fromarray(transform(image=image)['image'].astype(np.uint8))
            transformed.save(OUTPUT_YOLO_DATASET / type_ / 'images' / (image_new_id + '.jpg'))

            shutil.copy(
                src=INPUT_YOLO_DATASET / type_ / 'labels' / (image_name + '.txt'),
                dst=OUTPUT_YOLO_DATASET / type_ / 'labels' / (image_new_id + '.txt')
            )


if __name__ == '__main__':
    main()
