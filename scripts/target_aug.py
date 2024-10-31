from src import DATA_PATH
import os
import cv2
import albumentations as A

input_folder  = DATA_PATH.resolve().parent.parent.parent / 'dataset' / 'detection' / 'TargetRAW'
output_folder = DATA_PATH.resolve().parent.parent.parent / 'dataset' / 'detection' / 'TargetAUG'

output_folder.mkdir(exist_ok=True, parents=True)

transform = A.Compose([
    A.CLAHE(p=1),
])

for filename in os.listdir(input_folder):
    if filename.endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)

        transformed = transform(image=image)
        augmented_image = transformed['image']

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, augmented_image)