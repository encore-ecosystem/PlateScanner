from clearml import Model, Task
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os

DATA = Path().resolve().parent / 'dataset' / 'detection' / 'TargetAUG'

def main():
    config = {
        "batch": 8,
        "imgsz": 1280
    }
    model = YOLO(Model(
        model_id='305eb7d3ac9941eb9cd16e0afb5a470f'
    ).get_local_copy())

    # Create a ClearML Task
    task = Task.init(
        project_name="PlateScanner",
        task_name="Assessment of target values"
    )
    task.set_parameter(name="model_variant", value='yolov5nu')
    task.connect(config)

    # Prediction
    test_data_path = DATA
    test_images = list(test_data_path.glob('*.bmp'))
    predictions = []
    for img_path in tqdm(test_images):
        results = model.predict(source=str(img_path), conf=0.06)
        predictions.append(results)

        # Saving to ClearML
        for result in results:
            task.get_logger().report_image(
                title=f"Prediction for {os.path.basename(img_path)}",
                series='Predictions',
                iteration=len(predictions),
                image=result.plot()
            )



if __name__ == '__main__':
    main()
