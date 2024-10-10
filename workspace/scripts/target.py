from clearml import Model, Task
from ultralytics import YOLO
from pathlib import Path
import os

DATA = Path() / 'dataset' / 'detection' / 'target pictures aug'

def main():
    config = {
        "batch": 8,
        "imgsz": 1280
    }
    model = YOLO(Model(
        model_id='8e27da48d69149e3bd760f87975acc0f'
    ).get_local_copy())

    # Create a ClearML Task
    task = Task.init(
        project_name="PlateScanner",
        task_name="Assessment of target values"
    )
    task.set_parameter(name="model_variant", value='yolov11x')
    task.connect(config)

    # Prediction
    test_data_path = (DATA).resolve()
    test_images = list(test_data_path.glob('*.bmp'))
    predictions = []
    for img_path in test_images:
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
