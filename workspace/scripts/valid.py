from clearml import Model, Task
from ultralytics import YOLO
from pathlib import Path
import os

DATA = Path() / 'dataset' / 'detection' / 'validation and testing' / 'YOLO'

def main():
    # Validation
    config1 = {
        "batch": 8,
        "imgsz": 1280,
        "data": (DATA / 'Rus' / 'data.yaml').resolve()
    }
    config2 = {
        "batch": 8,
        "imgsz": 1280,
        "data": (DATA / 'Cis' / 'data.yaml').resolve()
    }

    model = YOLO(Model(
        model_id='10e0c7a8128f4a038ee4aabd113880d6'
    ).get_local_copy())

    # Create a ClearML Task
    task = Task.init(
        project_name="PlateDetector",
        task_name="Validation FT YOLOv8"
    )
    task.set_parameter(name="model_variant", value='yolov8m')
    task.connect(config1)

    results1 = model.val(**config1)

    iteration = 0
    task.get_logger().report_scalar(title='Validation Metrics mAP Rus', series="mAP 50", value=results1.box.map50, iteration=iteration)
    task.get_logger().report_scalar(title='Validation Metrics mAP Rus', series="mAP 75", value=results1.box.map75, iteration=iteration)
    task.get_logger().report_scalar(title='Validation Metrics mAP Rus', series="mAP 50-95", value=results1.box.maps.item(), iteration=iteration)
    task.get_logger().report_scalar(title='Number of Classes Rus', series="Number of Classes", value=results1.box.nc, iteration=iteration)
    task.get_logger().report_scalar(title='Precision and Recall Rus', series="Precision", value=results1.box.p.item(), iteration=iteration)
    task.get_logger().report_scalar(title='Precision and Recall Rus', series="Recall", value=results1.box.r.item(), iteration=iteration)

    task.connect(config2)
    results2 = model.val(**config2)
    task.get_logger().report_scalar(title='Validation Metrics mAP Cis', series="mAP 50", value=results2.box.map50, iteration=iteration)
    task.get_logger().report_scalar(title='Validation Metrics mAP Cis', series="mAP 75", value=results2.box.map75, iteration=iteration)
    task.get_logger().report_scalar(title='Validation Metrics mAP Cis', series="mAP 50-95", value=results2.box.maps.item(), iteration=iteration)
    task.get_logger().report_scalar(title='Number of Classes Cis', series="Number of Classes", value=results2.box.nc, iteration=iteration)
    task.get_logger().report_scalar(title='Precision and Recall Cis', series="Precision", value=results2.box.p.item(), iteration=iteration)
    task.get_logger().report_scalar(title='Precision and Recall Cis', series="Recall", value=results2.box.r.item(), iteration=iteration)

    # Prediction
    test_data_path1 = (DATA / 'Rus' / 'test' / 'images').resolve()
    test_data_path2 = (DATA / 'Cis' / 'test' / 'images').resolve()
    test_images = list(test_data_path1.glob('*.jpg')) + list(test_data_path2.glob('*.jpg'))
    predictions = []
    for img_path in test_images:
        results = model.predict(source=str(img_path))
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
