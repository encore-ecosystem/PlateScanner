import shutil

from nodeflow.builtin import PathVariable, Boolean, Result, Integer
from ultralytics import YOLO
from clearml import Task
from pathlib import Path
import cvtk

from platescanner import PLATESCANNER_ROOT_PATH


def validate_yolo(model_path: PathVariable, dataset_path: PathVariable, imgsz: Integer, use_clearml: Boolean) -> Result:
    dataset = cvtk.determine_dataset(dataset_path.value)

    use_temp = False
    if not isinstance(dataset, cvtk.YOLO_Dataset):
        use_temp = True
        dataset = cvtk.autoconvert(dataset_path.value, cvtk.YOLO_Dataset)
        dataset_path = PLATESCANNER_ROOT_PATH / 'temp' / dataset_path.value.name
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset.write(dataset_path)

    config = {
        "batch"       : 1,
        "imgsz"       : imgsz.value,
        "data"        : (dataset_path / 'data.yaml').__str__(),
    }
    (__validate_yolo_with_clearml if use_clearml.value else __validate_yolo_without_clearml)(model_path, config)

    if use_temp:
        shutil.rmtree(dataset_path.value)

    return Result(value=True)


def __validate_yolo_with_clearml(model_path: PathVariable, config: dict):
        model = YOLO(model_path.value.__str__())
        model_variant = model_path.value.stem

        # Create a ClearML Task
        task = Task.init(
            project_name = "PlateScanner",
            task_name    = f"[Validate] <{model_variant}> on < dataset (4) >"
        )
        task.set_parameter(name="model_variant", value=model_variant)
        task.connect(config)

        results = model.val(**config)

        iteration = 0
        task.get_logger().report_scalar(
            title='Validation Metrics mAP Rus',
            series="mAP 50",
            value=results.box.map50,
            iteration=iteration,
        )
        task.get_logger().report_scalar(
            title='Validation Metrics mAP Rus',
            series="mAP 75",
            value=results.box.map75,
            iteration=iteration,
        )
        task.get_logger().report_scalar(title='Validation Metrics mAP Rus', series="mAP 50-95",
                                        value=results.box.maps.item(), iteration=iteration)
        task.get_logger().report_scalar(title='Number of Classes Rus', series="Number of Classes",
                                        value=results.box.nc, iteration=iteration)
        task.get_logger().report_scalar(title='Precision and Recall Rus', series="Precision",
                                        value=results.box.p.item(), iteration=iteration)
        task.get_logger().report_scalar(title='Precision and Recall Rus', series="Recall", value=results.box.r.item(),
                                        iteration=iteration)
        # Prediction
        predictions = []
        for img_path in (Path(config['data']).parent / 'test' / 'images').resolve().glob('*.jpg'):
            results = model.predict(source=str(img_path))
            predictions.append(results)

            # Saving to ClearML
            for result in results:
                task.get_logger().report_image(
                    title=f"Prediction for {img_path.name}",
                    series='Predictions',
                    iteration=len(predictions),
                    image=result.plot()
                )


def __validate_yolo_without_clearml(model_path: PathVariable, config: dict):
    YOLO(model_path.value.__str__()).val(**config)


__all__ = [
    'validate_yolo'
]
