from nodeflow.builtin import PathVariable, Boolean, Result, Integer
from ultralytics import YOLO
from clearml import Task

from platescanner.nodeflow_env.variables import MyPath


def train_yolo(model_path: MyPath, dataset_path: PathVariable, imgsz: Integer, epochs: Integer, use_clearml: Boolean) -> Result:
    config = {
        "batch"       : -1,
        "imgsz"       : imgsz.value,
        "epochs"      : epochs.value,
        "data"        : (dataset_path / 'data.yaml').__str__(),
        "save_period" : 10,
        "augment"     : True,
    }
    (__train_yolo_with_clearml if use_clearml.value else __train_yolo_without_clearml)(model_path, config)
    return Result(value=True)

def __train_yolo_with_clearml(model_path: MyPath, config: dict):
    model_variant = model_path.value.stem
    task = Task.init(
        project_name = "PlateScanner",
        task_name    = f"Fine-tuning YOLO <{model_variant}> on dataset (3) with obb"
    )
    task.set_parameter(name="model_variant", value=model_variant)
    model = YOLO(model_path.value.__str__())

    task.connect(config)
    model.train(**config)

def __train_yolo_without_clearml(model_path: MyPath, config: dict):
    YOLO(model_path.value.__str__()).train(**config)


__all__ = [
    'train_yolo'
]
