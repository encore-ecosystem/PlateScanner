from clearml import Model, Task
from ultralytics import YOLO
from pathlib import Path


def main():
    config = {
        "batch": 3,
        "imgsz": 1280,
        "epochs": 50,
        "name": "YOLOv8",
        "data": (Path() / 'dataset' / 'detection' / 'data_6_CIS_Y_13476_2618_1419' / 'data.yaml').resolve()
    }

    model = YOLO(Model(
        model_id='65eebe7f5e5c45d5ac167d7097a08b11'
    ).get_local_copy())

    # Create a ClearML Task
    task = Task.init(
        project_name="PlateDetector",
        task_name="Fine Tuning YOLOv8"
    )
    task.set_parameter(name="model_variant", value='yolov8m')
    task.connect(config)

    model.train(**config)


if __name__ == '__main__':
    main()
