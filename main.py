from ultralytics import YOLO
from clearml import Task
from src import DATA_PATH


MODEL = 'YOLO11x'

config = {
    "batch": 2,
    "imgsz": 1280,
    "epochs": 50,
    "name": MODEL,
    "data": DATA_PATH / 'AUGMerged' / 'YOLO' / 'data.yaml',
    "save_period": 2
}


def main():
    # Create a ClearML Task
    task = Task.init(
        project_name = "PlateScanner",
        task_name    = f"Testing {MODEL} on augmented dataset"
    )

    model_variant = MODEL.lower()

    task.set_parameter(name="model_variant", value=model_variant)

    model = YOLO(f'{model_variant}.pt')

    args = config
    task.connect(args)

    model.train(**args)


if __name__ == '__main__':
    main()
