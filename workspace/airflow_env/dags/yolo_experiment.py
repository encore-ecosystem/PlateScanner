from ultralytics import YOLO
from clearml import Task
from src import DATA_PATH

config = {
    "batch": 8,
    "imgsz": 1280,
    "epochs": 20,
    "name": "YOLOv5",
    "data": [(path / 'data.yaml').resolve() for path in DATA_PATH.iterdir() if '_Y_' in path.name][0],
}


def main():
    # Create a ClearML Task
    task = Task.init(
        project_name="Testing Yolo 1",
        task_name="my yolo task 3"
    )

    # Load a model
    model_variant = "yolov5nu"
    # Log "model_variant" parameter to task
    task.set_parameter(name="model_variant", value=model_variant)

    # Load the YOLOv8 model
    model = YOLO(f'{model_variant}.pt')

    # Put all YOLOv8 arguments in a dictionary and pass it to ClearML
    # When the arguments are later changed in UI, they will be overridden here!
    args = config
    task.connect(args)

    # Train the model
    # If running remotely, the arguments may be overridden by ClearML if they were changed in the UI
    results = model.train(**args)
