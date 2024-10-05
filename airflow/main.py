from ultralytics import YOLO, settings
from src import DATA_PATH
# import mlflow

settings.update({"mlflow": False})
settings.reset()

config = {
    "batch": 8,
    "imgsz": 1280,
    "epochs": 1,
    "name": "YOLOv8",
    "data": [(path / 'data.yaml').resolve() for path in DATA_PATH.iterdir() if '_Y_' in path.name][0],
}


def main():
    # mlflow.set_tracking_uri(uri="http://localhost:5000")
    #
    # # set the experiment id
    # ## experiment_name="YOLOv8 Basic Usage of MLFlow"
    # mlflow.set_experiment(experiment_id="0")

    # start run
    # with mlflow.start_run():
        # Logging parameters
        # mlflow.log_param("epochs", config["epochs"])
        # mlflow.log_param("batch_size", config["batch"])
        # mlflow.log_param("img_size", config["imgsz"])
        # mlflow.log_param("data_path", config["data"])
        # mlflow.log_param("experiment_name", config["name"])

        # Load and train your YOLOv8 model
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        name=config["name"],
        imgsz=config["imgsz"],
    )
        #
        # # Metrics
        # metrics = results.metrics
        #
        # # Logging  Metric
        # for metric, value in metrics.items():
        #     mlflow.log_metric(metric, value)
        #
        # # Logging model
        # mlflow.pytorch.log_model(model.model, "YOLOv8_model")


if __name__ == '__main__':
    main()
