from ultralytics import YOLO
from platescanner import DATASETS_PATH


def main():
    model = YOLO(model="yolov5nu")
    path = DATASETS_PATH / "car_detection" / "data.yaml"
    results = model.train(data=path.__str__(), epochs=100, imgsz=640, device=0, batch=-1)


if __name__ == "__main__":
    main()
