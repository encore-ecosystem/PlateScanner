from ultralytics import YOLO
from app.src import DATA_PATH, RESULTS_DIR


def main():
    # Working with folders
    if not (DATA_PATH.exists() and DATA_PATH.is_file()):
        raise FileNotFoundError(f"Dataset yaml '{DATA_PATH}' does not exist. Please check the path.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load a model
    model = YOLO(RESULTS_DIR / "yolov8n.pt")

    # Use the model
    results = model.train(
        data    = DATA_PATH,
        epochs  = 1,
        project = RESULTS_DIR,
    )


if __name__ == '__main__':
    # main()
    import time
    while True:
        print("All works fine!")
        time.sleep(5)
