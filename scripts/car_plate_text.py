from experimentator import ExperimentatorClient, Experiment, Pipeline, Trainer, Measurer
from prologger import ConsoleLogger

from platescanner import EXPERIMENTATOR_PATH, MODELS_PATH, DATASETS_PATH
from platescanner.models.detectors import Yolo
from platescanner.models.recognizers.blank import Plug
from platescanner.utils import to_mvp


def main():
    client = ExperimentatorClient(EXPERIMENTATOR_PATH)

    pipeline = Pipeline(
        models = [
            Yolo(), # Car Detector
            Yolo(), # Plate Detector
            Plug(weight_path  = MODELS_PATH / "blank" / "plug.pcl"),    # Plate Recognizer
        ]
    )
    trainers = [
        Trainer(
            train_dataset   = to_mvp(DATASETS_PATH / "product_dataset"),
            test_dataset    = to_mvp(DATASETS_PATH / "product_dataset"),
            eval_dataset    = to_mvp(DATASETS_PATH / "product_dataset"),
            epochs          = 10,
            checkpoint_step = 5,
            resume          = True,
            measurer        = Measurer(),
            logger          = ConsoleLogger(),
        )
        for _ in range(3)
    ]

    experiment = Experiment(
        name = "car_plate_text",
        pipeline = pipeline,
        trainers = trainers,
        logger   = ConsoleLogger()
    )

    client.run_experiment(experiment)


if __name__ == '__main__':
    main()
