from platescanner.pipelines import PTPipeline
from platescanner.utils import to_mvp
from src.platescanner import DATASETS_PATH


def main():
    dataset = to_mvp(DATASETS_PATH / 'product_dataset')
    pipeline = PTPipeline()
    result = pipeline.predict(dataset)
    for image_result in result:
        print(image_result)


if __name__ == '__main__':
    main()
