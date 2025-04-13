from platescanner.pipelines import CPTPipeline
from platescanner.utils import to_mvp
from src.platescanner import DATASETS_PATH


def main():
    dataset = to_mvp(DATASETS_PATH / 'product_dataset')
    pipeline = CPTPipeline()
    result = pipeline.predict(dataset)
    print(result)


if __name__ == '__main__':
    main()
