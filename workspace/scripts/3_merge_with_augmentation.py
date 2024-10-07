from src import DATA_PATH
from src.data_utils.combiner import Combiner
from src.data_utils.adapters import AdapterOutputType, AutoAdapter


def main():
    datasets = [
        DATA_PATH / 'AugmentedAll'      / 'YOLO',
        DATA_PATH / 'AugmentedOnlyGray' / 'YOLO',
    ]
    dataset = Combiner.merge(
        adapters    = [AutoAdapter(dataset_path=path) for path in datasets],
        output_type = AdapterOutputType.YOLO
    )
    dataset.save(DATA_PATH / 'AUGMerged' / 'YOLO')


if __name__ == '__main__':
    main()
