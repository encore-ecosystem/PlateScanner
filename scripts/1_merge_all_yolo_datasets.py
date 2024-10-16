from src import DATA_PATH
from src.data_utils.combiner import Combiner
from src.data_utils.adapters import AdapterOutputType, AutoAdapter
from tqdm import tqdm

def main():
    datasets = (DATA_PATH / 'Train' / 'YOLO').iterdir()
    dataset = Combiner.merge(
        adapters    = [AutoAdapter(dataset_path=path) for path in datasets],
        output_type = AdapterOutputType.YOLO
    )
    dataset.save(DATA_PATH / 'Merged' / 'YOLO')


if __name__ == '__main__':
    main()
