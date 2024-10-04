from pathlib import Path

# ================
# Constants
# ================
DATA_PATH   = Path() / 'dataset' / 'datasets.yaml'
RESULTS_DIR = Path() / 'results' / 'training'


__all__ = [
    'DATA_PATH',
    'RESULTS_DIR',
]
