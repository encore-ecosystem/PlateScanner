import os

from platescanner import PROJECT_ROOT_PATH
from pathlib import Path


def handle_path(path: str, should_exist: bool = True) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (Path(os.getcwd()) / path).resolve()

    if not (should_exist and path.exists()):
        print(f"Output path does not exist: {path}")
        exit(-1)

    return path

def handle_confidence_level(confidence_level: str) -> float:
    if not confidence_level.isdigit() or not 0 <= int(confidence_level) <= 100:
        print("Invalid confidence level.")
        exit(-1)

    return float(confidence_level) / 100


__all__ = [
    'handle_path',
    'handle_confidence_level',
]
