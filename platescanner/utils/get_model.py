from src import MODEL_PATH_FOLDER
from pathlib import Path


def get_model_cli() -> Path:
    try:
        while True:
            models = []
            for i, model_path in enumerate(MODEL_PATH_FOLDER.glob("*.pt")):
                model_name = model_path.stem
                type_ = 'OBB' if 'obb' in model_name else 'BB '
                print(f"{i}) {type_} {model_name} ")
                models.append(model_path)
            idx = input(">>> ")
            if not (idx.isdigit() and int(idx) in range(len(models))):
                print("Invalid input")
                continue
            return models[int(idx)]

    except KeyboardInterrupt:
        pass


__all__ = [
    'get_model_cli'
]
