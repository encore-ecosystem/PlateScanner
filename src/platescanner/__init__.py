from pathlib import Path
import warnings
import torch
warnings.filterwarnings("ignore")

# ================
# Hyperparameters
# ================
DEFAULT_CONFIDENCE_LEVEL   = 0.06
BBOX_EDGE_COLOR            = 'red'
BBOX_TEXT_COLOR            = 'red'
BBOX_TEXT_FONTSIZE         = 5
BBOX_TEXT_HORIZONTAL_SHIFT = -125
BBOX_TEXT_VERTICAL_SHIFT   = -20

# ================
# Constants
# ================
PLATESCANNER_ROOT_PATH = Path()
EXPERIMENTATOR_PATH    = PLATESCANNER_ROOT_PATH / ".experimentator"
DATASETS_PATH          = PLATESCANNER_ROOT_PATH / "datasets"
MODELS_PATH            = PLATESCANNER_ROOT_PATH / "model_weights"
CALIBRATION_DATASET    = PLATESCANNER_ROOT_PATH / "calibration"
TEMP_FOLDER            = PLATESCANNER_ROOT_PATH / "TEMP"
TORCHHUB_PATH          = PLATESCANNER_ROOT_PATH / ".torchhub"

#
torch.hub.set_dir(TORCHHUB_PATH)
