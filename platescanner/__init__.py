from pathlib import Path

# ================
# Hyperparameters
# ================
DEFAULT_CONFIDENCE_LEVEL   = 0.06
BBOX_EDGE_COLOR            = 'red'
BBOX_TEXT_COLOR            = 'red'
BBOX_TEXT_FONTSIZE         = 8
BBOX_TEXT_HORIZONTAL_SHIFT = -125
BBOX_TEXT_VERTICAL_SHIFT   = -20

# ================
# Constants
# ================
PROJECT_ROOT_PATH   = Path(__file__).parent.parent.resolve()
MODEL_PATH_FOLDER   = PROJECT_ROOT_PATH / "models"
CALIBRATION_DATASET = PROJECT_ROOT_PATH / "calibration"
FSR_MODEL_PATH      = MODEL_PATH_FOLDER / "FSR" / "FSRCNN_x4.pb"
TEMP_FOLDER = PROJECT_ROOT_PATH / "TEMP"