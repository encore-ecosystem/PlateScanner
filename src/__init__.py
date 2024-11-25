from pathlib import Path

# ================
# Hyperparameters
# ================
DEFAULT_CONFIDENCE_LEVEL   = 0.06
BBOX_EDGE_COLOR            = 'red'
BBOX_TEXT_COLOR            = 'red'
BBOX_TEXT_FONTSIZE         = 4
BBOX_TEXT_HORIZONTAL_SHIFT = -125
BBOX_TEXT_VERTICAL_SHIFT   = -20

# ================
# Constants
# ================
PROJECT_ROOT_PATH = Path(__file__).parent.parent.resolve()
MODEL_PATH_FOLDER = PROJECT_ROOT_PATH / "models"
