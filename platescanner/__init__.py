import os
from pathlib import Path
import warnings
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
PLATESCANNER_ROOT_PATH   = os.getenv("PLATESCANNER_ROOT_PATH")
if PLATESCANNER_ROOT_PATH is None:
    raise ValueError("PLATESCANNER_ROOT_PATH must be set")
PLATESCANNER_ROOT_PATH = Path(PLATESCANNER_ROOT_PATH)
MODEL_PATH_FOLDER   = PLATESCANNER_ROOT_PATH / "models"
CALIBRATION_DATASET = PLATESCANNER_ROOT_PATH / "calibration"
FSR_MODEL_PATH      = MODEL_PATH_FOLDER / "FSR" / "FSRCNN_x4.pb"
TEMP_FOLDER = PLATESCANNER_ROOT_PATH / "TEMP"

# ================
# Setting seed for reproducibility of experiments
# ================
import torch
import numpy as np
import random
import cv2

seed = 52
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cv2.setRNGSeed(seed)
