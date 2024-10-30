from pathlib import Path

# ================
# Constants
# ================
DATA_PATH   = Path().resolve() / 'dataset' / 'detection'


from ultralytics.data.augment import Albumentations

import albumentations as A
def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        # Insert required transformation here
        T = [
            A.ToGray(p=1),
            A.CLAHE(p=1),
        ]
        self.transform = A.Compose(T)

Albumentations.__init__ = __init__

__all__ = [
    'DATA_PATH',
]
