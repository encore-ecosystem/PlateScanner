from ultralytics.data.augment import Albumentations
from nodeflow.node import Variable
from nodeflow.builtin import Result
import albumentations


class Compose(Variable):
    def __init__(self, value: albumentations.Compose):
        super().__init__(value)


def load_augmentations(transform_compose: Compose) -> Result:
    Albumentations.__init__ = lambda self, p: __apply_augmentations(self, transform_compose.value)
    return Result(True)

def __apply_augmentations(self, transforms: albumentations.Compose):
    # List of possible spatial transforms
    spatial_transforms = {
        "Affine",
        "BBoxSafeRandomCrop",
        "CenterCrop",
        "CoarseDropout",
        "Crop",
        "CropAndPad",
        "CropNonEmptyMaskIfExists",
        "D4",
        "ElasticTransform",
        "Flip",
        "GridDistortion",
        "GridDropout",
        "HorizontalFlip",
        "Lambda",
        "LongestMaxSize",
        "MaskDropout",
        "MixUp",
        "Morphological",
        "NoOp",
        "OpticalDistortion",
        "PadIfNeeded",
        "Perspective",
        "PiecewiseAffine",
        "PixelDropout",
        "RandomCrop",
        "RandomCropFromBorders",
        "RandomGridShuffle",
        "RandomResizedCrop",
        "RandomRotate90",
        "RandomScale",
        "RandomSizedBBoxSafeCrop",
        "RandomSizedCrop",
        "Resize",
        "Rotate",
        "SafeRotate",
        "ShiftScaleRotate",
        "SmallestMaxSize",
        "Transpose",
        "VerticalFlip",
        "XYMasking",
    }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms
    self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in transforms.transforms)
    self.p = 1.0
    self.transform = transforms


__all__ = [
    'Compose',
    'load_augmentations',
]
