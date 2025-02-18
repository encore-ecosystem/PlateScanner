import torch
import torchvision.transforms as T
from PIL.Image import Image
from PIL import Image

import albumentations as A
import numpy as np
import cv2

from platescanner import FSR_MODEL_PATH

# LOAD FSR
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(FSR_MODEL_PATH.__str__())
sr.setModel("fsrcnn", 3)


def preprocess_license_plate(plate_image: Image):
    plate_image_np = pil_to_np(plate_image)
    if not(plate_image_np.ndim == 2 or plate_image_np.shape[-1] == 1):
        plate_image_np = A.ToGray(p=1.0, num_output_channels=1)(image=plate_image_np)['image']
    super_resolved = sr.upsample(plate_image_np)
    augmented = A.Compose([
        A.CLAHE(clip_limit=2, tile_grid_size=(1, 1), p=1.0),
        A.Morphological(p=1.0, scale=(4, 4), operation="erosion"),
    ])(image=super_resolved)['image']

    super_resolved_pil = np_to_pil(augmented)
    return super_resolved_pil


class RecognitionModel:
    models = ['parseq', 'parseq_tiny', 'abinet', 'crnn', 'trba', 'vitstr']
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def _get_model(self, name):
        if name in self._model_cache:
            return self._model_cache[name]
        model = torch.hub.load('baudm/parseq', name, pretrained=True).eval().to(self.device)
        self._model_cache[name] = model
        return model

    @torch.inference_mode()
    def __call__(self, model_name, image):
        if image is None:
            return '', []
        model = self._get_model(model_name)
        image = self._preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        # Greedy decoding
        pred = model(image).softmax(-1)
        label, _ = model.tokenizer.decode(pred)
        raw_label, raw_confidence = model.tokenizer.decode(pred, raw=True)
        # Format confidence values
        max_len = 25 if model_name == 'crnn' else len(label[0]) + 1
        conf = list(map('{:0.1f}'.format, raw_confidence[0][:max_len].tolist()))
        return label[0], [raw_label[0][:max_len], conf]

def pil_to_np(image):
    return np.array(image)

def np_to_pil(image_np):
    return Image.fromarray(image_np)

__all__ = [
    'preprocess_license_plate',
    'RecognitionModel',
]
