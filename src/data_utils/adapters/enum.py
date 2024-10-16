from enum import Enum

class AdapterOutputType(str, Enum):
    YOLO = 'YOLO'
    COCO = 'COCO'


__all__ = [
    'AdapterOutputType'
]
