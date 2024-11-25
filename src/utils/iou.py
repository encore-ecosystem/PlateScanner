from src.bbox.abstract import Bbox


def bbox_iou(bbox_A: Bbox, bbox_B: Bbox) -> float:
    inter_area = bbox_A & bbox_B
    try:
        iou = inter_area / (bbox_A.area() + bbox_B.area() - inter_area)
    except ZeroDivisionError:
        iou = 0
    return iou


__all__ = [
    'bbox_iou'
]
