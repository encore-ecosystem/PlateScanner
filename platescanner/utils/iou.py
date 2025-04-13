from cvtk.interfaces import Bbox


def bbox_iou(bbox1: Bbox, bbox2: Bbox) -> float:
    inter_area = bbox1 & bbox2
    try:
        iou = inter_area / (bbox1.area() + bbox2.area() - inter_area)
    except ZeroDivisionError:
        iou = 0
    return iou


__all__ = [
    'bbox_iou'
]
