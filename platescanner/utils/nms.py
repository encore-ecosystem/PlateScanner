from platescanner.utils.iou import bbox_iou
from cvtk.interfaces import Bbox


def nms(bboxes: list[Bbox], iou_threshold: float) -> list[Bbox]:
    result_bboxes = set(bboxes)
    for bbox_A in bboxes:
        to_delete = set()
        for bbox_B in result_bboxes - {bbox_A}:
            if bbox_A.category == bbox_B.category and bbox_iou(bbox_A, bbox_B) > iou_threshold:
                if bbox_A.confidence < bbox_B.confidence:
                    to_delete.add(bbox_A)
                else:
                    to_delete.add(bbox_B)
        result_bboxes -= to_delete
    return list(result_bboxes)


__all__ = [
    'nms'
]
