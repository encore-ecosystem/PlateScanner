from src.utils.iou import bbox_iou
# bbox: (x, y, x, y, confidence, category)

def nms(bboxes: list, iou_threshold: float) -> list:
    bboxes = tuple(tuple(x) for x in bboxes)
    result_bboxes = set(bboxes)
    for bbox_A in bboxes:
        to_delete = set()
        for bbox_B in result_bboxes - {bbox_A}:
            if bbox_A[5] == bbox_B[5] and bbox_iou(bbox_A, bbox_B) > iou_threshold:
                if bbox_A[4] < bbox_B[4]:
                    to_delete.add(bbox_A)
                else:
                    to_delete.add(bbox_B)
        result_bboxes -= to_delete
    return list(result_bboxes)


__all__ = [
    'nms'
]
