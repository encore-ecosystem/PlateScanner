# bbox: (x, y, x, y, confidence, category)

def bbox_iou(bbox_A, bbox_B) -> float:
    x_a, y_a = max(bbox_A[0], bbox_B[0]), max(bbox_A[1], bbox_B[1])
    x_b, y_b = min(bbox_A[2], bbox_B[2]), min(bbox_A[3], bbox_B[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # <deprecated>
    # bbox_a_area = (bbox_A[2] - bbox_A[0] + 1) * (bbox_A[3] - bbox_A[1] + 1)
    # bbox_b_area = (bbox_B[2] - bbox_B[0] + 1) * (bbox_B[3] - bbox_B[1] + 1)

    bbox_a_area = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
    bbox_b_area = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
    try:
        iou = inter_area / (bbox_a_area + bbox_b_area - inter_area)
    except ZeroDivisionError:
        iou = 0
    return iou


__all__ = [
    'bbox_iou'
]
