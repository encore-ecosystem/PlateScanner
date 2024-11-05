from src.utils.box_area import box_area
# bbox: (x, y, x, y, confidence, category)

def bbox_to_total_area_filter(bboxes: list, area_threshold: float, total_area: int) -> list:
    filtered_bboxes = [bbox for bbox in bboxes if box_area(bbox) / total_area <= area_threshold]
    return filtered_bboxes


__all__ = [
    'bbox_to_total_area_filter'
]