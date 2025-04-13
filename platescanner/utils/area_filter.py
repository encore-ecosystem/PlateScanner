from cvtk.interfaces import Bbox


def bbox_to_total_area_filter(bboxes: list[Bbox], area_threshold: float) -> list[Bbox]:
    result = [bbox for bbox in bboxes if bbox.area() <= area_threshold]
    return result


__all__ = [
    'bbox_to_total_area_filter'
]
