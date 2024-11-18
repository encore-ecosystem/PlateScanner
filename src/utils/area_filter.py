from src.bbox.bbox_abs import Bbox


def bbox_to_total_area_filter(bboxes: tuple[Bbox, ...], area_threshold: float) -> tuple[Bbox, ...]:
    return tuple([bbox for bbox in bboxes if bbox.area() <= area_threshold])


__all__ = [
    'bbox_to_total_area_filter'
]
