# bbox: (x, y, x, y, confidence, category)

def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

__all__ = [
    'box_area'
]
