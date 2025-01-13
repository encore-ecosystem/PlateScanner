from matplotlib.patches import Polygon
from platescanner import BBOX_EDGE_COLOR, BBOX_TEXT_HORIZONTAL_SHIFT, BBOX_TEXT_VERTICAL_SHIFT, BBOX_TEXT_COLOR, BBOX_TEXT_FONTSIZE
from platescanner.bbox import Bbox
import matplotlib.pyplot as plt


def draw_bbox(
        axs: plt.Axes,
        bbox: Bbox,
        text: str,
        text_h_shift: int = BBOX_TEXT_HORIZONTAL_SHIFT,
        text_v_shift: int = BBOX_TEXT_VERTICAL_SHIFT,
        text_color  : str = BBOX_TEXT_COLOR,
        edge_color  : str = BBOX_EDGE_COLOR,
):
    polygone = [[int(point[0]), int(point[1])] for point in bbox.get_poly()]
    axs.add_patch(Polygon(polygone, fill=False, edgecolor=edge_color))
    point = polygone[0]
    axs.text(point[0] + text_h_shift, point[1] + text_v_shift, s=text, color=text_color, fontsize=BBOX_TEXT_FONTSIZE)


__all__ = [
    'draw_bbox'
]
