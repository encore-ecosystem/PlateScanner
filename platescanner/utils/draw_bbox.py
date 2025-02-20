from pathlib import Path

from PIL import Image
from matplotlib.patches import Polygon
from platescanner import BBOX_EDGE_COLOR, BBOX_TEXT_HORIZONTAL_SHIFT, BBOX_TEXT_VERTICAL_SHIFT, BBOX_TEXT_COLOR, BBOX_TEXT_FONTSIZE
from cvtk.bbox import Bbox
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



def put_info_on_image(image: Image, image_stem: str, output_dir: Path, bboxes: list[tuple[Bbox, str]]):
    width, height = image.size

    fig, axs = plt.subplots()
    axs.imshow(image)
    axs.axis('off')
    fig.patch.set_visible(False)

    for idx, (bbox, text) in enumerate(bboxes):
        draw_bbox(axs=axs, bbox=bbox.to_image_scale(width, height), text=text)

    plt.savefig(output_dir / f"{image_stem}.png", dpi=300)
    plt.close(fig)


__all__ = [
    'draw_bbox',
    'put_info_on_image'
]
