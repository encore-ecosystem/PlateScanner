from Levenshtein import distance as levenshtein_distance
from platescanner.utils import bbox_iou
from cvtk.bbox import Bbox

threshold = 0.5

def evaluate_metrics(ground_truth_text: dict[str, tuple[Bbox, str]], recognized_text: dict[str, tuple[Bbox, str]]) -> tuple[dict, dict]:
    lev_scores = {}
    business_scores = {}
    for image_stem in ground_truth_text.keys():
        lev_scores[image_stem] = []
        business_scores[image_stem] = []
        for gt_bbox, gt_text in ground_truth_text[image_stem]:
            for pred_bbox, pred_text in recognized_text.get(image_stem, []):
                if bbox_iou(gt_bbox, pred_bbox) < threshold or gt_text is None:
                    continue

                business_scores[image_stem] += [business_score(gt_text, pred_text)]
                lev_scores[image_stem] += [levenshtein_score(gt_text, pred_text)]

    for image_stem in lev_scores.keys():
        lev_scores[image_stem] = sum(lev_scores[image_stem]) / len(lev_scores[image_stem]) if len(lev_scores[image_stem]) != 0 else 0

    for image_stem in business_scores.keys():
        business_scores[image_stem] = sum(business_scores[image_stem]) if len(business_scores[image_stem]) != 0 else 0

    return lev_scores, business_scores


def levenshtein_score(gt_text: str, pred_text: str) -> float:
    lev_dist = levenshtein_distance(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text))
    return 1 - (lev_dist / max_len)


def business_score(target_text: str, pred_text: str) -> int:
    return 1 if target_text == pred_text else 0


__all__ = [
    'evaluate_metrics',
    'levenshtein_score',
    'business_score',
]
