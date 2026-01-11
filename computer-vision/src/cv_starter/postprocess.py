from typing import List, Tuple

import numpy as np

from cv_starter.pipeline import Detection


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return 0.0 if union == 0 else float(inter / union)


def apply_threshold(raw, threshold: float) -> List[Detection]:
    boxes, scores, labels = raw
    out: List[Detection] = []

    for box, score, label in zip(boxes, scores, labels):
        if float(score) >= threshold:
            x1, y1, x2, y2 = [int(v) for v in box]
            out.append(
                Detection(
                    label=str(label),
                    confidence=float(score),
                    box=(x1, y1, x2, y2),
                )
            )
    return out


def nms(dets: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []

    for d in dets:
        keep = True
        for k in kept:
            if iou(d.box, k.box) >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(d)

    return kept
