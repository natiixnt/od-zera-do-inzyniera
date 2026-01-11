from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from cv_starter.preprocess import preprocess
from cv_starter.model import load_model, predict
from cv_starter.postprocess import apply_threshold, nms
from cv_starter.visualize import draw_detections


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class PipelineResult:
    detections: List[Detection]
    visualized_bgr: np.ndarray


_MODEL = None


def run_pipeline(
    image_bgr: np.ndarray,
    threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> PipelineResult:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()

    inp = preprocess(image_bgr)  # (1, 3, H, W) float32

    raw = predict(_MODEL, inp)
    dets = apply_threshold(raw, threshold=threshold)
    dets = nms(dets, iou_threshold=iou_threshold)

    vis = draw_detections(image_bgr.copy(), dets)
    return PipelineResult(detections=dets, visualized_bgr=vis)
