from typing import List

import cv2
import numpy as np

from cv_starter.pipeline import Detection


def draw_detections(image_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            f"{d.label} {d.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image_bgr


def apply_domain_shift(
    image_bgr: np.ndarray,
    darken: float = 0.0,
    blur: int = 0,
    noise: int = 0,
) -> np.ndarray:
    img = image_bgr.astype(np.float32)

    if darken > 0:
        img = img * (1.0 - darken)

    if blur and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if noise and noise > 0:
        n = np.random.normal(0, noise, size=img.shape).astype(np.float32)
        img = img + n

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
