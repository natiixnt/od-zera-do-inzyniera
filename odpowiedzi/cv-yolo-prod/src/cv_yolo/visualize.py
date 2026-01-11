from __future__ import annotations

from typing import List

import cv2
import numpy as np

from cv_yolo.predict import Detection


# Wizualizacja jest krytyczna w CV.
# Bez niej latwo uwierzyc w zle wyniki, bo liczby wygladaja okej.
# PNG z boxami to najszybszy debug tool jaki masz.
def draw_detections(image_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = d.box

        # Prostokat detekcji
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Etykieta + confidence nad boxem
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
