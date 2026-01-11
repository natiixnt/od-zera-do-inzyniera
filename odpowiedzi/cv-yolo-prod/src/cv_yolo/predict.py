from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2


class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model_name = model_name
        self._model = YOLO(model_name)

        # names zwykle sa w srodku modelu
        self._names = None
        try:
            self._names = self._model.model.names
        except Exception:
            self._names = None

    def predict(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        results = self._model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
        r0 = results[0]

        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()
        cls = r0.boxes.cls.cpu().numpy()

        dets: List[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            kk = int(k)
            if isinstance(self._names, dict) and kk in self._names:
                label = str(self._names[kk])
            else:
                label = str(kk)

            dets.append(
                Detection(
                    label=label,
                    confidence=float(c),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return dets
