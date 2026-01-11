from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Jedna detekcja = jeden obiekt wykryty na obrazie.
# Trzymamy to w jednym formacie niezaleznie od tego, jaki model/framework jest pod spodem.
@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) w pikselach


class YoloDetector:
    """
    Wrapper na ultralytics YOLO.

    Cel:
    - ukryc detale biblioteki (results, tensors, cls ids)
    - zwrocic czysty, prosty format list[Detection]
    - miec jedno miejsce, gdzie w razie czego podmieniasz model
      (np. ONNX Runtime zamiast ultralytics) bez przepisywania reszty repo
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        # Import tutaj (a nie na gorze pliku) bo:
        # - przy unit testach albo w srodowisku bez ultralytics
        #   mozesz latwiej mockowac/izolowac
        from ultralytics import YOLO

        self.model_name = model_name
        self._model = YOLO(model_name)

        # YOLO ma mapping id->nazwa klasy (COCO), ale czasem moze byc niedostepny.
        self._names = None
        try:
            self._names = self._model.model.names  # dict[int,str]
        except Exception:
            self._names = None

    def predict(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        # Ultralytics potrafi przyjac numpy array w BGR i sam robi preprocess.
        # My podajemy conf=conf_threshold, zeby od razu odciac bardzo niepewne detekcje.
        results = self._model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
        r0 = results[0]

        # Brak boxow = brak detekcji.
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        # xyxy: (N,4) float
        # conf: (N,) float
        # cls: (N,) float (id klasy)
        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()
        cls = r0.boxes.cls.cpu().numpy()

        dets: List[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            kk = int(k)

            # Zamieniamy id klasy na czytelna nazwe, jesli sie da.
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
