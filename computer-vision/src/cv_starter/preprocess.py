import cv2
import numpy as np


def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Zwraca tensor w formacie (1, 3, size, size) float32 w zakresie 0..1.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    # BGR -> RGB
    img = img[:, :, ::-1]

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # add batch
    img = np.expand_dims(img, axis=0)

    return img
