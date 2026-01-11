import numpy as np


def load_model():
    """
    Placeholder.
    W realnym projekcie: torch.load / onnxruntime.InferenceSession / itp.
    """
    return {"name": "dummy-detector"}


def predict(model, inp: np.ndarray):
    """
    Placeholder predykcji.
    Zwraca format: boxes (N,4), scores (N,), labels (N,)
    """
    # Dwa bardzo podobne boxy, zeby bylo co filtrowac NMS-em.
    boxes = np.array([[30, 40, 180, 200], [35, 45, 175, 195]], dtype=np.float32)
    scores = np.array([0.90, 0.55], dtype=np.float32)
    labels = np.array(["person", "person"], dtype=object)
    return boxes, scores, labels
