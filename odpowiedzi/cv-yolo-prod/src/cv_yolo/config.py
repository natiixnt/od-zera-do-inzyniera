import os
from dataclasses import dataclass


@dataclass
class Settings:
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.50"))
    out_dir: str = os.getenv("OUT_DIR", "out")


def get_settings() -> Settings:
    return Settings()
