import os
from dataclasses import dataclass


# Ten plik trzyma konfiguracje runtime.
# Czemu tak? Bo w realu nie chcesz hardcodowac:
# - nazwy modelu
# - progu confidence
# - folderu output
# tylko wstrzykujesz to z zewnatrz (env, .env, config w k8s itp).
@dataclass
class Settings:
    # Jaki model YOLO ma byc zaladowany (np. yolov8n.pt, yolov8s.pt).
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")

    # Prog confidence - ponizej tego wyniki wywalamy.
    # To jest pokretlo trade-off: FP vs FN.
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.50"))

    # Gdzie zapisujemy outputy (PNG, JSON, CSV).
    out_dir: str = os.getenv("OUT_DIR", "out")


def get_settings() -> Settings:
    # Minimalny "factory" - w przyszlosci mozna tu dodac walidacje.
    return Settings()
