import argparse
import json
import time
from pathlib import Path

import cv2

from cv_yolo.config import get_settings
from cv_yolo.predict import YoloDetector
from cv_yolo.visualize import draw_detections


def det_to_dict(d):
    # Zamieniamy dataclass Detection na zwykly dict - idealny do JSON/logow.
    x1, y1, x2, y2 = d.box
    return {
        "label": d.label,
        "confidence": float(d.confidence),
        "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    }


def main():
    # CLI paramy - minimalne, ale wystarcza do pracy i debugowania.
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--json_out", default="")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    # Settings z envow (zobacz src/cv_yolo/config.py)
    s = get_settings()
    model_name = args.model if args.model else s.yolo_model
    conf = args.conf if args.conf is not None else s.conf_threshold

    # 1) wczytaj obraz
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"cannot read image: {args.image}")

    # 2) zaladuj model
    det = YoloDetector(model_name=model_name)

    # 3) inference + latency
    t0 = time.perf_counter()
    dets = det.predict(image_bgr, conf_threshold=conf)
    t1 = time.perf_counter()

    # 4) wizualizacja (debug)
    vis = draw_detections(image_bgr.copy(), dets)

    # 5) output paths
    out_dir = Path(s.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (out_dir / (Path(args.image).stem + "_yolo.png"))
    json_path = Path(args.json_out) if args.json_out else out_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 6) zapis PNG
    cv2.imwrite(str(out_path), vis)

    # 7) zapis JSON - tu jest cala prawda o detekcjach (do analizy/monitoringu)
    payload = {
        "image": str(Path(args.image)),
        "model": model_name,
        "conf_threshold": float(conf),
        "detections": [det_to_dict(d) for d in dets],
        "stats": {
            "detections_count": len(dets),
            "latency_ms": (t1 - t0) * 1000.0,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] model={model_name}")
    print(f"[INFO] conf_threshold={conf}")
    print(f"[INFO] detections={len(dets)}")
    print(f"[INFO] latency_ms={(t1 - t0) * 1000.0:.2f}")
    print(f"[INFO] saved_png={out_path}")
    print(f"[INFO] saved_json={json_path}")


if __name__ == "__main__":
    main()
