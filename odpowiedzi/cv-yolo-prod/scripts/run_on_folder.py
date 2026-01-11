import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from cv_yolo.config import get_settings
from cv_yolo.predict import YoloDetector
from cv_yolo.visualize import draw_detections


def det_to_dict(d):
    x1, y1, x2, y2 = d.box
    return {
        "label": d.label,
        "confidence": float(d.confidence),
        "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    }


def is_image(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder z obrazami")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    s = get_settings()
    model_name = args.model if args.model else s.yolo_model
    conf = args.conf if args.conf is not None else s.conf_threshold

    in_dir = Path(args.dir)
    if not in_dir.exists():
        raise ValueError(f"dir does not exist: {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(s.out_dir) / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    det = YoloDetector(model_name=model_name)

    images = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and is_image(p)]
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    rows = []
    for p in images:
        image_bgr = cv2.imread(str(p))
        if image_bgr is None:
            continue

        t0 = time.perf_counter()
        dets = det.predict(image_bgr, conf_threshold=conf)
        t1 = time.perf_counter()

        # metryki proste ale uzyteczne
        cnt = len(dets)
        avg_conf = sum(d.confidence for d in dets) / cnt if cnt > 0 else 0.0
        latency_ms = (t1 - t0) * 1000.0

        vis = draw_detections(image_bgr.copy(), dets)

        png_out = out_dir / f"{p.stem}_yolo.png"
        json_out = out_dir / f"{p.stem}_yolo.json"
        cv2.imwrite(str(png_out), vis)

        payload = {
            "image": str(p),
            "model": model_name,
            "conf_threshold": float(conf),
            "detections": [det_to_dict(d) for d in dets],
            "stats": {"detections_count": cnt, "avg_conf": avg_conf, "latency_ms": latency_ms},
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        rows.append(
            {
                "image": str(p),
                "detections_count": cnt,
                "avg_conf": round(avg_conf, 6),
                "latency_ms": round(latency_ms, 3),
            }
        )

        print(f"[INFO] {p.name}: det={cnt} avg_conf={avg_conf:.3f} latency_ms={latency_ms:.2f}")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "detections_count", "avg_conf", "latency_ms"])
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] done. images={len(rows)}")
    print(f"[INFO] metrics_csv={csv_path}")


if __name__ == "__main__":
    main()
