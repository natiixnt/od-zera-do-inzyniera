import argparse
from pathlib import Path

import cv2

from cv_starter.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"cannot read image: {args.image}")

    result = run_pipeline(image_bgr, threshold=args.threshold)
    cv2.imwrite(str(out_path), result.visualized_bgr)

    print(f"[INFO] detections={len(result.detections)}")
    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()
