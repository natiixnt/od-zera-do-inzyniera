import argparse
import cv2

from cv_starter.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--min", type=float, default=0.1)
    ap.add_argument("--max", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"cannot read image: {args.image}")

    thr = args.min
    while thr <= args.max + 1e-9:
        result = run_pipeline(image_bgr, threshold=thr)
        print(f"threshold={thr:.2f} -> detections={len(result.detections)}")
        thr += args.step


if __name__ == "__main__":
    main()
