import argparse
from pathlib import Path

import cv2

from cv_starter.pipeline import run_pipeline
from cv_starter.visualize import apply_domain_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"cannot read image: {args.image}")

    variants = {
        "clean": image_bgr,
        "dark": apply_domain_shift(image_bgr, darken=0.35),
        "blur": apply_domain_shift(image_bgr, blur=5),
        "noise": apply_domain_shift(image_bgr, noise=15),
    }

    for name, img in variants.items():
        res = run_pipeline(img, threshold=args.threshold)
        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), res.visualized_bgr)
        print(f"{name}: detections={len(res.detections)} -> {out_path}")


if __name__ == "__main__":
    main()
