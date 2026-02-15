#!/usr/bin/env python3
"""Batch-process a directory of images.

Usage:
    python examples/batch_images.py \
        --model path/to/model.onnx \
        --input-dir ./photos \
        --output-dir ./photos_anonymized
"""

import argparse
from pathlib import Path

from privacyguard import PrivacyGuard


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch image de-identification")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input-dir", required=True, help="Input directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--method", default="gaussian", choices=["gaussian", "pixelate", "solid"])
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guard = PrivacyGuard(
        model_path=args.model,
        method=args.method,
        conf_threshold=args.conf,
    )

    images = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(images)} images in {input_dir}")

    for img_path in images:
        out_path = output_dir / img_path.name
        detections = guard.process_image(img_path, out_path)
        print(f"  {img_path.name}: {len(detections)} detections")

    print("Done.")


if __name__ == "__main__":
    main()
