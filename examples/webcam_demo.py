#!/usr/bin/env python3
"""Live webcam de-identification demo.

Usage:
    python examples/webcam_demo.py --model path/to/yolov8n-face.onnx
"""

import argparse

from privacyguard import PrivacyGuard


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam de-identification")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--method", default="gaussian", choices=["gaussian", "pixelate", "solid"])
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    guard = PrivacyGuard(
        model_path=args.model,
        method=args.method,
        conf_threshold=args.conf,
    )

    print(f"Starting webcam (camera {args.camera}) — press 'q' to quit")
    guard.run(source=args.camera, display=True)


if __name__ == "__main__":
    main()
