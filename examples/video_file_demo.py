#!/usr/bin/env python3
"""Process a video file and write de-identified output.

Usage:
    python examples/video_file_demo.py \
        --model path/to/model.onnx \
        --input input.mp4 \
        --output output_anonymized.mp4
"""

import argparse

from privacyguard import PrivacyGuard


def main() -> None:
    parser = argparse.ArgumentParser(description="Video file de-identification")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--method", default="gaussian", choices=["gaussian", "pixelate", "solid"])
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--show", action="store_true", help="Display live preview")
    args = parser.parse_args()

    guard = PrivacyGuard(
        model_path=args.model,
        method=args.method,
        conf_threshold=args.conf,
    )

    print(f"Processing {args.input} -> {args.output}")
    guard.process_video(args.input, args.output, show=args.show)
    print("Done.")


if __name__ == "__main__":
    main()
