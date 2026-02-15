#!/usr/bin/env python3
"""Advanced privacy pipeline with ensemble detection, profiling, and metadata stripping.

This demonstrates production-grade features:
- Multi-model ensemble detection (face + plate simultaneously)
- Real-time performance profiling
- Metadata stripping for compliance
- Confidence-aware adaptive anonymization

Usage:
    python examples/advanced_pipeline.py \
        --face-model yolov8n-face.onnx \
        --plate-model yolov8n-plate.onnx \
        --input video.mp4 \
        --output anonymized.mp4
"""

import argparse
import time
from pathlib import Path

import cv2

from privacyguard import (
    Anonymizer,
    EnsembleConfig,
    EnsembleDetector,
    MetadataStripper,
    Method,
    Profiler,
    PrivacyGuard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced privacy de-identification with ensemble + profiling",
    )
    parser.add_argument("--face-model", required=True, help="Face detection ONNX model")
    parser.add_argument("--plate-model", help="License plate detection ONNX model (optional)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument(
        "--method",
        choices=["gaussian", "pixelate", "solid"],
        default="gaussian",
        help="Anonymization method",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive blur (stronger for high confidence)",
    )
    parser.add_argument("--strip-metadata", action="store_true", help="Strip file metadata")
    args = parser.parse_args()

    # Build ensemble config
    config = EnsembleConfig(
        face_model=args.face_model,
        plate_model=args.plate_model,
        confidence_threshold=0.4,
    )

    # Initialize ensemble detector
    print("🔧 Initializing ensemble detector...")
    try:
        detector = EnsembleDetector(config)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return

    # Initialize anonymizer
    anonymizer = Anonymizer(method=Method(args.method))
    profiler = Profiler()

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    print(f"📹 Processing {total_frames} frames at {fps:.1f} FPS ({w}x{h})")
    print("⏱️  Profiling in progress...")

    profiler.start()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect with profiling
            detect_time, detections = profiler.timeit(detector.detect, frame)

            # Adaptive blur: stronger for high confidence
            if args.adaptive and detections:
                for det in detections:
                    # Adjust kernel size based on confidence
                    kernel_factor = int(20 + det.confidence * 20)
                    if kernel_factor % 2 == 0:
                        kernel_factor += 1
                    anon = Anonymizer(
                        method=Method.GAUSSIAN,
                        gaussian_ksize=(kernel_factor, kernel_factor),
                    )
                    frame = anon.apply(frame, [det])
            else:
                frame = anonymizer.apply(frame, detections)

            # Record metrics
            confidences = [d.confidence for d in detections]
            profiler.record_frame(
                latency_ms=detect_time,
                detection_count=len(detections),
                confidences=confidences,
            )

            writer.write(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                pct = int(100 * frame_count / total_frames)
                print(f"  [{pct:>3}%] {frame_count:>5}/{total_frames} frames | "
                      f"Detections: {len(detections)} | Latency: {detect_time:.1f}ms")

    finally:
        cap.release()
        writer.release()

    # Generate report
    report = profiler.stop()
    print("\n" + str(report))

    # Strip metadata if requested
    if args.strip_metadata:
        print("🔐 Stripping file metadata...")
        temp_file = Path(args.output).with_stem(f"{Path(args.output).stem}_temp")
        MetadataStripper.strip_image(args.output, temp_file)
        temp_file.rename(args.output)
        print("✅ Metadata stripped")

    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
