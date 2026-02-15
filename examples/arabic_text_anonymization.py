"""Arabic Text Detection and Anonymization Example.

Detects Arabic text in images and videos, and blurs it while preserving context.
Demonstrates privacy-preserving processing for bilingual documents.
"""

from __future__ import annotations

import cv2
import numpy as np

from privacyguard.detectors.text import ArabicTextDetector, TextDetectorConfig
from privacyguard.utils.arabic import ArabicProcessor, ScriptType


def detect_and_anonymize_text(video_path: str | None = None, webcam: bool = False) -> None:
    """Detect and anonymize Arabic text in video.

    Args:
        video_path: Path to video file
        webcam: Use webcam if True
    """
    # Initialize text detector
    config = TextDetectorConfig(use_paddle_ocr=False)  # Fallback mode without OCR
    detector = ArabicTextDetector(config)

    # Open video source
    if webcam or video_path is None:
        cap = cv2.VideoCapture(0)
        print("Opening webcam - will blur detected text regions (Ctrl+C to exit)...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opening video: {video_path}")

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Resize for processing
            frame = cv2.resize(frame, (640, 480))

            # Detect text regions (will use fallback method if OCR not available)
            regions = detector.detect_text_regions(frame)

            # Draw and blur detected regions
            for region in regions:
                x1, y1, x2, y2 = region["bbox"]
                script = region["script"]

                # Draw bounding box
                color = (0, 255, 0) if script == ScriptType.ARABIC else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw script label
                label = f"{script.value.upper()}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Blur region
                roi = frame[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                frame[y1:y2, x1:x2] = blurred

            # Display stats
            cv2.putText(
                frame,
                f"Detected regions: {len(regions)} | Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display
            cv2.imshow("Arabic Text Detection & Anonymization", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, {len(regions)} text regions detected...")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} total frames")


def process_document_image(image_path: str) -> None:
    """Process document image with Arabic text anonymization.

    Args:
        image_path: Path to document image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    print(f"Processing document: {image_path}")

    config = TextDetectorConfig(use_paddle_ocr=False)
    detector = ArabicTextDetector(config)

    # Resize for processing
    image = cv2.resize(image, (800, 600))

    # Analyze document content
    regions = detector.detect_text_regions(image)

    print(f"Detected {len(regions)} text regions")

    # Categorize by script
    arabic_regions = [r for r in regions if r["script"] == ScriptType.ARABIC]
    latin_regions = [r for r in regions if r["script"] == ScriptType.LATIN]
    mixed_regions = [r for r in regions if r["script"] == ScriptType.MIXED]

    print(
        f"  - Arabic: {len(arabic_regions)}"
        f"  - Latin: {len(latin_regions)}"
        f"  - Mixed: {len(mixed_regions)}"
    )

    # Anonymize only Arabic text
    result = image.copy()
    for region in arabic_regions:
        x1, y1, x2, y2 = region["bbox"]
        roi = result[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (31, 31), 0)
        result[y1:y2, x1:x2] = blurred

    # Display before/after
    comparison = np.hstack([image, result])
    cv2.imshow("Before | After (Arabic anonymized)", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    output_path = image_path.replace(".jpg", "_anonymized.jpg").replace(".png", "_anonymized.png")
    cv2.imwrite(output_path, result)
    print(f"Saved anonymized document: {output_path}")


def demonstrate_script_detection() -> None:
    """Demonstrate script type detection."""
    print("\n=== Script Type Detection Demo ===\n")

    test_strings = [
        ("مرحبا بك في لبنان", "Arabic"),
        ("Bonjour le monde", "French"),
        ("Hello World", "English"),
        ("مرحبا Hello", "Mixed Arabic-English"),
        ("Bienvenue في لبنان", "Mixed French-Arabic"),
    ]

    for text, description in test_strings:
        script = ArabicProcessor.detect_script(text)
        print(f"{description:20} | Text: {text:30} | Script: {script.value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--demo":
            demonstrate_script_detection()
        elif arg == "--webcam":
            detect_and_anonymize_text(webcam=True)
        elif arg.lower().endswith((".jpg", ".jpeg", ".png")):
            process_document_image(arg)
        else:
            detect_and_anonymize_text(video_path=arg)
    else:
        # Default: demonstrate and start webcam
        demonstrate_script_detection()
        print("\nStarting live text detection (Ctrl+C to exit)...")
        detect_and_anonymize_text(webcam=True)
