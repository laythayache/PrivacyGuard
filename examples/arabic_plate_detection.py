"""Arabic/Lebanese License Plate Detection Example.

Demonstrates detection and anonymization of Arabic license plates.
Works with both Arabic and Latin script plates commonly found in Lebanon.
"""

from __future__ import annotations

import cv2
import numpy as np

from privacyguard.detectors.arabic_plate import ArabicPlateDetector, PlateConfig


def detect_and_anonymize_plates(video_path: str | None = None, webcam: bool = False) -> None:
    """Detect and anonymize license plates in video.

    Args:
        video_path: Path to video file (None for webcam)
        webcam: Use webcam if True
    """
    # Initialize detector (without ONNX model for demo)
    config = PlateConfig(model_path=None, confidence_threshold=0.5)
    detector = ArabicPlateDetector(config)

    # Open video source
    if webcam or video_path is None:
        cap = cv2.VideoCapture(0)
        print("Opening webcam (Ctrl+C to exit)...")
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

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Demo: Draw rectangular region where plates would be detected
            # (In production, would use actual ONNX detection)
            height, width = frame.shape[:2]

            # Simulate plate region (typically in lower part of vehicle)
            plate_region = [width // 2 - 60, height - 100, width // 2 + 60, height - 50]
            x1, y1, x2, y2 = plate_region

            # Draw detected region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Arabic Plate Region",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Apply blur to simulate anonymization
            roi = frame[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (31, 31), 0)
            frame[y1:y2, x1:x2] = blurred

            # Display
            cv2.imshow("Arabic Plate Detection & Anonymization", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} total frames")


def process_image_with_plates(image_path: str) -> None:
    """Process single image with plate detection.

    Args:
        image_path: Path to image file
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    config = PlateConfig()
    detector = ArabicPlateDetector(config)

    # Resize image
    image = cv2.resize(image, (640, 480))
    height, width = image.shape[:2]

    # Simulate plate detection and anonymization
    plate_region = [width // 2 - 60, height - 100, width // 2 + 60, height - 50]
    x1, y1, x2, y2 = plate_region

    # Blur plate region
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (31, 31), 0)
    image[y1:y2, x1:x2] = blurred

    # Draw annotation
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        "Anonymized Plate",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # Save result
    output_path = image_path.replace(".jpg", "_anonymized.jpg").replace(".png", "_anonymized.png")
    cv2.imwrite(output_path, image)
    print(f"Saved anonymized image: {output_path}")

    # Display
    cv2.imshow("Arabic Plate Anonymization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--webcam":
            detect_and_anonymize_plates(webcam=True)
        else:
            # Process image or video file
            path = sys.argv[1]
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                process_image_with_plates(path)
            else:
                detect_and_anonymize_plates(video_path=path)
    else:
        # Default: use webcam
        detect_and_anonymize_plates(webcam=True)
