"""Identity Document Anonymization Example.

Demonstrates selective anonymization of identity documents:
- Preserves face region for recognition/identification
- Blurs ID numbers, text, and sensitive data
- Handles selective anonymization strategies
"""

from __future__ import annotations

import cv2
import numpy as np

from privacyguard.detectors.document import DocumentConfig, DocumentDetector


def process_document_selective_blur(image_path: str) -> None:
    """Process identity document with selective anonymization.

    Keeps face visible, blurs ID numbers and text.

    Args:
        image_path: Path to document image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    print(f"Processing document: {image_path}")

    # Initialize detector
    config = DocumentConfig(blur_strategy="selective", preserve_face=True)
    detector = DocumentDetector(config)

    # Resize image
    original_height, original_width = image.shape[:2]
    if original_width > 800:
        scale = 800 / original_width
        image = cv2.resize(image, (800, int(original_height * scale)))

    # Detect documents in image
    regions = detector.detect_document_regions(image)
    print(f"Detected {len(regions)} document regions")

    # Process each region
    result = image.copy()
    for i, region in enumerate(regions):
        print(f"  Region {i + 1}: {region['document_type']}, has_face={region['has_face']}")
        result = detector.anonymize_document(result, region)

    # Create comparison (before/after)
    if original_width > 800:
        image = cv2.resize(image, (original_width, original_height))
        result = cv2.resize(result, (original_width, original_height))

    h, w = image.shape[:2]
    if w > 1200:
        scale = 1200 / w
        display_image = cv2.resize(image, (int(w * scale), int(h * scale)))
        display_result = cv2.resize(result, (int(w * scale), int(h * scale)))
    else:
        display_image = image
        display_result = result

    comparison = np.hstack([display_image, display_result])

    # Display
    cv2.imshow("Before | After (Selective Blur - Face Preserved)", comparison)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save results
    output_path = image_path.replace(".jpg", "_anonymized.jpg").replace(".png", "_anonymized.png")
    cv2.imwrite(output_path, result)
    print(f"Saved anonymized document: {output_path}")


def process_document_full_blur(image_path: str) -> None:
    """Process identity document with full anonymization.

    Blurs entire document region.

    Args:
        image_path: Path to document image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    print(f"Processing document (full blur): {image_path}")

    # Initialize detector with full blur
    config = DocumentConfig(blur_strategy="full")
    detector = DocumentDetector(config)

    # Resize if needed
    original_shape = image.shape
    if image.shape[1] > 800:
        scale = 800 / image.shape[1]
        image = cv2.resize(image, (800, int(image.shape[0] * scale)))

    # Detect and anonymize
    regions = detector.detect_document_regions(image)
    result = image.copy()

    for i, region in enumerate(regions):
        print(f"  Blurring region {i + 1}...")
        result = detector.anonymize_document(result, region)

    # Restore original size for display
    result = cv2.resize(result, (original_shape[1], original_shape[0]))
    image = cv2.resize(image, (original_shape[1], original_shape[0]))

    # Display comparison
    h, w = image.shape[:2]
    if w > 1200:
        scale = 1200 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        result = cv2.resize(result, (int(w * scale), int(h * scale)))

    comparison = np.hstack([image, result])
    cv2.imshow("Before | After (Full Blur)", comparison)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save
    output_path = image_path.replace(".jpg", "_anonymized_full.jpg").replace(
        ".png", "_anonymized_full.png"
    )
    cv2.imwrite(output_path, result)
    print(f"Saved fully anonymized document: {output_path}")


def detect_from_camera() -> None:
    """Detect documents in real-time from camera.

    Shows document regions as they're detected.
    """
    print("Starting camera (Ctrl+C to exit)...")
    print("Hold an ID or document up to the camera to see detection")

    cap = cv2.VideoCapture(0)
    config = DocumentConfig()
    detector = DocumentDetector(config)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            # Detect documents
            regions = detector.detect_document_regions(frame)

            # Draw detections
            for region in regions:
                x1, y1, x2, y2 = region["bbox"]
                color = (0, 255, 0) if region["has_face"] else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"Doc (Face: {region['has_face']})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Info text
            cv2.putText(
                frame,
                f"Detected: {len(regions)} documents",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Document Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--camera":
            detect_from_camera()
        elif arg == "--full":
            # Full anonymization
            process_document_full_blur(sys.argv[2] if len(sys.argv) > 2 else "document.jpg")
        else:
            # Default: selective blur
            process_document_selective_blur(arg)
    else:
        print("Usage:")
        print("  python document_anonymization.py <image.jpg>     - Selective blur")
        print("  python document_anonymization.py --full <image>  - Full blur")
        print("  python document_anonymization.py --camera        - Real-time detection")
        print("\nStarting camera detection...")
        detect_from_camera()
